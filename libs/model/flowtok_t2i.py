import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp

import ml_collections
import torch.utils.checkpoint
import open_clip

from .trans_autoencoder import FlowEncoder, Adaptor


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    CrossFlow: update it for CFG with indicator
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)

    def forward(self, labels):
        embeddings = self.embedding_table(labels.int())
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries from x, keys/values from context."""
    def __init__(self, hidden_size, num_heads, qkv_bias=True):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.kv = nn.Linear(hidden_size, hidden_size * 2, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, context):
        B, Nq, C = x.shape
        Nk = context.shape[1]
        q = self.q(x).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, Nk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, Nq, C)
        return self.proj(out)


class SatEncoderLayer(nn.Module):
    """One pre-norm self-attention + MLP block for the satellite context encoder.
    No adaLN, no timestep conditioning, no cross-attn — the satellite condition is
    clean input, independent of the diffusion timestep (Arm 8)."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=int(hidden_size * mlp_ratio),
                       act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SatContextEncoder(nn.Module):
    """Timestep-independent self-attention encoder that refines satellite K/V
    tokens before per-block cross-attention (Arm 8). Full self-attention over the
    whole sat sequence (T*L tokens) => spatiotemporal, not axial."""
    def __init__(self, hidden_size, num_heads, depth=6, mlp_ratio=4.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [SatEncoderLayer(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_cross_attn=False, **block_kwargs):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        n_mod = 9 if use_cross_attn else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, n_mod * hidden_size, bias=True)
        )
        if use_cross_attn:
            self.norm_ca = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.cross_attn = CrossAttention(hidden_size, num_heads)

    def forward(self, x, c, context=None):
        return torch.utils.checkpoint.checkpoint(self._forward, x, c, context, use_reentrant=False)

    def _forward(self, x, c, context=None):
        if self.use_cross_attn:
            (shift_msa, scale_msa, gate_msa,
             shift_ca, scale_ca, gate_ca,
             shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(9, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_ca.unsqueeze(1) * self.cross_attn(modulate(self.norm_ca(x), shift_ca, scale_ca), context)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class FlowTok(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        config,
        num_latent_tokens=77,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=2, # for cfg indicator
    ):
        super().__init__()
        # cond_concat_channels: M3 token-diffusion channel-concats sat tokens
        # with the noisy radar tokens, doubling DiT input width. Default False =
        # legacy (in == out == config.channels); old ckpts/configs unaffected.
        self.out_channels = config.channels
        self.in_channels = (config.channels * 2
                            if getattr(config, "cond_concat_channels", False)
                            else config.channels)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_latent_tokens = num_latent_tokens

        self.x_embedder = nn.Linear(self.in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # Some legacy configs do not define `use_t2t_temperature`.
        # Default to False for backward compatibility.
        self.use_t2t_temperature = getattr(config, "use_t2t_temperature", False)
        if self.use_t2t_temperature:
            self.t2t_temperature = nn.Parameter(torch.log(torch.tensor(1/0.07)))
        else:
            self.t2t_temperature = None

        # NOTE: 原版实现中 pos_embed 是一个固定长度为 num_latent_tokens 的参数，
        # 这会限制只能处理固定长度的 token（单帧）。为了支持可变帧数（视频），
        # 这里改为在前向时按当前序列长度动态生成 1D sin-cos 位置编码。
        # 保留 num_latent_tokens 仅作为默认长度配置，不再用作严格限制。
        self.pos_embed = None
        # pos_n_per_frame controls the "tokens per frame" used by the pos embed.
        # Default = num_latent_tokens (77). seqconcat mode sets it to 2*num_latent_tokens
        # so spatial_pos distinguishes modalities inside each fat frame.
        self.pos_n_per_frame = num_latent_tokens

        # Arm 6: opt-in 3-axis pos-emb (spatial D/2 + temporal D/4 + modality D/4).
        # Default False == legacy 2-axis (spatial+temporal). No new params either way.
        self.use_modality_pos_emb = getattr(config, "use_modality_pos_emb", False)

        # Arm 7: opt-in cross-attention conditioning on satellite tokens.
        # Default False == legacy self-attn-only blocks; no new params when off.
        self.use_cross_attention = getattr(config, "use_cross_attention", False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_cross_attn=self.use_cross_attention)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        if self.use_cross_attention:
            self.context_embedder = nn.Linear(config.channels, hidden_size, bias=True)
            self.sat_ctx_layers = getattr(config, "sat_context_encoder_layers", 0)
            self.use_sat_context_encoder = self.sat_ctx_layers > 0
            if self.use_sat_context_encoder:
                self.sat_context_encoder = SatContextEncoder(
                    hidden_size, num_heads, depth=self.sat_ctx_layers, mlp_ratio=mlp_ratio)
        self.initialize_weights()

        self.context_encoder = FlowEncoder(d_model=config.clip_dim, N=config.textVAE.num_blocks,
                                            head_num=config.textVAE.num_attention_heads, d_ff=config.textVAE.hidden_dim, 
                                            latten_size=config.channels * 2, dropout=config.textVAE.dropout_prob, last_norm=False)
        
        self.context_projector = nn.Sequential(
            nn.Linear(config.clip_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, config.channels),
        )
            
        if config.textVAE.clip_loss_weight > 0.0:
            self.open_clip, _, self.open_clip_preprocess = open_clip.create_model_and_transforms('ViT-L-16-SigLIP-256', pretrained=None)
            self.open_clip_output = Adaptor(input_dim=1024, tar_dim=num_latent_tokens*config.channels)

            del self.open_clip.text
            del self.open_clip.logit_bias


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Make FlowTok compatible with legacy checkpoints that contain
        `t2t_temperature` and `pos_embed` parameters.
        """
        # Handle common wrapper keys first (e.g., {"state_dict": ...}).
        if isinstance(state_dict, dict):
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

        # Shallow copy so that we don't modify caller's dict.
        state_dict = dict(state_dict)

        # Legacy `t2t_temperature`: only load it when the current config uses it.
        if not getattr(self, "use_t2t_temperature", False) and "t2t_temperature" in state_dict:
            state_dict.pop("t2t_temperature")

        # Legacy `pos_embed`: it is now computed on the fly in `_build_pos_embed`,
        # so we drop it from old checkpoints to avoid "unexpected key" errors.
        if self.pos_embed is None and "pos_embed" in state_dict:
            state_dict.pop("pos_embed")

        return super().load_state_dict(state_dict, strict=strict)

    def _build_pos_embed(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """
        Build positional embedding with explicit temporal (frame) signal for V2V.
        - Spatial: token index within each frame (0 .. num_latent_tokens-1), repeated per frame.
        - Temporal: frame index (0 .. T-1), same value for all 77 tokens in that frame.
        So the model gets a clear "which frame" signal for temporal modeling.
        For I2I (L=77), temporal is all 0; for V2V (L=T*77), temporal distinguishes frames.
        """
        if getattr(self, "use_modality_pos_emb", False):
            L = seq_len
            Ltok = self.num_latent_tokens
            half = L // 2
            assert half % Ltok == 0, "seq_len//2 must be divisible by num_latent_tokens"
            idx = np.arange(L, dtype=np.float32)
            spatial_pos = idx % Ltok
            temporal_pos = (idx % half) // Ltok
            modality_pos = idx // half
            D = self.hidden_size
            d_sp = (D // 2) - ((D // 2) % 2)
            d_tp = (D // 4) - ((D // 4) % 2)
            d_mod = D - d_sp - d_tp
            assert d_mod % 2 == 0, f"modality dim {d_mod} must be even"
            sp = get_1d_sincos_pos_embed_from_grid(d_sp, spatial_pos)
            tp = get_1d_sincos_pos_embed_from_grid(d_tp, temporal_pos)
            md = get_1d_sincos_pos_embed_from_grid(d_mod, modality_pos)
            pos = np.concatenate([sp, tp, md], axis=-1)
            return torch.from_numpy(pos).to(device=device, dtype=dtype).unsqueeze(0)
        L = seq_len
        n_per_frame = getattr(self, "pos_n_per_frame", self.num_latent_tokens)
        # Within-frame spatial position: 0,1,...,n_per_frame-1 repeated each frame.
        # In seqconcat mode n_per_frame=2L so the first L positions (0..L-1) label
        # sat tokens and the second L (L..2L-1) label lgt tokens — modality is
        # encoded directly in spatial_pos.
        spatial_pos = np.arange(L, dtype=np.float32) % n_per_frame
        # Frame index: 0,0,...,0, 1,1,...,1, ... (n_per_frame same values per frame)
        temporal_pos = (np.arange(L, dtype=np.float32) // n_per_frame)

        spatial_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size // 2, spatial_pos)   # (L, D)
        temporal_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size // 2, temporal_pos)  # (L, D)
        #pos_embed = torch.from_numpy(spatial_embed + temporal_embed).to(device=device, dtype=dtype).unsqueeze(0)  # [1, L, D]
        pos_embed = torch.from_numpy(np.concatenate([spatial_embed, temporal_embed], axis=-1)).to(device=device, dtype=dtype).unsqueeze(0)  # [1, L, D]
        return pos_embed

    def _forward(self, x, t, null_indicator, context=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        context: (N, L_ctx, C_in) optional satellite token context for cross-attention (Arm 7)
        """
        # x: [B, L, C_in]; L = 77 (I2I) or T*77 (V2V)
        B, L, _ = x.shape
        x = self.x_embedder(x)  # [B, L, D]

        # Positional embedding: spatial (within-frame token index) + temporal (frame index)
        pos_embed = self._build_pos_embed(seq_len=L, device=x.device, dtype=x.dtype)
        x = x + pos_embed  # (N, L, D)

        ctx = None
        if self.use_cross_attention and context is not None:
            ctx = self.context_embedder(context)
            ctx = ctx + self._build_pos_embed(seq_len=ctx.shape[1], device=ctx.device, dtype=ctx.dtype)
            if getattr(self, "use_sat_context_encoder", False):
                ctx = self.sat_context_encoder(ctx)

        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(null_indicator)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c, ctx) if self.use_cross_attention else block(x, c)
        x = self.final_layer(x, c)                # (N, T, out_channels)
        return [x]
    
    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def _text_encoder(self, condition_context):
        # [B, 77, 768] -> [B, 77, 16]
        output = self.context_encoder(condition_context)
        mu, log_var = torch.chunk(output, 2, dim=-1)        
        z = self._reparameterize(mu, log_var)
        return [z, mu, log_var]

    def _text_projector(self, condition_context):
        z = self.context_projector(condition_context)

        return z, self.t2t_temperature
    
    def _img_clip(self, image_input):
        image_latent = self.open_clip.encode_image(image_input)
        image_latent = self.open_clip_output(image_latent)

        return image_latent, self.open_clip.logit_scale
    
    def forward(self, x, t = None, text_encoder=False, text_projector=False, image_clip=False, null_indicator=None, context=None):
        if text_encoder:
            return self._text_encoder(condition_context = x)
        elif text_projector:
            return self._text_projector(condition_context = x)
        elif image_clip:
            return self._img_clip(image_input = x)
        else:
            return self._forward(x=x, t=t, null_indicator=null_indicator, context=context)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def FlowTok_H(config, **kwargs):
    return FlowTok(config=config, depth=36, hidden_size=1280, num_heads=20, **kwargs)

def FlowTok_XL(config, **kwargs):
    return FlowTok(config=config, depth=28, hidden_size=1152, num_heads=16, **kwargs)

def FlowTok_L(config, **kwargs):
    return FlowTok(config=config, depth=24, hidden_size=1024, num_heads=16, **kwargs)

def FlowTok_B(config, **kwargs):
    return FlowTok(config=config, depth=12, hidden_size=768, num_heads=12, **kwargs)

def FlowTok_S(config, **kwargs):
    return FlowTok(config=config, depth=12, hidden_size=384, num_heads=6, **kwargs)