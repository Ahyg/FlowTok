
import logging
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchdiffeq
import random

from diffusion.base_solver import Solver
import numpy as np
from torchvision import transforms


def check_zip(*args):
    args = [list(arg) for arg in args]
    length = len(args[0])
    for arg in args:
        assert len(arg) == length
    return zip(*args)


def kl_divergence(source, target):
    q_raw = source.view(-1)
    p_raw = target.view(-1)

    p = F.softmax(p_raw, dim=0)
    q = F.softmax(q_raw, dim=0)

    
    q_log = torch.log(q)
    kl_div_1 = F.kl_div(q_log, p, reduction='sum')

    return kl_div_1


class TimeStepSampler:
    """
    Abstract class to sample timesteps for flow matching.
    """

    def sample_time(self, x_start):
        # In flow matching, time is in range [0, 1] and 1 indicates the original image; 0 is pure noise
        # this convention is *REVERSE* of diffusion
        raise NotImplementedError

class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        # if self.world_size > 1:
        #     all_image_features, all_text_features = gather_features(
        #         image_features, text_features,
        #         self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

        #     if self.local_loss:
        #         logits_per_image = logit_scale * image_features @ all_text_features.T
        #         logits_per_text = logit_scale * text_features @ all_image_features.T
        #     else:
        #         logits_per_image = logit_scale * all_image_features @ all_text_features.T
        #         logits_per_text = logits_per_image.T
        # else:
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class ResolutionScaledTimeStepSampler(TimeStepSampler):
    def __init__(self, scale: float, base_time_step_sampler: TimeStepSampler):
        self.scale = scale
        self.base_time_step_sampler = base_time_step_sampler

    @torch.no_grad()
    def sample_time(self, x_start):
        base_time = self.base_time_step_sampler.sample_time(x_start)
        # based on eq (23) of https://arxiv.org/abs/2403.03206
        scaled_time = (base_time * self.scale) / (1 + (self.scale - 1) * base_time)
        return scaled_time


class LogitNormalSampler(TimeStepSampler):
    def __init__(self, normal_mean: float = 0, normal_std: float = 1):
        # follows https://arxiv.org/pdf/2403.03206.pdf
        # sample from a normal distribution
        # pass the output through standard logistic function, i.e., sigmoid
        self.normal_mean = float(normal_mean)
        self.normal_std = float(normal_std)

    @torch.no_grad()
    def sample_time(self, x_start):
        x_normal = torch.normal(
            mean=self.normal_mean,
            std=self.normal_std,
            size=(x_start.shape[0],),
            device=x_start.device,
        )
        x_logistic = torch.nn.functional.sigmoid(x_normal)
        return x_logistic


class UniformTimeSampler(TimeStepSampler):
    @torch.no_grad()
    def sample_time(self, x_start):
        # [0, 1] and 1 indicates the original image; 0 is pure noise
        return torch.rand(x_start.shape[0], device=x_start.device)


def _interleave_cond_target(cond, target, num_latent_tokens):
    """Per-frame interleave: each fat frame becomes [cond_ti(L) | target_ti(L)].

    cond, target: [B, T*L, C] (frame-major). Returns [B, T*2L, C] laid out as
    [cond_t0, target_t0, cond_t1, target_t1, ...]. With pos_n_per_frame=2L the DiT
    sees T frames where spatial 0..L-1 = cond, L..2L-1 = target, and cond_ti /
    target_ti share the same temporal (frame) position — the aligned counterpart
    to the block 'token_concat' layout.
    """
    B, TL, C = cond.shape
    L = num_latent_tokens
    T = TL // L
    cond_f = cond.reshape(B, T, L, C)
    target_f = target.reshape(B, T, L, C)
    fat = torch.cat([cond_f, target_f], dim=2)        # [B, T, 2L, C]
    return fat.reshape(B, T * 2 * L, C)


def _deinterleave_target(seq, num_latent_tokens):
    """Inverse for the target half of _interleave_cond_target.

    seq: [B, T*2L, C] -> the target (second) half of each fat frame, [B, T*L, C]
    in the original frame-major target order.
    """
    B, two_TL, C = seq.shape
    L = num_latent_tokens
    T = two_TL // (2 * L)
    fat = seq.reshape(B, T, 2 * L, C)
    return fat[:, :, L:, :].reshape(B, T * L, C)


class FlowMatching(nn.Module):
    def __init__(
        self,
        sigma_min: float = 1e-5,
        sigma_max: float = 1.0,
        timescale: float = 1.0,
        noising_type: str = "none",
        noising_scale: float = 0.1,
        flow_prediction_target: str = "velocity",
        flow_cond_mode: str = "none",
        **kwargs,
    ):
        # LatentDiffusion/DDPM will create too many class variables we do not need
        super().__init__(**kwargs)
        self.time_step_sampler = LogitNormalSampler()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.timescale = timescale
        self.noising_type = noising_type
        self.noising_scale = noising_scale
        # "velocity" (legacy): predict dψ/dt. "radar_tokens": predict x1 (radar
        # tokens) directly; converted back to velocity for the ODE step — a
        # mathematically equivalent reparameterization. Default keeps old runs.
        self.flow_prediction_target = flow_prediction_target
        # "none" (legacy): cond enters as flow x0 (sat tokens are the noise).
        # "token_concat": noise=randn, cond prepended in seq-dim at every nnet
        # call; output's last L tokens are supervised. Default keeps old runs.
        assert flow_cond_mode in ("none", "token_concat", "token_concat_interleaved", "token_concat_modality", "cross_attention")
        self.flow_cond_mode = flow_cond_mode

        self.clip_loss = ClipLoss()

        self.resizer = transforms.Resize(256) # for clip

    def sample_noise(self, x_start):
        # simple IID noise
        return torch.randn_like(x_start, device=x_start.device) * self.sigma_max
    
    def mos(self, err, start_dim=1, con_mask=None):  # mean of square
        if con_mask is not None:
            return (err.pow(2).mean(dim=-1) * con_mask).sum(dim=-1) / con_mask.sum(dim=-1)
        else:
            return err.pow(2).flatten(start_dim=start_dim).mean(dim=-1)

    
    def Xentropy(self, pred, tar, con_mask=None): 
        if con_mask is not None:
            return (nn.functional.cross_entropy(pred, tar, reduction='none') * con_mask).sum(dim=-1) / con_mask.sum(dim=-1)
        else:
            return nn.functional.cross_entropy(pred, tar, reduction='none').mean(dim=-1)
    
    def l2_reg(self, pred, lam = 0.0001): 
        return lam * torch.norm(pred, p=2, dim=(1, 2, 3)) ** 2

    # model forward and prediction
    def forward(
        self,
        x, # image tokens
        nnet,
        cond, # text tokens
        all_config,
        batch_img_clip=None,
        valid_mask=None,
    ):
        timesteps = self.time_step_sampler.sample_time(x)
        return self.p_losses_textVAE_flowtok(
            x, cond, timesteps, nnet, all_config,
            batch_img_clip=batch_img_clip,
            valid_mask=valid_mask,
        )
    
    def p_losses_textVAE_flowtok(
        self,
        x_start,
        cond,
        t,
        nnet,
        all_config,
        batch_img_clip=None,
        valid_mask=None,
    ):
        """
        CrossFLow training for DiT
        """
        contrastive_loss_weight = all_config.losses.contrastive_loss_weight
        kld_loss_weight = all_config.losses.kld_loss_weight

        # 可选：允许在某些任务中关闭 textVAE，直接使用 cond 作为噪声起点（例如 sat tokens -> radar tokens）。
        # 为了兼容旧配置，默认启用 textVAE，只有在 config.use_text_vae_encoder == False 时才跳过。
        use_text_vae_encoder = getattr(all_config, "use_text_vae_encoder", True)

        # token_concat flow: noise = randn (decoupled from cond), cond is
        # prepended in seq-dim at every nnet call. textVAE / KLD / contrastive
        # are unused because cond enters via concat rather than as flow start.
        if self.flow_cond_mode in ("token_concat", "token_concat_interleaved", "token_concat_modality"):
            B_, L_, _ = x_start.shape
            noise = torch.randn_like(x_start)
            x_start_local = x_start.clone()
            null_indicator = torch.zeros(
                B_, dtype=torch.bool, device=x_start.device
            )
            x_noisy = self.psi(t, x=noise, x1=x_start_local)
            if self.flow_cond_mode == "token_concat_interleaved":
                # Aligned layout: per-frame [sat_ti | radar_ti]; supervise radar half.
                L_tok = int(all_config.vq_model.num_latent_tokens)
                inp = _interleave_cond_target(cond, x_noisy, L_tok)   # [B, T*2L, C]
                pred_full = nnet(inp, t=t, null_indicator=null_indicator)[0]
                prediction = _deinterleave_target(pred_full, L_tok)   # [B, T*L, C]
            else:
                inp = torch.cat([cond, x_noisy], dim=1)         # [B, 2L, C]
                pred_full = nnet(inp, t=t, null_indicator=null_indicator)[0]
                prediction = pred_full[:, L_:, :]               # supervise radar half
            if self.flow_prediction_target == "radar_tokens":
                fm_target = x_start_local
            else:
                fm_target = self.Dt_psi(t, x=noise, x1=x_start_local)
            if valid_mask is not None:
                err = (prediction - fm_target).pow(2).mean(dim=-1)  # [B, L]
                loss_diff = (err * valid_mask).sum() / valid_mask.sum().clamp(min=1)
            else:
                loss_diff = self.mos(prediction - fm_target)
            zero = x_start.new_zeros([])
            return loss_diff, {
                'diff_loss': loss_diff,
                'contrastive_loss': zero,
                'kld_loss': zero,
            }

        if self.flow_cond_mode == "cross_attention":
            B_, L_, _ = x_start.shape
            noise = torch.randn_like(x_start)
            x_start_local = x_start.clone()
            null_indicator = torch.zeros(B_, dtype=torch.bool, device=x_start.device)
            x_noisy = self.psi(t, x=noise, x1=x_start_local)
            prediction = nnet(x_noisy, t=t, null_indicator=null_indicator, context=cond)[0]
            if self.flow_prediction_target == "radar_tokens":
                fm_target = x_start_local
            else:
                fm_target = self.Dt_psi(t, x=noise, x1=x_start_local)
            if valid_mask is not None:
                err = (prediction - fm_target).pow(2).mean(dim=-1)
                loss_diff = (err * valid_mask).sum() / valid_mask.sum().clamp(min=1)
            else:
                loss_diff = self.mos(prediction - fm_target)
            zero = x_start.new_zeros([])
            return loss_diff, {
                'diff_loss': loss_diff,
                'contrastive_loss': zero,
                'kld_loss': zero,
            }

        if use_text_vae_encoder:
            x0, mu, log_var = nnet(cond, text_encoder=True)
        else:
            # 直接使用 cond 作为 latent 起点；mu/log_var 仅为占位，以便后续逻辑安全运行。
            x0 = cond
            mu = torch.zeros_like(cond)
            log_var = torch.zeros_like(cond)

        # Initialize losses to zero for the common case where the weights are 0.
        # This avoids UnboundLocalError when contrastive or KLD losses are disabled.
        contrastive_loss = x0.new_zeros([])
        kld_loss = x0.new_zeros([])

        if self.noising_type != "none":
            random_noise = torch.randn_like(x0)
            if self.noising_type == "random":
                noising_scale = random.random()
            elif self.noising_type == "constant":
                noising_scale = self.noising_scale
            x0 = x0 + noising_scale * random_noise

        ############ loss for Text VE
        if use_text_vae_encoder and contrastive_loss_weight > 0:
            B, L, C = cond.shape
            cond_projected, t2t_temperature = nnet(cond, text_projector = True)

            x0_flat = x0.reshape(B, -1)  # Shape: [B, L*C]
            cond_flat = cond_projected.reshape(B, -1)  # Shape: [B, L*C]
            x0_norm = F.normalize(x0_flat, dim=-1)  # Shape: [B, L*C]
            cond_norm = F.normalize(cond_flat, dim=-1)  # Shape: [B, L*C]

            logit_scale = t2t_temperature.exp()
            logits_per_x0 = logit_scale * x0_norm @ cond_norm.T  # Shape: [B, L, L]
            logits_per_cond = logit_scale * cond_norm @ x0_norm.T  # Shape: [B, L, L]

            targets = torch.arange(B, device=x0_norm.device)
            x0_loss = F.cross_entropy(logits_per_x0, targets)
            cond_loss = F.cross_entropy(logits_per_cond, targets)
            contrastive_loss = (x0_loss + cond_loss) * contrastive_loss_weight / 2

        if use_text_vae_encoder and kld_loss_weight > 0:
            kld_loss = -0.5 * torch.sum(1 + log_var - (0.3 * mu) ** 6 - log_var.exp())
            kld_loss = kld_loss * kld_loss_weight
        
        ############ loss for FM
        noise = x0.reshape(x_start.shape)

        # Clone x_start before any in-place modification so the caller's tensor
        # (e.g. radar_tokens used by adapter_out recon loss) is not corrupted.
        x_start = x_start.clone()

        null_indicator = torch.from_numpy(np.array([random.random() < all_config.nnet.model_args.cfg_indicator for _ in range(x_start.shape[0])])).to(x_start.device)
        if null_indicator.sum() <= 1:
            null_indicator[null_indicator==True] = False
        else:
            target_null = x_start[null_indicator]
            target_null = torch.cat((target_null[1:], target_null[:1]))
            x_start[null_indicator] = target_null
        
        x_noisy = self.psi(t, x=noise, x1=x_start)
        prediction = nnet(x_noisy, t = t, null_indicator = null_indicator)[0]

        if self.flow_prediction_target == "radar_tokens":
            fm_target = x_start                                  # predict x1
        else:
            fm_target = self.Dt_psi(t, x=noise, x1=x_start)      # legacy velocity

        # Optional mask for variable-length sequences (e.g. v2v with padding)
        if valid_mask is not None:
            # valid_mask: [B, L]; err: [B, L, C] -> flatten to [B, L] for mask
            err = (prediction - fm_target).pow(2).mean(dim=-1)  # [B, L]
            loss_diff = (err * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        else:
            loss_diff = self.mos(prediction - fm_target)

        loss = loss_diff + contrastive_loss + kld_loss
        loss_dict = {'diff_loss': loss_diff, 'contrastive_loss': contrastive_loss, 'kld_loss': kld_loss}

        return loss, loss_dict
        

    ## flow matching specific functions
    def psi(self, t, x, x1):
        assert (
            t.shape[0] == x.shape[0]
        ), f"Batch size of t and x does not agree {t.shape[0]} vs. {x.shape[0]}"
        assert (
            t.shape[0] == x1.shape[0]
        ), f"Batch size of t and x1 does not agree {t.shape[0]} vs. {x1.shape[0]}"
        assert t.ndim == 1
        t = self.expand_t(t, x)
        return (t * (self.sigma_min / self.sigma_max - 1) + 1) * x + t * x1

    def Dt_psi(self, t: torch.Tensor, x: torch.Tensor, x1: torch.Tensor):
        assert x.shape[0] == x1.shape[0]
        return (self.sigma_min / self.sigma_max - 1) * x + x1

    def expand_t(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t_expanded = t
        while t_expanded.ndim < x.ndim:
            t_expanded = t_expanded.unsqueeze(-1)
        return t_expanded.expand_as(x)




class ODEEulerFlowMatchingSolver(Solver):
    """
    ODE Solver for Flow matching that uses an Euler discretization
    Supports number of time steps at inference
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_size_type = kwargs.get("step_size_type", "step_in_dsigma")
        assert self.step_size_type in ["step_in_dsigma", "step_in_dt"]
        self.sample_timescale = 1.0 - 1e-5
        # x1(radar_tokens)-prediction sampler:
        #   "fixed_x0" (legacy default): velocity = x1_hat - x0_start(frozen) -> Euler telescopes to
        #       mean_i x1_hat(t_i) (time-AVERAGE of predictions; over-smooths). Kept for back-compat.
        #   "canonical": numerically-stable data-prediction step = the canonical FM Euler
        #       v=(x1_hat-x_t)/(1-t), written as x <- x1_hat + ((1-t_next)/(1-t_cur))*(x - x1_hat),
        #       converging to the sharp x1_hat(t~1). (NOT diffusion DDIM; just the same re-projection form.)
        self.x1_sampler = kwargs.get("x1_sampler", "fixed_x0")
        assert self.x1_sampler in ["fixed_x0", "canonical"]
        # EXPERIMENT (opt-in, eval-only): one-shot clean-radar readout. If x1_snap_t >= 0 and
        # prediction_target=="radar_tokens", at the first model-eval node with t_i >= x1_snap_t we
        # OUTPUT the model's x1_hat (predicted clean radar) directly and stop -- replacing the
        # remaining Euler integration. Avoids the fixed_x0 time-averaging over-smoothing by reading
        # the model's sharpest single guess at a late noise level. x1_snap_t<0 (default) = disabled
        # (exact legacy fixed_x0 behavior). x1_snap_t~0.99 ≈ "snap at the last step".
        self.x1_snap_t = float(kwargs.get("x1_snap_t", -1.0))

    @torch.no_grad()
    def sample_euler(
        self,
        x_T,
        unconditional_guidance_scale,
        has_null_indicator,
        t=[0, 1.0],
        **kwargs,
    ):
        """
        Euler solver for flow matching.
        Based on https://github.com/VinAIResearch/LFM/blob/main/sampler/karras_sample.py
        """
        t = torch.tensor(t)
        t = t * self.sample_timescale
        sigma_min = 1e-5
        sigma_max = 1.0
        sigma_steps = torch.linspace(
            sigma_min, sigma_max, self.num_time_steps + 1, device=x_T.device
        )
        discrete_time_steps_for_step = torch.linspace(
            t[0], t[1], self.num_time_steps + 1, device=x_T.device
        )
        discrete_time_steps_to_eval_model_at = torch.linspace(
            t[0], t[1], self.num_time_steps, device=x_T.device
        )

        x0_start = x_T.clone()           # fixed flow start; needed for x1->v reparam
        prediction_target = getattr(self, "prediction_target", "velocity")
        flow_cond_mode = getattr(self, "flow_cond_mode", "none")
        cond_tokens = getattr(self, "cond_tokens", None)
        cond_L = getattr(self, "cond_num_latent_tokens", None)
        is_token_concat = flow_cond_mode in ("token_concat", "token_concat_interleaved", "token_concat_modality")
        if is_token_concat:
            assert cond_tokens is not None, "token_concat requires cond_tokens"
            L_ = x_T.shape[1]
            if flow_cond_mode == "token_concat_interleaved":
                assert cond_L is not None, \
                    "token_concat_interleaved requires cond_num_latent_tokens"
        is_cross_attn = flow_cond_mode == "cross_attention"
        if is_cross_attn:
            assert cond_tokens is not None, "cross_attention requires cond_tokens"
        for i in range(self.num_time_steps):
            t_i = discrete_time_steps_to_eval_model_at[i]
            if is_token_concat:
                null_ind = torch.zeros(
                    x_T.shape[0], dtype=torch.bool, device=x_T.device
                )
                if flow_cond_mode == "token_concat_interleaved":
                    inp = _interleave_cond_target(cond_tokens, x_T, cond_L)
                    out_full = self.model(
                        inp, t=t_i.repeat(x_T.shape[0]), null_indicator=null_ind
                    )[-1]
                    model_out = _deinterleave_target(out_full, cond_L)
                else:
                    inp = torch.cat([cond_tokens, x_T], dim=1)
                    out_full = self.model(
                        inp, t=t_i.repeat(x_T.shape[0]), null_indicator=null_ind
                    )[-1]
                    model_out = out_full[:, L_:, :]
            elif is_cross_attn:
                null_ind = torch.zeros(x_T.shape[0], dtype=torch.bool, device=x_T.device)
                model_out = self.model(
                    x_T, t=t_i.repeat(x_T.shape[0]), null_indicator=null_ind, context=cond_tokens
                )[-1]
            else:
                model_out = self.get_model_output_flowtok(
                    x_T,
                    has_null_indicator = has_null_indicator,
                    t_continuous = t_i.repeat(x_T.shape[0]),
                    unconditional_guidance_scale = unconditional_guidance_scale,
                )
            if (prediction_target == "radar_tokens" and self.x1_snap_t >= 0.0
                    and float(t_i) >= self.x1_snap_t):
                # one-shot readout: output the predicted clean radar x1_hat at this late t,
                # skipping the rest of the Euler integration (see __init__ note).
                x_T = model_out
                break
            if prediction_target == "radar_tokens":
                velocity = (sigma_min / sigma_max - 1.0) * x0_start + model_out
            else:
                velocity = model_out
            if self.step_size_type == "step_in_dsigma":
                step_size = sigma_steps[i + 1] - sigma_steps[i]
            elif self.step_size_type == "step_in_dt":
                step_size = (
                    discrete_time_steps_for_step[i + 1]
                    - discrete_time_steps_for_step[i]
                )
            x_T = x_T + velocity * step_size

        intermediates = None
        return x_T, intermediates

    @torch.no_grad()
    def sample(
        self,
        *args,
        **kwargs,
    ):
        assert kwargs.get("ucg_schedule", None) is None
        assert kwargs.get("skip_type", None) is None
        assert kwargs.get("dynamic_threshold", None) is None
        assert kwargs.get("x0", None) is None
        assert kwargs.get("x_T") is not None
        assert kwargs.get("score_corrector", None) is None
        assert kwargs.get("normals_sequence", None) is None
        assert kwargs.get("callback", None) is None
        assert kwargs.get("quantize_x0", False) is False
        assert kwargs.get("eta", 0.0) == 0.0
        assert kwargs.get("mask", None) is None
        assert kwargs.get("noise_dropout", 0.0) == 0.0

        self.num_time_steps = kwargs.get("sample_steps")
        self.x_T_uncon = kwargs.get("x_T_uncon")
        # defensive: honor x1_sampler if passed to sample() too (constructor value preserved otherwise)
        self.x1_sampler = kwargs.get("x1_sampler", getattr(self, "x1_sampler", "fixed_x0"))
        self.x1_snap_t = float(kwargs.get("x1_snap_t", getattr(self, "x1_snap_t", -1.0)))
        self.prediction_target = kwargs.get("prediction_target", "velocity")
        self.flow_cond_mode = kwargs.get("flow_cond_mode", "none")
        self.cond_tokens = kwargs.get("cond_tokens", None)
        self.cond_num_latent_tokens = kwargs.get("cond_num_latent_tokens", None)

        samples, intermediates = super().sample(
            *args,
            sampling_method=self.sample_euler,
            do_make_schedule=False,
            **kwargs,
        )
        return samples, intermediates


class ODEFlowMatchingSolver(Solver):
    """
    ODE Solver for Flow matching that uses `dopri5`
    Does not support number of time steps based control
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_timescale = 1.0 - 1e-5

    # sampling for inference
    @torch.no_grad()
    def sample_transport(
        self,
        x_T,
        unconditional_guidance_scale,
        has_null_indicator,
        t=[0, 1.0],
        ode_opts={},
        **kwargs,
    ):
        num_evals = 0
        t = torch.tensor(t, device=x_T.device)
        if "options" not in ode_opts:
            ode_opts["options"] = {}
        ode_opts["options"]["step_t"] = [self.sample_timescale + 1e-6]

        def ode_func(t, x_T):
            nonlocal num_evals
            num_evals += 1
            model_output = self.get_model_output_dimr(
                x_T,
                has_null_indicator = has_null_indicator,
                t_continuous = t.repeat(x_T.shape[0]),
                unconditional_guidance_scale = unconditional_guidance_scale,
            )
            return model_output

        z = torchdiffeq.odeint(
            ode_func,
            x_T,
            t * self.sample_timescale,
            **{"atol": 1e-5, "rtol": 1e-5, "method": "dopri5", **ode_opts},
        )
        # first dimension of z contains solutions to different timepoints
        # we only need the last one (corresponding to t=1, i.e., image)
        z = z[-1]
        intermediates = None
        return z, intermediates

    @torch.no_grad()
    def sample(
        self,
        *args,
        **kwargs,
    ):
        assert kwargs.get("ucg_schedule", None) is None
        assert kwargs.get("skip_type", None) is None
        assert kwargs.get("dynamic_threshold", None) is None
        assert kwargs.get("x0", None) is None
        assert kwargs.get("x_T") is not None
        assert kwargs.get("score_corrector", None) is None
        assert kwargs.get("normals_sequence", None) is None
        assert kwargs.get("callback", None) is None
        assert kwargs.get("quantize_x0", False) is False
        assert kwargs.get("eta", 0.0) == 0.0
        assert kwargs.get("mask", None) is None
        assert kwargs.get("noise_dropout", 0.0) == 0.0
        samples, intermediates = super().sample(
            *args,
            sampling_method=self.sample_transport,
            do_make_schedule=False,
            **kwargs,
        )
        return samples, intermediates