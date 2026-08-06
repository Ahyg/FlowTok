"""i2i ABLATION 1 "pixfact" — PIXEL space + factorized DiT + flow-matching (T=1).

i2i mirror of the pixfact-v2v arm: single-frame (T=1) satellite -> radar. Removes the
learned FlowTiTok tokenizer/detokenizer: a parameter-free, exactly-invertible pixel
patchify (P=8) replaces encode/decode, so the SAME factorized DiT + flow-matching
operates directly on raw radar/sat pixel patches. i2i counterpart of the token-space
xattn-i2i arm, exactly as pixfact-v2v is to fact-v2v.

P=8 on 128x128 SINGLE frame: L=(128/8)^2=256 patches/frame, seq=T*L=1*256=256.
  radar patch dim = 1*8^2 = 64  -> model.channels=64      (UNCHANGED from v2v: per-frame)
  sat   patch dim = 11*8^2 = 704 -> model.cond_channels=704 (UNCHANGED from v2v: per-frame)
num_latent_tokens stays 256 (patches PER FRAME); at T=1 seq=256 and the sampler
seq-reshape infers T = 256 // 256 = 1. Pixels mapped [0,1]->[-1,1] in PixelPatchifier.
Fixed-600k comparison budget. Everything else identical to the pixfact-B v2v arm.
"""
import ml_collections
from dataclasses import dataclass


@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


model = Args(
    learn_sigma=False,
    channels=64,            # radar pixel-patch dim = C_radar * P^2 = 1 * 8^2  (per-frame; T-independent)
    cond_channels=704,      # sat pixel-patch dim   = C_sat   * P^2 = 11 * 8^2 (cross-attn KV; per-frame)
    use_t2i=False,
    clip_dim=16,
    num_clip_token=256,
    num_latent_tokens=256,  # patches/frame L = (128/8)^2; sets factorized n_per_frame + pos (STAYS 256 at T=1)
    gradient_checking=False,
    cfg_indicator=0.0,
    use_cross_attention=True,
    use_factorized_attn=True,
    noising_type="none",
    noising_scale=0.1,
    textVAE=Args(
        num_blocks=6,
        hidden_dim=256,
        num_attention_heads=4,
        dropout_prob=0.1,
        clip_loss_weight=0.0,
        align_quantized=False,
        use_pretrained=False,
        tokenizer_checkpoint="",
        freeze_encoder=False,
    ),
)


def d(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 1234

    # ===== ABLATION 1: pixel space (no learned tokenizer) =====
    config.pixel_space = True
    config.patch_size = 8

    # Tokenizer checkpoints are IGNORED under pixel_space (PixelPatchifier is parameter-free).
    config.sat_tokenizer_checkpoint = ""
    config.radar_tokenizer_checkpoint = ""

    config.train = d(
        n_steps=600_000,            # fixed-600k comparison budget
        batch_size=64,              # matches token i2i siblings (xattn-i2i / m8align-i2i); v2v pixfact was bs=8
        log_interval=100,
        eval_interval=2_000,
        save_interval=50_000,
        n_samples_eval=4,
        val_max_batches=64,
    )

    config.optimizer = d(
        name="adamw",
        lr=4e-4,
        weight_decay=0.03,
        betas=(0.9, 0.95),
    )

    config.lr_scheduler = d(
        name="customized",
        warmup_steps=5000,
    )

    config.vq_model = d(
        deterministic=False,
        token_size=16,
        vit_enc_model_size="base",
        vit_dec_model_size="large",
        vit_enc_patch_size=8,
        vit_dec_patch_size=8,
        num_latent_tokens=256,      # patches/frame; drives the sampler seq-reshape assert (STAYS 256)
        is_legacy=False,
        use_rmsnorm=False,
        use_swiglu=True,
        scale_factor=1.0,
    )

    config.sat_in_channels = 11
    config.sat_out_channels = 11
    config.radar_in_channels = 1
    config.radar_out_channels = 1
    config.ae_image_size = 128      # PixelPatchifier crop_size

    global model
    config.nnet = d(
        name="flowtok-b",
        model_args=model,
    )

    config.losses = d(
        contrastive_loss_weight=0.0,
        kld_loss_weight=0.0,
    )
    config.loss_coeffs = []

    config.generation_algorithm = "flow_matching"
    config.flow_prediction_target = "radar_tokens"
    config.flow_cond_mode = "cross_attention"

    config.use_text_vae_encoder = False
    config.cond_use_sat_lightning_tokens = False
    config.cond_token_fusion = "mean"

    config.dataset = d(
        filelist_path=(
            "/g/data/kl02/yh0308/Data/71/filelists/"
            "dataset_filelist_i2i_train_201906_202406_cond3nan1_clip16_p005_seed42.pkl"
        ),
        filelist_split="train",
        v2v=True,                   # STAYS True: i2i == T=1 v2v (same as xattn-i2i)
        num_frames=1,               # <-- the ONLY structural v2v->i2i switch
        frame_stride=1,
        num_workers_per_gpu=4,
        crop_size=128,
        ir_band_indices=None,
        use_lightning=True,
        augment=d(
            enabled=True,
            hflip=True,
            vflip=True,
        ),
    )

    config.workdir = (
        "/scratch/kl02/yh0308/Projv2v/Experiments/"
        "sat2radar_flowtok_i2i_cmp_pixfact_B_p8_cond3nan1"
    )
    config.ckpt_root = config.workdir + "/ckpts"
    config.sample_dir = config.workdir + "/samples"

    config.sample = d(
        sample_steps=20,
        n_samples=16,
        mini_batch_size=4,
        scale=1.0,
        noise_scale=0.1,
        path=config.sample_dir + "/samples_eval",
    )

    config.adapter_in_satellite = d(enabled=False)
    config.adapter_in_radar = d(enabled=False)
    config.adapter_out = d(enabled=False)

    return config
