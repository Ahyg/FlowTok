"""v2v ABLATION 1 "pixfact" TINY — 32-clip overfit gate (12k steps) before the full run.

Same architecture/settings as Sat2Radar-v2v-cmp-pixfact-B-p8-cond3nan1_gadi.py
(PIXEL space, P=8, factorized DiT + flow-matching) but on the 32-clip overfit pkl
with fast eval/save. Patchify is lossless, so a healthy run should drive diff_loss
to near-zero; failure to overfit isolates a wiring/normalization bug, not capacity.
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
    channels=64,            # radar pixel-patch dim = 1 * 8^2
    cond_channels=704,      # sat pixel-patch dim   = 11 * 8^2
    use_t2i=False,
    clip_dim=16,
    num_clip_token=256,
    num_latent_tokens=256,  # patches/frame L = (128/8)^2
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

    config.pixel_space = True
    config.patch_size = 8
    config.sat_tokenizer_checkpoint = ""
    config.radar_tokenizer_checkpoint = ""

    config.train = d(
        n_steps=12_000,
        batch_size=8,
        log_interval=100,
        eval_interval=500,
        save_interval=4_000,
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
        num_latent_tokens=256,
        is_legacy=False,
        use_rmsnorm=False,
        use_swiglu=True,
        scale_factor=1.0,
    )

    config.sat_in_channels = 11
    config.sat_out_channels = 11
    config.radar_in_channels = 1
    config.radar_out_channels = 1
    config.ae_image_size = 128

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

    # Reuses the architecture-agnostic 32-clip overfit pkl (same as fact-tiny).
    config.dataset = d(
        filelist_path=(
            "/scratch/kl02/yh0308/Projv2v/Experiments/"
            "sat2radar_flowtok_v2v_cmp_xattn_tiny/dataset_filelist.pkl"
        ),
        filelist_split="train",
        v2v=True,
        num_frames=16,
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
        "sat2radar_flowtok_v2v_cmp_pixfact_tiny"
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
