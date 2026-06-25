"""i2i cmp — Arm xattn-vpred (cross-attention conditioning, VELOCITY prediction).

Identical to Sat2Radar-i2i-cmp-xattn-B-bl128-cond3nan1_gadi.py EXCEPT
flow_prediction_target = "velocity" (the baseline xattn arm predicts "radar_tokens").
This isolates the prediction-target variable: cross-attn + Gaussian-noise x0, but the
nnet regresses the flow velocity dψ/dt directly instead of x1 (radar tokens).
Velocity prediction is NOT affected by the radar_tokens fixed_x0 time-averaging issue.

bl128 tokenizer (run4Bftgan / run4ftgan @ 300k, cond1 train) + FULL cond3nan1.
  - model size = flowtok-b
  - 200k steps, bs=64, num_frames=1
  - train pool = 71040 frames (5x of small20)
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
    channels=16,
    use_t2i=False,
    clip_dim=16,
    num_clip_token=128,
    num_latent_tokens=128,
    gradient_checking=False,
    cfg_indicator=0.0,
    noising_type="none",
    noising_scale=0.1,
    use_cross_attention=True,
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

    config.sat_tokenizer_checkpoint = (
        "/scratch/kl02/yh0308/Projv2v/Experiments/"
        "sat10ch_flowtitok_ae_bl128_vae_scratch_run4Bftgan_cond1_gadi/"
        "checkpoint-300000/ema_model/pytorch_model.bin"
    )
    config.radar_tokenizer_checkpoint = (
        "/scratch/kl02/yh0308/Projv2v/Experiments/"
        "radar_flowtitok_ae_bl128_vae_scratch_run4ftgan_cond1_gadi/"
        "checkpoint-300000/ema_model/pytorch_model.bin"
    )

    config.train = d(
        n_steps=600_000,
        batch_size=64,
        log_interval=100,
        eval_interval=2_000,
        save_interval=25_000,
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
        num_latent_tokens=128,
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
    config.flow_prediction_target = "velocity"
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
        v2v=True,
        num_frames=1,
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
        "sat2radar_flowtok_i2i_cmp_xattn_vpred_B_bl128_cond3nan1"
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
