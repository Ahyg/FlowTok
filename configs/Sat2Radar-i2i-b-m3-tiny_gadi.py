"""M1 baseline — i2i analog of run1 sat10ch-direct, FlowTok-B, 60k, 2021-summer.

Direct sat10ch-AE tokens -> radar-AE tokens via flow matching (velocity pred).
Single-variable controlled-ablation baseline; M2/M3/M4 each change one thing.
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
    num_clip_token=77,
    gradient_checking=False,
    cfg_indicator=0.0,          # CFG OFF — sat tokens are the only signal.
    noising_type="constant",
    noising_scale=0.1,
    # M3: channel-concat sat tokens with noisy radar tokens -> DiT in=2C, out=C.
    cond_concat_channels=True,
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

    # Run1 final AE checkpoints (trained at 128x128) — fixed across M1..M4.
    config.sat_tokenizer_checkpoint = (
        "/scratch/kl02/yh0308/Projv2v/Experiments/"
        "sat10ch_flowtitok_ae_bl77_vae_scratch_run1_gadi/"
        "checkpoint-200000/ema_model/pytorch_model.bin"
    )
    config.radar_tokenizer_checkpoint = (
        "/scratch/kl02/yh0308/Projv2v/Experiments/"
        "radar_flowtitok_ae_bl77_vae_scratch_run1_gadi/"
        "checkpoint-200000/ema_model/pytorch_model.bin"
    )

    config.train = d(
        n_steps=5_000,            # tiny overfit gate
        batch_size=16,            # 32-sample tiny set; drop_last -> need bs<=16

        log_interval=100,
        eval_interval=500,
        save_interval=2_500,
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
        vit_enc_patch_size=16,
        vit_dec_patch_size=16,
        num_latent_tokens=77,
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

    # M3 change: Diffi2i-style token diffusion (linear, pred_x0, DDIM-500).
    config.generation_algorithm = "diffusion"
    config.flow_prediction_target = "velocity"   # unused under diffusion
    config.diffusion = d(
        schedule="linear",
        target="pred_x0",
        train_timesteps=1000,
        sample_steps=100,         # tiny: faster vis sampling
        gamma="ddim",
    )

    # Direct flow: skip textVAE encoder, sat_tokens are flow x0 directly.
    config.use_text_vae_encoder = False
    config.cond_use_sat_lightning_tokens = False
    config.cond_token_fusion = "mean"

    config.dataset = d(
        filelist_path=(
            "/scratch/kl02/yh0308/Projv2v/Experiments/"
            "sat2radar_flowtok_i2i_b_diffusion_2021summer_tiny/dataset_filelist.pkl"
        ),
        filelist_split="train",
        v2v=True,
        num_frames=1,           # i2i: single frame
        frame_stride=1,
        num_workers_per_gpu=4,
        crop_size=128,
        ir_band_indices=None,   # all 10 IR bands (channels 0..9)
        use_lightning=True,
        augment=d(
            enabled=True,
            hflip=True,
            vflip=True,
        ),
    )

    config.workdir = (
        "/scratch/kl02/yh0308/Projv2v/Experiments/"
        "sat2radar_flowtok_i2i_b_diffusion_2021summer_tiny"
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
