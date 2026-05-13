import ml_collections
from dataclasses import dataclass


@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# Bidirectional sat10ch-direct: shared-weight FlowTok-XL DiT trained jointly on
# sat→radar (forward) + radar→sat (reverse) flows. Both AEs frozen.
# Train_step pattern: two forwards + two backwards + one optimizer.step.
# Local lab2 paths; periodic val on val_small (1st week of 2024/06).

model = Args(
    learn_sigma=False,
    channels=16,
    use_t2i=False,
    clip_dim=16,
    num_clip_token=77,
    gradient_checking=False,
    cfg_indicator=0.0,           # CFG OFF — sat/radar tokens are the only signal
    noising_type="constant",
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

    # Local lab2 AE checkpoints (run1 EMA).
    config.sat_tokenizer_checkpoint = (
        "/mnt/ssd_1/yghu/Data/flowtok_ckpts/"
        "sat10ch_flowtitok_ae_bl77_vae_scratch_run1_gadi/pytorch_model.bin"
    )
    config.radar_tokenizer_checkpoint = (
        "/mnt/ssd_1/yghu/Data/flowtok_ckpts/"
        "radar_flowtitok_ae_bl77_vae_scratch_run1_gadi/pytorch_model.bin"
    )

    config.train = d(
        n_steps=50_000,
        batch_size=2,             # single 4090, 16-frame v2v clips
        log_interval=100,
        eval_interval=2_000,
        save_interval=10_000,
        n_samples_eval=4,
        val_max_batches=28,       # ~all of val_small (56 clips / batch 2)
    )

    config.optimizer = d(
        name="adamw",
        lr=4e-4,
        weight_decay=0.03,
        betas=(0.9, 0.95),
    )

    config.lr_scheduler = d(
        name="customized",
        warmup_steps=2000,
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
        name="flowtok-xl",
        model_args=model,
    )

    # Auxiliary losses OFF (matches sat10ch-direct).
    config.losses = d(
        contrastive_loss_weight=0.0,
        kld_loss_weight=0.0,
    )
    config.loss_coeffs = []

    # textVAE OFF — sat tokens (fwd) / radar tokens (rev) used directly as flow x0.
    config.use_text_vae_encoder = False
    config.cond_use_sat_lightning_tokens = False
    config.cond_token_fusion = "mean"

    # ── Bidirectional config ────────────────────────────────────────────────
    # alpha = weight for forward direction (sat → radar)
    # beta  = weight for reverse direction (radar → sat)
    # beta > 0 enables bidir; beta == 0 falls back to uni-directional fwd.
    config.bidir = d(
        alpha=0.7,
        beta=0.3,
    )

    # ── Dataset ─────────────────────────────────────────────────────────────
    config.dataset = d(
        filelist_path=(
            "/mnt/ssd_1/yghu/Data/71_3m/filelists/"
            "dataset_filelist_v2v_train_202105_202110_halfvalid50_ct005.pkl"
        ),
        filelist_split="train",
        # Periodic val on val_small (1st week of 2024/06).
        val_filelist_path=(
            "/mnt/ssd_1/yghu/Data/71_3m/filelists/"
            "dataset_filelist_v2v_val_202406w1.pkl"
        ),
        v2v=True,
        num_frames=16,
        frame_stride=1,
        num_workers_per_gpu=4,
        crop_size=128,
        ir_band_indices=None,    # all 10 IR bands (0..9)
        use_lightning=True,
        augment=d(
            enabled=True,
            hflip=True,
            vflip=True,
        ),
    )

    config.workdir = (
        "/mnt/ssd_1/yghu/Experiments/"
        "sat2radar_flowtok_v2v_bidir_sat10ch_run1"
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

    # All adapters disabled.
    config.adapter_in_satellite = d(
        enabled=False,
        in_channels=11,
        mid_channels=32,
        num_blocks=3,
    )
    config.adapter_in_radar = d(
        enabled=False,
        in_channels=1,
        mid_channels=16,
        num_blocks=2,
    )
    config.adapter_out = d(
        enabled=False,
        mid_channels=16,
        num_blocks=2,
    )

    return config
