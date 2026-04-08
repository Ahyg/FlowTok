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
    cfg_indicator=0.10,
    noising_type="none",
    noising_scale=0.1,
    # Enable learnable temperature for contrastive alignment loss (L_align).
    # Required when contrastive_loss_weight > 0 to avoid NoneType error.
    use_t2t_temperature=True,
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

    ftok_path = "/g/data/kl02/yh0308/Data/flowtok_ckpts/FlowTiTok_512.bin"
    config.sat_tokenizer_checkpoint = ftok_path
    config.radar_tokenizer_checkpoint = ftok_path

    config.train = d(
        n_steps=200_000,
        batch_size=64,
        log_interval=100,
        eval_interval=1_000,
        save_interval=20_000,
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

    config.sat_in_channels = 3
    config.sat_out_channels = 3
    config.radar_in_channels = 3
    config.radar_out_channels = 3
    config.ae_image_size = 512

    global model
    config.nnet = d(
        name="flowtok-xl",
        model_args=model,
    )

    # ── Key change: enable textVAE as "Sat Token Projector" ──
    # Following the FlowTok paper (Sec 4.1), the source tokens (sat_tokens)
    # are projected through context_encoder (6 Transformer blocks) with
    # KLD regularization + contrastive alignment loss before flow matching.
    config.losses = d(
        contrastive_loss_weight=1.0,   # L_align: preserve semantic info
        kld_loss_weight=1e-4,          # L_kld: regularize toward N(0,1)
    )
    config.loss_coeffs = []
    config.use_text_vae_encoder = True   # enable context_encoder projector

    # sat IR + lightning token fusion for i2i
    config.cond_use_sat_lightning_tokens = True
    config.cond_token_fusion = "mean"

    config.dataset = d(
        filelist_path="/g/data/kl02/yh0308/Data/71/filelists/dataset_filelist_i2i_train_201906_202312_ct005.pkl",
        filelist_split="train",
        v2v=True,
        num_frames=1,
        frame_stride=1,
        num_workers_per_gpu=4,
        crop_size=128,
        ir_band_indices=[0, 2, 6],
        use_lightning=True,
        augment=d(
            enabled=True,
            hflip=True,
            vflip=True,
        ),
    )

    config.workdir = "/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_run_i2i_satlight_tokenfusion_textVAE"
    config.ckpt_root = config.workdir + "/ckpts"
    config.sample_dir = config.workdir + "/samples"

    config.sample = d(
        sample_steps=20,
        n_samples=16,
        mini_batch_size=4,
        scale=2.0,
        noise_scale=0.1,
        path=config.sample_dir + "/samples_eval",
    )

    config.adapter_in_satellite = d(enabled=False)
    config.adapter_in_radar = d(enabled=False)
    config.adapter_out = d(enabled=False)

    return config
