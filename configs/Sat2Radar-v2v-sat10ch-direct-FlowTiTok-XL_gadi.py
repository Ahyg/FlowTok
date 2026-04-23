import ml_collections
from dataclasses import dataclass


@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# Direct sat10ch-AE tokens -> radar-AE tokens via flow matching.
# - sat AE: sat10ch run1 (11 in/out channels, 77 tokens, 16-dim)
# - radar AE: radar run1 (1 in/out channels, 77 tokens, 16-dim)
# - Flow starts from sat tokens directly (use_text_vae_encoder=False), no projector,
#   no fusion, no text VAE. Lightweight Gaussian noise is added to x0 at each step.
# - CFG disabled (cfg_indicator=0.0).

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

    # Run1 final AE checkpoints (trained at 128x128).
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
        n_steps=200_000,
        batch_size=8,
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

    # Same tokenizer architecture as sat10ch/radar AE training (base enc / large dec,
    # patch 16, 77 latent tokens, 16-dim). scale_factor=1.0 to match AE training.
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

    # Channel counts match the AE checkpoints exactly (no adaptation needed).
    config.sat_in_channels = 11
    config.sat_out_channels = 11
    config.radar_in_channels = 1
    config.radar_out_channels = 1
    # sat10ch / radar run1 AE was trained at 128 crop.
    config.ae_image_size = 128

    global model
    config.nnet = d(
        name="flowtok-xl",
        model_args=model,
    )

    # No auxiliary losses: no projector, no textVAE, no contrastive / KLD / recon.
    config.losses = d(
        contrastive_loss_weight=0.0,
        kld_loss_weight=0.0,
    )
    config.loss_coeffs = []

    # Core switch: skip textVAE encoder, use sat_tokens as flow x0 directly.
    config.use_text_vae_encoder = False

    # No sat/lgt splitting: feed the whole 11-channel sat video through sat AE.
    config.cond_use_sat_lightning_tokens = False
    config.cond_token_fusion = "mean"   # unused; only "concat" triggers projector.

    # V2V dataset: 16 frames at 128x128, all 10 IR bands + lightning.
    config.dataset = d(
        filelist_path=(
            "/g/data/kl02/yh0308/Data/71/filelists/"
            "dataset_filelist_v2v_train_201906_202312_halfvalid50_ct005.pkl"
        ),
        filelist_split="train",
        v2v=True,
        num_frames=16,
        frame_stride=1,
        num_workers_per_gpu=4,
        crop_size=128,
        ir_band_indices=None,   # default -> all 10 IR bands (channels 0..9)
        use_lightning=True,
        augment=d(
            enabled=True,
            hflip=True,
            vflip=True,
        ),
    )

    config.workdir = (
        "/scratch/kl02/yh0308/Projv2v/Experiments/"
        "sat2radar_flowtok_run_v2v_sat10ch_direct_full"
    )
    config.ckpt_root = config.workdir + "/ckpts"
    config.sample_dir = config.workdir + "/samples"

    # Inference: scale=1.0 (no CFG), noise_scale matches training noising_scale.
    config.sample = d(
        sample_steps=20,
        n_samples=16,
        mini_batch_size=4,
        scale=1.0,
        noise_scale=0.1,
        path=config.sample_dir + "/samples_eval",
    )

    # All adapters disabled (AE channel counts match natively).
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
