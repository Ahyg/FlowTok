import ml_collections
from dataclasses import dataclass


@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# FlowTok backbone (sat condition; supports variable length T*77 for V2V)
model = Args(
    learn_sigma=False,
    channels=16,          # token dim (match FlowTiTok AE token_size)
    use_t2i=False,        # sat-to-radar, not text-to-image
    clip_dim=16,          # condition token dim (sat tokens as cond)
    num_clip_token=77,    # latent tokens per frame; sequence length T*77, dynamic pos_embed
    gradient_checking=False,
    cfg_indicator=0.10,
    noising_type="none",
    noising_scale=0.1,
    textVAE=Args(
        num_blocks=6,
        hidden_dim=256,
        num_attention_heads=4,
        dropout_prob=0.1,
        clip_loss_weight=0.0,   # 不用 text CLIP 对比损失
        align_quantized=False,
        use_pretrained=False,
        tokenizer_checkpoint="",
        freeze_encoder=False,
    ),
)


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    # ========= 基本设置 =========
    config.seed = 1234

    # Set paths to your pretrained AE checkpoints (sat 11ch -> 77 tokens; radar 1ch -> 77 tokens)
    config.sat_tokenizer_checkpoint = "/mnt/ssd_1/yghu/Experiments/sat_flowtitok_ae_bl77_vae_run1/checkpoint-650000/ema_model/pytorch_model.bin"
    config.radar_tokenizer_checkpoint = "/mnt/ssd_1/yghu/Experiments/radar_flowtitok_ae_bl77_vae_run1/checkpoint-450000/ema_model/pytorch_model.bin"

    # ========= 训练设置 =========
    config.train = d(
        n_steps=1_000_000,
        batch_size=64,
        log_interval=100,
        eval_interval=10_000,
        save_interval=200_000,
        n_samples_eval=4,
    )

    # ========= 优化器 / LR =========
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

    # ========= FlowTiTok AE 相关 =========
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
        scale_factor=1.0143,
    )

    # ========= FlowTok nnet 配置 =========
    global model
    config.nnet = d(
        name="flowtok-xl",  # DiT backbone (FlowTok), not FlowTiTok AE
        model_args=model,
    )

    # ========= loss 权重 =========
    config.losses = d(
        contrastive_loss_weight=0.0,
        kld_loss_weight=0.01,
    )
    config.loss_coeffs = []

    # ========= Dataset: I2I vs V2V =========
    # Pipeline:
    #   I2I: 11ch sat image -> sat tokenizer -> 77 tokens -> DiT -> radar tokens -> radar decoder -> 1ch radar image.
    #   V2V: n frames -> sat tokenizer -> n*77 tokens -> DiT -> n*77 radar tokens -> radar decoder -> radar video.
    # Use num_frames=1 for I2I (single frame); num_frames=16 or (min,max) for V2V. Filelist: build with --v2v --clip-length 1 for I2I or --clip-length 16 for fixed-length V2V.
    config.dataset = d(
        filelist_path="/mnt/ssd_1/yghu/Data/71_tiny/dataset_filelist_i2i.pkl",
        filelist_split="train",
        v2v=True,
        num_frames=1,   # I2I: set to 1. V2V: fixed 16 or tuple e.g. (1,16) for variable T
        frame_stride=1,
        num_workers_per_gpu=4,
        crop_size=128,
    )

    # ========= checkpoint / 日志 / 采样 =========
    config.workdir = "/mnt/ssd_1/yghu/Experiments/debug_sat2radar_video_run"
    config.ckpt_root = "/mnt/ssd_1/yghu/Experiments/debug_sat2radar_video_run/ckpts"
    config.sample_dir = "/mnt/ssd_1/yghu/Experiments/debug_sat2radar_video_run/samples"

    # 采样相关（flow matching ODE solver）
    config.sample = d(
        sample_steps=20,
        n_samples=16,
        mini_batch_size=4,
        scale=2.0,
        noise_scale=0.1,
        path="/mnt/ssd_1/yghu/Experiments/debug_sat2radar_video_run/samples_eval",
    )

    return config

