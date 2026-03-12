import ml_collections
from dataclasses import dataclass


@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# 基于 Sat2Radar-v2v-pretrained-FlowTiTok-XL 的配置，
# 打开 textVAE 的对比损失，用于增强 sat tokens 与 textVAE latent 之间的一致性。
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
    textVAE=Args(
        num_blocks=6,
        hidden_dim=256,
        num_attention_heads=4,
        dropout_prob=0.1,
        clip_loss_weight=1.0,   # textVAE 内部对比损失权重（保持与 FlowTok 设计兼容）
        align_quantized=False,
        use_pretrained=False,
        tokenizer_checkpoint="",
        freeze_encoder=False,
    ),
)

# 为了在对比损失中使用 t2t_temperature，需要显式打开该开关，
# 否则 FlowTok 会将其视为不存在，导致后续 t2t_temperature.exp() 报错。
model.use_t2t_temperature = True


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234

    ftok_path = "/mnt/ssd_1/yghu/Data/flowtok_ckpts/FlowTiTok_512.bin"
    config.sat_tokenizer_checkpoint = ftok_path
    config.radar_tokenizer_checkpoint = ftok_path

    # ========= Train =========
    config.train = d(
        n_steps=100_000,
        batch_size=4,
        log_interval=100,
        eval_interval=500,
        save_interval=20_000,
        n_samples_eval=4,
    )

    # ========= Optimizer / LR =========
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

    # ========= loss 权重 =========
    # 对比损失打开，用于 textVAE latent 与 cond= sat_tokens 之间的对齐；
    # KL 正则维持一个较小权重。
    config.losses = d(
        contrastive_loss_weight=0.1,
        kld_loss_weight=0.01,
    )
    config.loss_coeffs = []

    # 保持使用 textVAE encoder（与默认行为一致）。
    config.use_text_vae_encoder = True

    # ========= Dataset: Sat(0/2/6) -> Radar(last ch) =========
    config.dataset = d(
        filelist_path="/mnt/ssd_1/yghu/Data/71_3m/dataset_filelist_v2v.pkl",
        filelist_split="train",
        v2v=True,
        num_frames=16,
        frame_stride=1,
        num_workers_per_gpu=4,
        crop_size=128,
        ir_band_indices=[0, 2, 6],
        use_lightning=False,
    )

    config.workdir = "/mnt/ssd_1/yghu/Experiments/sat2radar_flowtok_run_v2v_contrastive"
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

    # ========= Adapters =========
    config.adapter_in_satellite = d(
        enabled=False,
        in_channels=3,
        mid_channels=32,
        num_blocks=3,
    )
    config.adapter_out = d(
        enabled=False,
    )

    return config

