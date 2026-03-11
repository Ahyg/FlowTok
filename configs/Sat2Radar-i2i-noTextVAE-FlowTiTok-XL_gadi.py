import ml_collections
from dataclasses import dataclass


@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# Gadi 版本的 noTextVAE 配置：
# 语义上与 Sat2Radar-v2v-noTextVAE-FlowTiTok-XL 一致，
# 但将权重路径、数据集 filelist 与 workdir 改为 Gadi 上的实际/推荐位置。
model = Args(
    learn_sigma=False,
    channels=16,          # token dim（匹配 FlowTiTok_512 的 token_size）
    use_t2i=False,        # sat-to-radar，不是 text-to-image
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
        clip_loss_weight=0.0,  # 不使用 CLIP 对比损失
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

    config.seed = 1234

    # FlowTiTok 预训练权重在 Gadi 上的推荐路径
    ftok_path = "/g/data/kl02/yh0308/Data/flowtok_ckpts/FlowTiTok_512.bin"
    config.sat_tokenizer_checkpoint = ftok_path
    config.radar_tokenizer_checkpoint = ftok_path

    # ========= Train =========
    # 训练超参可以根据需要调整；这里沿用 contrastive_gadi 配置的规模。
    config.train = d(
        n_steps=400_000,
        batch_size=64,
        log_interval=100,
        eval_interval=1000,
        save_interval=50_000,
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

    # AE 通道与 v2v/noTextVAE 本地配置保持一致：
    #   - 卫星：0/2/6 三个 IR 通道伪 RGB
    #   - 雷达：通过 3→1 的复制/截取适配
    config.sat_in_channels = 3
    config.sat_out_channels = 3
    config.radar_in_channels = 3
    config.radar_out_channels = 3

    # AE 预训练时的分辨率（FlowTiTok_512 在 512x512 上训练）
    config.ae_image_size = 512

    # ========= FlowTok nnet 配置 =========
    global model
    config.nnet = d(
        name="flowtok-xl",
        model_args=model,
    )

    # ========= loss 权重 =========
    # noTextVAE 方案：不使用 textVAE KL / 对比学习，仅优化 flow matching 本身。
    config.losses = d(
        contrastive_loss_weight=0.0,
        kld_loss_weight=0.0,
    )
    config.loss_coeffs = []

    # 显式关闭 textVAE encoder，用于 FlowMatching 中的条件判断。
    config.use_text_vae_encoder = False

    # ========= Dataset: Sat(0/2/6) -> Radar(last ch) =========
    # 使用在 Gadi 上构建的 v2v filelist；在评估脚本中通过 mode="i2i"/"v2v" 控制 T=1 或 T>1。
    config.dataset = d(
        filelist_path="/g/data/kl02/yh0308/Data/71/dataset_filelist_i2i.pkl",
        filelist_split="train",
        v2v=True,
        num_frames=16,
        frame_stride=1,
        num_workers_per_gpu=4,
        crop_size=128,
        ir_band_indices=[0, 2, 6],
        use_lightning=False,
    )

    # ========= workdir / ckpt / samples =========
    config.workdir = "/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_run_i2i_noTextVAE"
    config.ckpt_root = config.workdir + "/ckpts"
    config.sample_dir = config.workdir + "/samples"

    # 采样相关（flow matching ODE solver）
    config.sample = d(
        sample_steps=20,
        n_samples=16,
        mini_batch_size=4,
        scale=2.0,
        noise_scale=0.1,
        path=config.sample_dir + "/samples_eval",
    )

    return config

