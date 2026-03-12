import ml_collections
from dataclasses import dataclass


@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# FlowTok backbone: FlowTok-XL 风格，但用于 sat->radar（非 text->image）
model = Args(
    learn_sigma=False,
    channels=16,          # token dim（匹配 FlowTiTok_512 的 token_size）
    use_t2i=False,        # sat-to-radar，不是 text-to-image
    clip_dim=16,          # 这里仍然保留占位，真正的条件来自 sat tokens
    num_clip_token=77,    # 每帧 77 个 latent token
    gradient_checking=False,
    cfg_indicator=0.10,
    noising_type="none",
    noising_scale=0.1,
    textVAE=Args(
        num_blocks=6,
        hidden_dim=256,
        num_attention_heads=4,
        dropout_prob=0.1,
        clip_loss_weight=0.0,   # 不使用 CLIP 对比损失
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

    # 使用当前预训练的 image tokenizer（FlowTiTok_512.bin）
    ftok_path = "/mnt/ssd_1/yghu/Data/flowtok_ckpts/FlowTiTok_512.bin"
    config.sat_tokenizer_checkpoint = ftok_path
    config.radar_tokenizer_checkpoint = ftok_path

    # ========= Train =========
    config.train = d(
        n_steps=400_000,
        batch_size=4,
        log_interval=100,
        eval_interval=1000_000,
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
    # 注意：FlowTiTok_512 是在自然图像上 3 通道预训练的。
    # 这里我们约定：
    #   - 卫星：只用 0/2/6 三个 IR 通道作为“伪 RGB”，因此 AE 的 in/out=3
    #   - 雷达：单通道（最后一通道，0-60 dBZ），因此 AE 的 in/out=1
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

    # 这些字段会被 scripts/train_sat2radar_v2v.py 中的 _ae_config 读取，
    # 用来设置 FlowTiTok 的 in/out 通道数。
    # FlowTiTok_512 预训练是 3 通道 in / 3 通道 out。
    # 卫星：直接用 0/2/6 三个 IR 通道作为“伪 RGB”，因此 AE 的 in/out 都是 3。
    # 雷达：本身是单通道，但为了严格复用 3 通道 FlowTiTok_512 的权重，
    #       在送入 AE 前我们在脚本里把 1 通道复制到 3 通道，解码后再取第 1 个通道作为雷达图。
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
    config.losses = d(
        contrastive_loss_weight=0.0,
        kld_loss_weight=0.01,
    )
    config.loss_coeffs = []

    # ========= Dataset: Sat(0/2/6) -> Radar(last ch) =========
    # 使用 SatelliteRadarNpyDataset(sat2radar_v2v 模式)：
    #   - ir_band_indices=[0,2,6] => 只取 3 个 IR 通道
    #   - use_lightning=False     => 不拼接闪电通道
    config.dataset = d(
        filelist_path="/mnt/ssd_1/yghu/Data/71_3m/dataset_filelist_v2v.pkl",
        filelist_split="train",
        v2v=True,
        num_frames=16,   # 1 表示 i2i，>1 表示 v2v，可根据需要修改
        frame_stride=1,
        num_workers_per_gpu=4,
        crop_size=128,
        ir_band_indices=[0, 2, 6],
        use_lightning=False,
    )

    # ========= workdir / ckpt / samples =========
    config.workdir = "/mnt/ssd_1/yghu/Experiments/sat2radar_flowtitok_run_v2v"
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

    # ========= Adapters =========
    # AdapterIn: sat raw video -> tokenizer输入(3通道, 512x512)
    config.adapter_in_satellite = d(
        enabled=False,
        in_channels=3,  # 若后续加入 lightning，可改为 4
        mid_channels=32,
        num_blocks=3,
    )
    # AdapterOut: tokenizer解码输出 -> 目标雷达(单通道, 数据集分辨率)
    config.adapter_out = d(
        enabled=False,
    )

    return config

