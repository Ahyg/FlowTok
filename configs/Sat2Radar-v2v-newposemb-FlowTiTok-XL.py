import ml_collections
from dataclasses import dataclass


@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# 基础模型配置与 Sat2Radar-v2v-pretrained-FlowTiTok-XL 保持一致，
# 仅通过 config.use_text_vae_encoder 关闭 textVAE encoder 分支，
# 实现最基础的「sat tokens -> radar tokens」flow matching（方案 A）。
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
        clip_loss_weight=0.0,
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

    # AE 通道与基础 v2v 配置一致：sat 用 0/2/6 三个 IR 通道伪 RGB；radar 通过 3→1 的复制/截取适配。
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
    # 方案 A：不使用 textVAE KL / 对比学习，仅优化 flow matching 本身。
    config.losses = d(
        contrastive_loss_weight=0.0,
        kld_loss_weight=0.0,
    )
    config.loss_coeffs = []

    # 显式关闭 textVAE encoder，用于 FlowMatching 中的条件判断。
    # 当该标志为 False 时：
    #   - 训练：p_losses_textVAE_flowtok 中直接使用 cond (= sat_tokens) 作为噪声起点；
    #   - 采样：train/test/validate 中从 sat_tokens 出发做 ODE 采样。
    config.use_text_vae_encoder = False

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

    # ========= workdir / ckpt / samples =========
    config.workdir = "/mnt/ssd_1/yghu/Experiments/sat2radar_flowtok_run_v2v_newposemb"
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

