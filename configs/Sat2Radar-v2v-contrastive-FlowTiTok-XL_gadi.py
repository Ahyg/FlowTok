import ml_collections
from dataclasses import dataclass


@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# Gadi 版本配置（路径需根据实际环境修改）
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
        clip_loss_weight=1.0,
        align_quantized=False,
        use_pretrained=False,
        tokenizer_checkpoint="",
        freeze_encoder=False,
    ),
)

# 为了在对比损失中使用 t2t_temperature，需要显式打开该开关
model.use_t2t_temperature = True


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234

    # TODO: 按 Gadi 实际路径修改 FlowTiTok 权重路径
    ftok_path = "/g/data/kl02/yh0308/Data/flowtok_ckpts/FlowTiTok_512.bin"
    config.sat_tokenizer_checkpoint = ftok_path
    config.radar_tokenizer_checkpoint = ftok_path

    # ========= Train =========
    config.train = d(
        n_steps=400_000,
        batch_size=4,
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
    config.losses = d(
        contrastive_loss_weight=0.1,
        kld_loss_weight=0.01,
    )
    config.loss_coeffs = []

    config.use_text_vae_encoder = True

    # ========= Dataset: Sat(0/2/6) -> Radar(last ch) =========
    # TODO: 修改为在 Gadi 上的 dataset_filelist 路径
    config.dataset = d(
        filelist_path="/g/data/kl02/yh0308/Data/71/dataset_filelist_v2v.pkl",
        filelist_split="train",
        v2v=True,
        num_frames=16,
        frame_stride=1,
        num_workers_per_gpu=4,
        crop_size=128,
        ir_band_indices=[0, 2, 6],
        use_lightning=False,
    )

    # TODO: 修改为在 Gadi 上实际的实验输出目录
    config.workdir = "/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_run_v2v_contrastive"
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

    return config

