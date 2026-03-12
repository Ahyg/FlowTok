import ml_collections
from dataclasses import dataclass


@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


# 基础模型配置与 Sat2Radar-v2v-newposemb-FlowTiTok-XL 保持一致，
# 在该实验中显式开启 AdapterIn/AdapterOut，并启用 lightning 通道。
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
        clip_loss_weight=0.0,
        align_quantized=False,
        use_pretrained=False,
        tokenizer_checkpoint="",
        freeze_encoder=False,
    ),
)


def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 1234

    # TODO: 根据实际路径调整 FlowTiTok_512.bin 所在位置
    ftok_path = "/g/data/kl02/yh0308/Data/FlowTiTok_512.bin"
    config.sat_tokenizer_checkpoint = ftok_path
    config.radar_tokenizer_checkpoint = ftok_path

    # 保持与原 adapter 配置一致的训练设置（小 batch，主要用于 debug/adapter 实验）
    config.train = d(
        n_steps=200_000,
        batch_size=8,
        log_interval=100,
        eval_interval=1_000,
        save_interval=20_000,
        n_samples_eval=4,
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

    # tokenizer 仍使用 3 通道预训练权重；通过 AdapterIn 将原始 sat 通道映射到 3 通道
    config.sat_in_channels = 3
    config.sat_out_channels = 3
    config.radar_in_channels = 3
    config.radar_out_channels = 3
    config.ae_image_size = 512

    config.nnet = d(
        name="flowtok-xl",
        model_args=model,
    )

    config.losses = d(
        contrastive_loss_weight=0.0,
        kld_loss_weight=0.0,
        adapter_out_recon_weight=1.0,
    )
    config.loss_coeffs = []
    config.use_text_vae_encoder = False
    config.train_adapter_in_with_encoder_grad = True

    config.dataset = d(
        # TODO: 根据实际数据路径调整
        filelist_path="/g/data/kl02/yh0308/Data/71/dataset_filelist_v2v_3m.pkl",
        filelist_split="train",
        v2v=True,
        num_frames=16,
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

    # Gadi 上的实验输出目录
    config.workdir = "/scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_run_v2v_adapter"
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

    # adapter 设置：与原始 adapter 配置保持一致
    config.adapter_in_satellite = d(
        enabled=True,
        in_channels=4,
        mid_channels=32,
        num_blocks=3,
    )
    config.adapter_in_radar = d(
        enabled=True,
        in_channels=1,
        mid_channels=16,
        num_blocks=2,
    )
    config.adapter_out = d(
        enabled=True,
        mid_channels=16,
        num_blocks=2,
    )

    return config

