import ml_collections
from dataclasses import dataclass

@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

model = Args(
    learn_sigma = False, # different from DiT, we direct predict noise here
    channels = 16,
    use_t2i = True,
    clip_dim=768,
    num_clip_token=77,
    gradient_checking=False, # for larger model
    cfg_indicator=0.10,
    noising_type='none',
    noising_scale=0.1,
    textVAE = Args(
        num_blocks = 6,
        hidden_dim = 1024,
        num_attention_heads = 8,
        dropout_prob = 0.1,
        clip_loss_weight=0.0,
        align_quantized=False,
        use_pretrained=False,
        tokenizer_checkpoint='/opt/tiger/ju/ckpt/flowtitok_swiglu_bl77_vae/pytorch_model_512.bin',
        freeze_encoder=False,
    ),
)

def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.tokenizer_checkpoint = '/opt/tiger/ju/ckpt/flowtitok_swiglu_bl77_vae/pytorch_model_512.bin'

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl.pth',
        scale_factor=0.23010
    )

    config.train = d(
        n_steps=1_000_000,
        batch_size=4096,
        mode='cond',
        log_interval=250,
        eval_interval=5_000,
        save_interval=100_000,
        n_samples_eval=5,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0004,
        weight_decay=0.03,
        betas=(0.9, 0.95),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000,
    )

    config.vq_model = d(
        deterministic=False,
        token_size=16,
        vit_enc_model_size='base',
        vit_dec_model_size='large',
        vit_enc_patch_size=16,
        vit_dec_patch_size=16,
        num_latent_tokens=77,
        is_legacy=False,
        use_rmsnorm=False,
        use_swiglu=True,
        scale_factor=1.0143,
    )

    global model
    config.nnet = d(
        name='flowtok-xl',
        model_args=model,
    )
    config.losses = d(
        contrastive_loss_weight=1.0,
        kld_loss_weight=0.01,
    )
    config.loss_coeffs = []
    
    config.dataset = d(
        train_shards_path_or_url='pipe:hdfs dfs -cat hdfs://harunava/home/byte_ailab_us_cvg/user/ju.he/data/flowtitok_pretoken/datacomp6/{000000..000019}.tar::pipe:hdfs dfs -cat hdfs://harunava/home/byte_ailab_us_cvg/user/ju.he/data/flowtitok_pretoken/laionart/{000000..000019}.tar::pipe:hdfs dfs -cat hdfs://harunava/home/byte_ailab_us_cvg/user/ju.he/data/flowtitok_pretoken/laionpop/{000000..000007}.tar::pipe:hdfs dfs -cat hdfs://harunava/home/byte_ailab_us_cvg/user/ju.he/data/flowtitok_pretoken/journeydb/{000000..000019}.tar::pipe:hdfs dfs -cat hdfs://harunava/home/byte_ailab_us_cvg/user/ju.he/data/flowtitok_pretoken/dalle3/{000000..000013}.tar',
        eval_shards_path_or_url='pipe:hdfs dfs -cat hdfs://harunava/home/byte_ailab_us_cvg/user/ju.he/data/coco/shard-{00..04}.tar',
        max_train_examples=2_000_000_000,
        num_workers_per_gpu=12,
        resize_shorter_edge=512,
        crop_size=512,
        random_crop=False,
        random_flip=False,
        dataset_with_class_label=False,
        dataset_with_text_label=True,
        res_ratio_filtering=True,
        pretokenized=True
    )

    config.sample = d(
        sample_steps=20,
        n_samples=30000,
        mini_batch_size=10,
        cfg=False,
        scale=2,
        noise_scale=0.1,
        path=''
    )

    return config