"""V2V M8-aligned — M8's token_concat + x1(radar_tokens), but with the sat/radar
tokens INTERLEAVED per frame (aligned layout) instead of block-concatenated.

A/B partner of the block M8 (configs/Sat2Radar-v2v-m8-tokconcat-xpred-FlowTiTok-XL_gadi.py).
EVERYTHING is identical to that block M8 config except:
  flow_cond_mode = "token_concat_interleaved"   (block M8 uses "token_concat")
  + a different workdir.

Layout difference (the whole point of this arm):
  block M8           : inp = [ sat_0..15 | radar_0..15 ]  (2*16*77=2464 tokens)
                       pos_n_per_frame=77 -> sat frames temporal 0..15, radar 16..31
                       => sat frame i and radar frame i are temporally MISaligned (offset 16)
  this (interleaved) : inp = [ sat_0|radar_0 | sat_1|radar_1 | ... ] per fat frame = 2L
                       pos_n_per_frame=2L=154 -> sat_i and radar_i SHARE temporal pos i
                       => modality encoded by spatial half (0..76 sat, 77..153 radar),
                          time aligned. Mirrors the old sat+lgt seqconcat pos-embed.

Same as block M8: predict x1 (radar_tokens), sat cond clean & prepended each step,
loss only on the radar half. Implemented via flow_cond_mode="token_concat_interleaved"
(diffusion/flow_matching.py _interleave_cond_target/_deinterleave_target); the block
"token_concat" path is untouched so the in-flight block-M8 job is unaffected.
"""
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
    cfg_indicator=0.0,
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
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 1234

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

    config.sat_in_channels = 11
    config.sat_out_channels = 11
    config.radar_in_channels = 1
    config.radar_out_channels = 1
    config.ae_image_size = 128

    global model
    config.nnet = d(
        name="flowtok-xl",
        model_args=model,
    )

    config.losses = d(
        contrastive_loss_weight=0.0,
        kld_loss_weight=0.0,
    )
    config.loss_coeffs = []

    # ===== M8 method, INTERLEAVED (aligned) layout =====
    config.generation_algorithm = "flow_matching"
    config.flow_prediction_target = "radar_tokens"          # predict x1
    config.flow_cond_mode = "token_concat_interleaved"      # <-- the only change vs block M8

    config.use_text_vae_encoder = False
    config.cond_use_sat_lightning_tokens = False
    config.cond_token_fusion = "mean"

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
        ir_band_indices=None,
        use_lightning=True,
        augment=d(
            enabled=True,
            hflip=True,
            vflip=True,
        ),
    )

    config.workdir = (
        "/scratch/kl02/yh0308/Projv2v/Experiments/"
        "sat2radar_flowtok_v2v_m8align_tokconcat_xpred_full"
    )
    config.ckpt_root = config.workdir + "/ckpts"
    config.sample_dir = config.workdir + "/samples"

    config.sample = d(
        sample_steps=20,
        n_samples=16,
        mini_batch_size=4,
        scale=1.0,
        noise_scale=0.1,
        path=config.sample_dir + "/samples_eval",
    )

    config.adapter_in_satellite = d(enabled=False)
    config.adapter_in_radar = d(enabled=False)
    config.adapter_out = d(enabled=False)

    return config
