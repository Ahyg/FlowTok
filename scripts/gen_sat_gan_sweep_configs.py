#!/usr/bin/env python3
"""Generate the sat10ch AE late-GAN sweep configs — the sat analog of the radar
late-GAN sweep (scripts/gen_gan_sweep_configs.py), run on user request.

Base = the sat10ch perceptual WINNER (Variant B from the 2026-05-22 recipe-sweep
results): patch_size 8 (enc+dec), num_latent_tokens 128, kl_weight 6.02e-8 (÷10),
perceptual_weight 1.1, perceptual_per_channel=true (weights all 1), l2 recon,
tiny-enc/small-dec, token_size 16, vae, 11ch in/out, per_gpu_batch_size 8
(per_channel × 11ch OOMs at batch 16/32 on 24 GB). Variant B is the right base
when the AE feeds downstream flow-matching (rFID/LPIPS/FSS matter).
("Combining wins has not been jointly tested" per the results doc — these runs
are also that joint test for sat, with a GAN-off anchor included.)

Template: configs/rs_s10_b0_p8.yaml (carries patch=8 + the screening regime).

GAN axis (mirror radar, user-confirmed): discriminator_weight ∈ {0.001, 0.01,
0.1, 1.0}, late gentle engage at disc_start = 60000 (0.6× of 100k, after recon
plateaus — mirrored from radar). discriminator_learning_rate 1e-4, lecam 1e-3,
discriminator_factor 1.0 (TA-TiTok). Plus a paired GAN-OFF anchor at the same
100k horizon → isolates the GAN effect.

Emits 5 training configs + 1 GAN-firing smoke config into configs/.
"""
import copy
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "rs_s10_b0_p8.yaml"
EXP_ROOT = "/mnt/ssd_2/yghu/Experiments/ae_recipe_sweep/gan_sat"
PROJECT = "ae_gan_sat_sweep_20260604"

# Gadi (NCI) overrides — same recipe + GAN axis, only the data/output paths
# change. Data + val handling are copied from the user's working Gadi sat AE
# config (sat10ch_flowtitok_ae_bl77_vae_scratch_run3_gadi.yaml). conda env on
# Gadi for scripts/train_flowtitok_ae.py is `1d-tokenizer` (see PBS scripts).
GADI_EXP_ROOT = "/scratch/kl02/yh0308/Projv2v/Experiments/ae_recipe_sweep/gan_sat"
GADI_TRAIN_PKL = ("/g/data/kl02/yh0308/Data/71/filelists/"
                  "dataset_filelist_i2i_train_201906_202312_ct005.pkl")

CONFIRM_STEPS = 100000
WARMUP = 10000            # 10% of horizon
# GAN params mirror the radar sweep (user-confirmed): disc_start 60k (engage
# after recon plateau, 0.6× horizon), disc_lr 1e-4, lecam 1e-3, disc_factor 1.0.
DISC_START = 60000        # mirror radar: after recon convergence
DISC_LR = 1.0e-04         # TA-TiTok
LECAM = 0.001             # TA-TiTok

# The sat Variant-B (perceptual) winner recipe, stacked on the patch=8 template.
KL_WINNER = 6.02e-08              # ÷10
PERC_WINNER = 1.1                # variant B perceptual weight
PER_CHANNEL_WEIGHTS = [1] * 11   # keep lightning ch10 wt=1 (zeroing it hurt mse)
BATCH = 8                        # per_channel × 11ch ceiling at 24 GB

# (cell_suffix, discriminator_weight | None=GAN off anchor)
CELLS = {
    "nogan": None,
    "w0001": 0.001,
    "w001":  0.01,
    "w01":   0.1,
    "w1":    1.0,
}


def base_cfg():
    """Template + the sat Variant-B winner recipe + 100k GAN-ready regime (GAN
    still off here; per-cell mutator turns it on)."""
    cfg = OmegaConf.load(TEMPLATE)
    cfg.experiment.project = PROJECT
    cfg.experiment.save_every = 10000      # rolling latest (resume only)
    cfg.experiment.eval_every = 5000       # 20 val checks over 100k -> best_val
    cfg.experiment.generate_every = 10000
    cfg.experiment.log_every = 200
    # sat Variant-B winner recipe (stack kl÷10 + perc1.1 + per_channel on patch=8)
    cfg.losses.kl_weight = KL_WINNER
    cfg.losses.perceptual_weight = PERC_WINNER
    cfg.losses.perceptual_per_channel = True
    cfg.losses.perceptual_per_channel_weights = list(PER_CHANNEL_WEIGHTS)
    # per_channel × 11ch memory ceiling
    cfg.training.per_gpu_batch_size = BATCH
    # gentle-GAN stabilizers (shared; harmless when GAN is off)
    cfg.losses.lecam_regularization_weight = LECAM
    cfg.optimizer.params.discriminator_learning_rate = DISC_LR
    # confirm horizon
    cfg.training.max_train_steps = CONFIRM_STEPS
    cfg.lr_scheduler.params.warmup_steps = WARMUP
    return cfg


def build(suffix, disc_weight):
    cfg = base_cfg()
    cell = f"s10_gan_{suffix}"
    out_dir = f"{EXP_ROOT}/{cell}"
    cfg.experiment.name = cell
    cfg.experiment.output_dir = out_dir
    cfg.experiment.logging_dir = f"{out_dir}/logs"
    if disc_weight is None:
        cfg.losses.discriminator_start = 9999999       # GAN never fires (anchor)
        # leave discriminator_weight at template default; it is unused when off
    else:
        cfg.losses.discriminator_start = DISC_START
        cfg.losses.discriminator_weight = disc_weight
    dest = REPO / "configs" / f"{cell}.yaml"
    OmegaConf.save(cfg, dest)
    return dest.name, cell


def build_gadi(suffix, disc_weight):
    """Same recipe + GAN axis as build(), but with Gadi data/output paths.
    Emits configs/s10_gan_<suffix>_gadi.yaml for `qsub` on NCI Gadi."""
    cfg = base_cfg()
    cell = f"s10_gan_{suffix}"
    out_dir = f"{GADI_EXP_ROOT}/{cell}"
    cfg.experiment.name = f"{cell}_gadi"
    cfg.experiment.output_dir = out_dir
    cfg.experiment.logging_dir = f"{out_dir}/logs"
    cfg.dataset.params.filelist_path = GADI_TRAIN_PKL
    if disc_weight is None:
        cfg.losses.discriminator_start = 9999999
    else:
        cfg.losses.discriminator_start = DISC_START
        cfg.losses.discriminator_weight = disc_weight
    dest = REPO / "configs" / f"{cell}_gadi.yaml"
    OmegaConf.save(cfg, dest)
    return dest.name, cell


def build_smoke():
    """60-step config that ACTUALLY fires the GAN (disc_start=20) so the
    discriminator forward/backward + its optimizer are validated — AND so the
    per_channel(×11) + GAN memory peak is checked at batch 8 — before the real
    5×100k launch. Uses the strongest weight (1.0) to surface instability."""
    cfg = base_cfg()
    cfg.experiment.name = "smoke_s10_gan"
    # NOTE: write to ssd_2 (root LV is 100% full on lab2).
    cfg.experiment.output_dir = f"{EXP_ROOT}/smoke_s10_gan"
    cfg.experiment.logging_dir = f"{EXP_ROOT}/smoke_s10_gan/logs"
    cfg.experiment.save_every = 30
    cfg.experiment.eval_every = 30
    cfg.experiment.generate_every = 30
    cfg.experiment.log_every = 10
    cfg.losses.discriminator_start = 20
    cfg.losses.discriminator_weight = 1.0
    cfg.training.max_train_steps = 60
    cfg.lr_scheduler.params.warmup_steps = 10
    dest = REPO / "configs" / "smoke_s10_gan.yaml"
    OmegaConf.save(cfg, dest)
    return dest.name


def main():
    made = [build(s, w) for s, w in CELLS.items()]
    made_gadi = [build_gadi(s, w) for s, w in CELLS.items()]
    smoke = build_smoke()
    print(f"Generated {len(made)} lab training configs:")
    for fn, cell in made:
        print(f"  configs/{fn}  ->  {EXP_ROOT}/{cell}")
    print(f"Generated {len(made_gadi)} Gadi training configs:")
    for fn, cell in made_gadi:
        print(f"  configs/{fn}  ->  {GADI_EXP_ROOT}/{cell}")
    print(f"Smoke config: configs/{smoke}  (disc_start=20, 60 steps, weight 1.0, batch {BATCH})")


if __name__ == "__main__":
    main()
