#!/usr/bin/env python3
"""Generate the radar AE late-GAN sweep configs (the deferred Stage 3 of the
2026-05-20 loss-recipe spec, run on user request).

Base = the radar GAN-free WINNER from the 2026-05-22 recipe-sweep results:
  patch_size 8 (enc+dec), num_latent_tokens 128, kl_weight 6.02e-8 (÷10),
  perceptual_weight 0.6, l2 recon, tiny-enc/small-dec, token_size 16, vae.
(The recommended-radar recipe; its three wins stacked. "Combining wins has not
been jointly tested" per the results doc — these runs are also that joint test,
GAN-off anchor included.)

Template: configs/rs_radar_b0_p8.yaml (carries patch=8 + the screening regime).

GAN axis (user-chosen, brackets 0.1 by a decade each side):
  discriminator_weight ∈ {0.001, 0.01, 0.1, 1.0}, late gentle engage at
  disc_start = 0.55 × 100k = 55k (after recon converges ~60k in this regime),
  discriminator_learning_rate 5e-5, lecam 0.01 (run3's only correct ingredient).
Plus a paired GAN-OFF anchor at the same 100k horizon → isolates the GAN effect.

Emits 5 training configs + 1 GAN-firing smoke config into configs/.
"""
import copy
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "rs_radar_b0_p8.yaml"
EXP_ROOT = "/mnt/ssd_2/yghu/Experiments/ae_recipe_sweep/gan"
PROJECT = "ae_gan_sweep_20260603"

CONFIRM_STEPS = 100000
WARMUP = 10000            # 10% of horizon
# Non-swept GAN params follow TA-TiTok's tatitok_bl64_vae.yaml (user request):
# disc_lr=1e-4, lecam=1e-3, disc_factor=1.0, disc_weight baseline 0.1 (swept axis).
# EXCEPTION: disc_start is a user override (not TA-TiTok's 0.31× fraction) —
# engage the GAN at 60k, AFTER recon plateaus (~60k in this tiny-model regime
# per the 2026-05-20 spec), so the GAN refines a converged recon instead of
# fighting it (avoids run2's early-GAN divergence). 60k of 100k = 0.6× horizon.
DISC_START = 60000        # user override: after recon convergence (~60k)
DISC_LR = 1.0e-04         # TA-TiTok
LECAM = 0.001             # TA-TiTok

# The radar winner recipe (stacked on the patch=8 template's kl/perc).
KL_WINNER = 6.02e-08      # ÷10
PERC_WINNER = 0.6

# (cell_suffix, discriminator_weight | None=GAN off anchor)
CELLS = {
    "nogan": None,
    "w0001": 0.001,
    "w001":  0.01,
    "w01":   0.1,
    "w1":    1.0,
}


def base_cfg():
    """Template + the radar winner recipe + 100k GAN-ready regime (GAN still off
    here; per-cell mutator turns it on)."""
    cfg = OmegaConf.load(TEMPLATE)
    cfg.experiment.project = PROJECT
    cfg.experiment.save_every = 10000      # rolling latest (resume only)
    cfg.experiment.eval_every = 5000       # 20 val checks over 100k -> best_val
    cfg.experiment.generate_every = 10000
    cfg.experiment.log_every = 200
    # radar winner recipe (stack kl÷10 + perc0.6 on top of patch=8)
    cfg.losses.kl_weight = KL_WINNER
    cfg.losses.perceptual_weight = PERC_WINNER
    # gentle-GAN stabilizers (shared; harmless when GAN is off)
    cfg.losses.lecam_regularization_weight = LECAM
    cfg.optimizer.params.discriminator_learning_rate = DISC_LR
    # confirm horizon
    cfg.training.max_train_steps = CONFIRM_STEPS
    cfg.lr_scheduler.params.warmup_steps = WARMUP
    return cfg


def build(suffix, disc_weight):
    cfg = base_cfg()
    cell = f"radar_gan_{suffix}"
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


def build_smoke():
    """60-step config that ACTUALLY fires the GAN (disc_start=20) so the
    discriminator forward/backward + its optimizer are validated before the
    real 5×100k launch. Uses the strongest weight (1.0) to surface instability."""
    cfg = base_cfg()
    cfg.experiment.name = "smoke_radar_gan"
    # NOTE: write to ssd_2 (320G free), NOT /tmp — root LV is 100% full on lab2.
    cfg.experiment.output_dir = f"{EXP_ROOT}/smoke_radar_gan"
    cfg.experiment.logging_dir = f"{EXP_ROOT}/smoke_radar_gan/logs"
    cfg.experiment.save_every = 30
    cfg.experiment.eval_every = 30
    cfg.experiment.generate_every = 30
    cfg.experiment.log_every = 10
    cfg.losses.discriminator_start = 20
    cfg.losses.discriminator_weight = 1.0
    cfg.training.max_train_steps = 60
    cfg.lr_scheduler.params.warmup_steps = 10
    dest = REPO / "configs" / "smoke_radar_gan.yaml"
    OmegaConf.save(cfg, dest)
    return dest.name


def main():
    made = [build(s, w) for s, w in CELLS.items()]
    smoke = build_smoke()
    print(f"Generated {len(made)} training configs:")
    for fn, cell in made:
        print(f"  configs/{fn}  ->  {EXP_ROOT}/{cell}")
    print(f"Smoke config: configs/{smoke}  (disc_start=20, 60 steps, weight 1.0)")


if __name__ == "__main__":
    main()
