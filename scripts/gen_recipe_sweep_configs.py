#!/usr/bin/env python3
"""Generate the AE loss-recipe sweep configs (Stage 0 + Stage 1).

Spec:  docs/specs/2026-05-20-ae-loss-recipe-sweep-design.md
Template: configs/sat10ch_ae_tok128_lab2.yaml (the validated 2026-05-14 regime:
tiny enc + small dec ~37M, 128 tokens, run1 recipe, GAN off, seed 42).

Each cell = baseline (run1 recipe @128) with EXACTLY ONE loss knob changed
(the explicit antidote to run2's all-at-once failure). 40k-step screening,
GAN off (Stages 2-3 are sequentially dependent and run later).

Emits 12 training configs + 2 smoke configs into configs/.
"""
import copy
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "sat10ch_ae_tok128_lab2.yaml"
EXP_ROOT = "/mnt/ssd_2/yghu/Experiments/ae_recipe_sweep"
PROJECT = "ae_recipe_sweep_20260520"

SCREEN_STEPS = 40000
WARMUP = 4000          # 10% of horizon (matches spec / prior sweep convention)
KL_BASE = 6.02e-07     # = 1e-6 * 77/128 (per-token-normalized, verified in 2026-05-14)

# (cell_suffix, knob-mutator) — applied on top of the per-modality baseline.
LEVERS = {
    "b0":      lambda L: None,                                  # baseline, run1 recipe
    "s1_kl05": lambda L: L.__setitem__("kl_weight", 6.02e-08),  # kl x0.1
    "s1_kl10": lambda L: L.__setitem__("kl_weight", 6.02e-06),  # kl x10  (~run3's 1e-5 regime)
    "s1_pc":   lambda L: L.__setitem__("perceptual_per_channel", True),
    "s1_p06":  lambda L: L.__setitem__("perceptual_weight", 0.6),
    "s1_p16":  lambda L: L.__setitem__("perceptual_weight", 1.6),
}

# (modality_tag, overrides applied to model + dataset)
MODALITIES = {
    "radar": dict(in_channels=1, out_channels=1, mode="radar", use_lightning=False),
    "s10":   dict(in_channels=11, out_channels=11, mode="satellite", use_lightning=True),
}


def build(modality, mods, suffix, lever):
    cfg = OmegaConf.load(TEMPLATE)
    cell = f"{modality}_{suffix}"
    out_dir = f"{EXP_ROOT}/{cell}"

    cfg.experiment.project = PROJECT
    cfg.experiment.name = cell
    cfg.experiment.output_dir = out_dir
    cfg.experiment.logging_dir = f"{out_dir}/logs"
    cfg.experiment.save_every = 5000      # slot="latest"
    cfg.experiment.eval_every = 2000      # val metrics + best_val  -> intermediate results overnight
    cfg.experiment.generate_every = 5000
    cfg.experiment.log_every = 200

    cfg.model.vq_model.in_channels = mods["in_channels"]
    cfg.model.vq_model.out_channels = mods["out_channels"]
    cfg.dataset.params.mode = mods["mode"]
    cfg.dataset.params.use_lightning = mods["use_lightning"]

    cfg.training.max_train_steps = SCREEN_STEPS
    cfg.lr_scheduler.params.warmup_steps = WARMUP

    if lever is not None:
        lever(cfg.losses)

    dest = REPO / "configs" / f"rs_{cell}.yaml"
    OmegaConf.save(cfg, dest)
    return dest.name, cell


def build_smoke(modality):
    """Tiny 60-step config for the pre-launch correctness gate."""
    mods = MODALITIES[modality]
    cfg = OmegaConf.load(TEMPLATE)
    cfg.experiment.project = PROJECT
    cfg.experiment.name = f"smoke_{modality}"
    cfg.experiment.output_dir = f"/tmp/rs_smoke_{modality}"
    cfg.experiment.logging_dir = f"/tmp/rs_smoke_{modality}/logs"
    cfg.experiment.save_every = 30
    cfg.experiment.eval_every = 30
    cfg.experiment.generate_every = 30
    cfg.experiment.log_every = 10
    cfg.model.vq_model.in_channels = mods["in_channels"]
    cfg.model.vq_model.out_channels = mods["out_channels"]
    cfg.dataset.params.mode = mods["mode"]
    cfg.dataset.params.use_lightning = mods["use_lightning"]
    cfg.training.max_train_steps = 60
    cfg.lr_scheduler.params.warmup_steps = 10
    dest = REPO / "configs" / f"rs_smoke_{modality}.yaml"
    OmegaConf.save(cfg, dest)
    return dest.name


def main():
    made = []
    for modality, mods in MODALITIES.items():
        for suffix, lever in LEVERS.items():
            made.append(build(modality, mods, suffix, lever))
    smoke = [build_smoke("radar"), build_smoke("s10")]
    print(f"Generated {len(made)} training configs:")
    for fn, cell in made:
        print(f"  configs/{fn}  ->  {EXP_ROOT}/{cell}")
    print(f"Smoke configs: {smoke}")


if __name__ == "__main__":
    main()
