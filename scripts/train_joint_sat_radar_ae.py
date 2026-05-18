"""Joint sat + radar FlowTiTok AE trainer with a cross-modal token similarity loss.

Design: docs/specs/2026-05-18-joint-sat-radar-tokenizer-design.md

One script trains a sat AE (11ch) and a radar AE (1ch) on the SAME time-paired
frames, SAME seed/arch/step-count/order. The only knob that differs between the
two experiment groups is `joint.sim_weight`:

    sim_weight == 0  ->  Group A (separate baseline; no gradient crosses modalities)
    sim_weight  > 0  ->  Group B (joint; index-wise cosine alignment on posterior means)

It deliberately reuses the battle-tested builders from utils.train_utils
(`get_config`, `create_model_and_loss_module`, CLIP text guidance, `EMAModel`)
but owns the per-step loop so the two-model + similarity coupling is explicit
and fully reviewable. Checkpoints are written as plain `pytorch_model.bin`
under <out>/{sat,radar}/checkpoint-{latest,best_val,final}/ so the v2v configs
load them exactly like any other frozen FlowTiTok tokenizer.

Run:
  CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
    scripts/train_joint_sat_radar_ae.py --config=configs/joint_ae_joint_lab2.yaml
"""
import copy
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.dataset import SatelliteRadarNpyDataset, collate_sat2radar_v2v
from utils.logger import setup_logger
from utils.train_utils import (
    get_config,
    create_clip_model,
    create_model_and_loss_module,
    _encode_text_with_clip,
)


# --------------------------------------------------------------------------- #
# Config derivation                                                           #
# --------------------------------------------------------------------------- #
def _derive_modality_config(base, in_ch, out_ch, name_suffix):
    """Clone the shared config and specialise it for one modality.

    Only in/out channels and the experiment name change; arch / recipe / data
    are shared so Group A vs B differ by exactly one knob (sim_weight).
    """
    cfg = copy.deepcopy(base)
    cfg.model.vq_model.in_channels = in_ch
    cfg.model.vq_model.out_channels = out_ch
    cfg.experiment.name = f"{base.experiment.name}_{name_suffix}"
    return cfg


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
def _build_loader(filelist_path, split, batch_size, num_workers, augment, seed):
    """Paired single-frame loader (mode=sat2radar_v2v, num_frames=1).

    Yields sat_video [B,1,11,H,W] + radar_video [B,1,1,H,W] for the SAME
    timestamp — the pairing the similarity loss needs.
    """
    ds = SatelliteRadarNpyDataset(
        filelist_path=filelist_path,
        filelist_split=split,
        mode="sat2radar_v2v",
        num_frames=1,
        frame_stride=1,
        use_lightning=True,
        ir_band_indices=None,
        augment=augment,
        augment_hflip=augment,
        augment_vflip=augment,
    )
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_sat2radar_v2v,
        drop_last=(split == "train"),
        pin_memory=True,
        generator=g,
    )


def _split_batch(batch, device, crop_size):
    """[B,1,C,H,W] -> [B,C,crop,crop] for sat & radar; build per-modality text."""
    sat = batch["sat_video"][:, 0].to(device, non_blocking=True).float()
    radar = batch["radar_video"][:, 0].to(device, non_blocking=True).float()
    if sat.shape[-1] != crop_size or sat.shape[-2] != crop_size:
        sat = F.interpolate(sat, size=(crop_size, crop_size),
                            mode="bilinear", align_corners=False)
        radar = F.interpolate(radar, size=(crop_size, crop_size),
                              mode="bilinear", align_corners=False)
    sat_paths = [p[0] if isinstance(p, (list, tuple)) else p
                 for p in batch.get("sat_paths", [])]
    radar_paths = [p[0] if isinstance(p, (list, tuple)) else p
                   for p in batch.get("radar_paths", [])]
    sat_text = [f"A multispectral satellite infrared and lightning image from "
                f"{os.path.basename(str(p))}." for p in sat_paths]
    radar_text = [f"A radar reflectivity image from "
                  f"{os.path.basename(str(p))}." for p in radar_paths]
    return sat, radar, sat_text, radar_text


# --------------------------------------------------------------------------- #
# Similarity loss                                                             #
# --------------------------------------------------------------------------- #
def index_wise_cosine_loss(post_sat, post_radar):
    """1 - mean_k cos( S[:,:,k], R[:,:,k] ) on posterior means.

    posteriors.mean is [B, token_size, 1, N]; squeeze the singleton and take
    cosine over the token_size axis, per latent-token index k (the exact axis
    the v2v flow operates on). Gradients flow into BOTH encoders by design.
    """
    s = post_sat.mean.squeeze(2)      # [B, token_size, N]
    r = post_radar.mean.squeeze(2)    # [B, token_size, N]
    cos = F.cosine_similarity(s, r, dim=1)   # [B, N]
    return (1.0 - cos).mean()


def index_wise_infonce_loss(post_sat, post_radar, logit_scale):
    """Per-index symmetric InfoNCE on posterior means (anti-collapse).

    For each latent-token index k the same-timestamp (sat_k, radar_k) pair is
    the positive; the same-index tokens of the other B-1 batch samples are the
    negatives. A collapsed code cannot tell sample i from j, so this loss
    *rises* under collapse instead of vanishing (unlike index-wise cosine —
    see design §3.3). `logit_scale` is the learnable CLIP log-temperature; it
    is `.exp()`d here (caller clamps it ≤ ln(100) per step).
    """
    s = post_sat.mean.squeeze(2).permute(2, 0, 1)     # [N, B, D]
    r = post_radar.mean.squeeze(2).permute(2, 0, 1)    # [N, B, D]
    N, B, _ = s.shape
    if B < 2:                       # no negatives → InfoNCE undefined; no-op
        return s.new_zeros(())
    s = F.normalize(s, dim=-1)
    r = F.normalize(r, dim=-1)
    logits = logit_scale.exp() * torch.bmm(s, r.transpose(1, 2))   # [N,B,B]
    tgt = torch.arange(B, device=s.device).expand(N, B).reshape(-1)
    s2r = F.cross_entropy(logits.reshape(N * B, B), tgt)              # sat→radar
    r2s = F.cross_entropy(logits.transpose(1, 2).reshape(N * B, B), tgt)  # radar→sat
    return 0.5 * (s2r + r2s)


# --------------------------------------------------------------------------- #
# Checkpoint slots (plain pytorch_model.bin, atomic)                          #
# --------------------------------------------------------------------------- #
def save_slot(model, ema_model, accelerator, base_dir, slot, step, logger,
              use_ema):
    """Write EMA(or raw) weights to <base_dir>/checkpoint-<slot>/pytorch_model.bin."""
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(model)
    if use_ema and ema_model is not None:
        ema_model.store(unwrapped.parameters())
        ema_model.copy_to(unwrapped.parameters())
    out = Path(base_dir) / f"checkpoint-{slot}"
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "pytorch_model.bin.tmp"
    torch.save(unwrapped.state_dict(), tmp)
    os.replace(tmp, out / "pytorch_model.bin")
    OmegaConf.save(OmegaConf.create({"global_step": step, "slot": slot}),
                   out / "metadata.yaml")
    if use_ema and ema_model is not None:
        ema_model.restore(unwrapped.parameters())
    logger.info(f"[ckpt] {slot} -> {out/'pytorch_model.bin'} (step {step})")


# --------------------------------------------------------------------------- #
# Validation (lightweight paired recon L2, EMA weights)                       #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_paired_l2(sat_model, radar_model, sat_ema, radar_ema, loader,
                   accelerator, crop_size, clip_tok, clip_enc, max_batches,
                   use_ema):
    dev = accelerator.device
    models = [(sat_model, sat_ema), (radar_model, radar_ema)]
    for m, e in models:
        mu = accelerator.unwrap_model(m)
        if use_ema and e is not None:
            e.store(mu.parameters()); e.copy_to(mu.parameters())
        m.eval()
    sat_se = radar_se = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        sat, radar, st, rt = _split_batch(batch, dev, crop_size)
        tg_s = _encode_text_with_clip(st, clip_tok, clip_enc, dev)
        tg_r = _encode_text_with_clip(rt, clip_tok, clip_enc, dev)
        sat_rec, _ = sat_model(sat, tg_s)
        radar_rec, _ = radar_model(radar, tg_r)
        sat_se += F.mse_loss(sat_rec, sat).item()
        radar_se += F.mse_loss(radar_rec, radar).item()
        n += 1
    for m, e in models:
        mu = accelerator.unwrap_model(m)
        if use_ema and e is not None:
            e.restore(mu.parameters())
        m.train()
    n = max(n, 1)
    return sat_se / n, radar_se / n


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    config = get_config()
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    out_dir = config.experiment.output_dir
    os.makedirs(out_dir, exist_ok=True)
    config.experiment.logging_dir = os.path.join(out_dir, "logs")

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        logging_dir=config.experiment.logging_dir,
        split_batches=False,
    )
    logger = setup_logger(name="JointSatRadarAE", log_level="INFO",
                          output_file=f"{out_dir}/log{accelerator.process_index}.txt")

    sim_weight = float(config.joint.sim_weight)
    # Backward-compat: old configs lack these → default to the original
    # cosine path with no warmup, so existing A/B reproduce bit-for-bit.
    sim_loss_kind = str(config.joint.get("sim_loss", "cosine"))
    infonce_warmup = int(config.joint.get("infonce_warmup_steps", 1000))
    infonce_init_temp = float(config.joint.get("infonce_init_temp", 0.07))
    use_infonce = (sim_loss_kind == "infonce") and (sim_weight > 0.0)
    use_ema = bool(config.training.use_ema)
    crop = int(config.dataset.preprocessing.crop_size)
    if accelerator.is_main_process:
        logger.info(
            f"sim_weight={sim_weight} sim_loss={sim_loss_kind} "
            f"({'JOINT (Group B)' if sim_weight > 0 else 'SEPARATE (Group A)'})"
            + (f" | InfoNCE: init_temp={infonce_init_temp} "
               f"warmup={infonce_warmup}" if use_infonce else ""))
        OmegaConf.save(config, Path(out_dir) / "config.yaml")

    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)

    # Two models, identical arch, different channels — built by the proven builder.
    sat_cfg = _derive_modality_config(
        config, config.joint.sat_in_channels, config.joint.sat_out_channels, "sat")
    radar_cfg = _derive_modality_config(
        config, config.joint.radar_in_channels, config.joint.radar_out_channels, "radar")
    sat_model, sat_ema, sat_loss = create_model_and_loss_module(
        sat_cfg, logger, accelerator, model_type="flowtitok")
    radar_model, radar_ema, radar_loss = create_model_and_loss_module(
        radar_cfg, logger, accelerator, model_type="flowtitok")
    clip_encoder, clip_tokenizer = create_clip_model()

    # One optimizer over both AEs + non-discriminator loss params (matches the
    # run1 single-model recipe, just doubled). Discriminator is disabled via
    # discriminator_start >> max_train_steps so its params never get grad.
    def trainable(module, drop_disc=False):
        for nm, p in module.named_parameters():
            if not p.requires_grad:
                continue
            if drop_disc and "discriminator" in nm:
                continue
            yield p
    params = (list(trainable(sat_model)) + list(trainable(radar_model))
              + list(trainable(sat_loss, True)) + list(trainable(radar_loss, True)))
    # Learnable CLIP log-temperature for InfoNCE (design §3.3). Created always
    # (1 scalar, harmless) but only optimized/clamped when InfoNCE is active,
    # so the sim_weight=0 separate baseline stays a clean disjoint-param
    # ablation (no extra optimized param, no cross-modal gradient).
    logit_scale = torch.nn.Parameter(
        torch.tensor(math.log(1.0 / infonce_init_temp),
                     device=accelerator.device))
    param_groups = [{"params": params}]
    if use_infonce:
        param_groups.append({"params": [logit_scale], "weight_decay": 0.0})
    opt = torch.optim.AdamW(
        param_groups,
        lr=config.optimizer.params.learning_rate,
        betas=(config.optimizer.params.beta1, config.optimizer.params.beta2),
        weight_decay=config.optimizer.params.weight_decay,
    )
    max_steps = int(config.training.max_train_steps)
    warmup = int(config.lr_scheduler.params.warmup_steps)
    end_ratio = float(config.lr_scheduler.params.end_lr) / float(
        config.optimizer.params.learning_rate)

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        prog = (step - warmup) / max(max_steps - warmup, 1)
        return end_ratio + 0.5 * (1 - end_ratio) * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_loader = _build_loader(
        config.dataset.params.filelist_path, "train",
        config.training.per_gpu_batch_size, config.dataset.params.num_workers,
        augment=True, seed=config.training.seed or 42)
    val_loader = _build_loader(
        config.joint.val_filelist_path, "val",
        config.training.per_gpu_batch_size, config.dataset.params.num_workers,
        augment=False, seed=123)

    sat_model, radar_model, sat_loss, radar_loss, clip_encoder, opt = accelerator.prepare(
        sat_model, radar_model, sat_loss, radar_loss, clip_encoder, opt)
    if use_ema:
        sat_ema.to(accelerator.device)
        radar_ema.to(accelerator.device)

    best_val = float("inf")
    step = 0
    t0 = time.time()
    logger.info(f"***** Joint AE training: {max_steps} steps, "
                f"batch {config.training.per_gpu_batch_size} *****")

    sat_model.train(); radar_model.train()
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            sat, radar, st, rt = _split_batch(batch, accelerator.device, crop)
            with torch.no_grad():
                tg_s = _encode_text_with_clip(st, clip_tokenizer, clip_encoder,
                                              accelerator.device)
                tg_r = _encode_text_with_clip(rt, clip_tokenizer, clip_encoder,
                                              accelerator.device)
            with accelerator.accumulate([sat_model, radar_model,
                                         sat_loss, radar_loss]):
                sat_rec, sat_post = sat_model(sat, tg_s)
                radar_rec, radar_post = radar_model(radar, tg_r)
                l_sat, sat_d = sat_loss(sat, sat_rec, sat_post, step,
                                        mode="generator")
                l_radar, radar_d = radar_loss(radar, radar_rec, radar_post,
                                              step, mode="generator")
                if sim_weight > 0.0:
                    if use_infonce:
                        l_sim = index_wise_infonce_loss(
                            sat_post, radar_post, logit_scale)
                        ramp = (min(1.0, step / infonce_warmup)
                                if infonce_warmup > 0 else 1.0)
                    else:
                        l_sim = index_wise_cosine_loss(sat_post, radar_post)
                        ramp = 1.0          # cosine path: no warmup (compat)
                else:
                    l_sim = torch.zeros((), device=accelerator.device)
                    ramp = 0.0
                eff_w = sim_weight * ramp
                loss = l_sat + l_radar + eff_w * l_sim
                accelerator.backward(loss)
                if config.training.max_grad_norm and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params,
                                                config.training.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                if use_infonce:                  # CLIP: clamp temp ≤ ln(100)
                    with torch.no_grad():
                        logit_scale.clamp_(max=math.log(100.0))

            if accelerator.sync_gradients:
                if use_ema:
                    sat_ema.step(accelerator.unwrap_model(sat_model).parameters())
                    radar_ema.step(accelerator.unwrap_model(radar_model).parameters())
                step += 1

                if step % config.experiment.log_every == 0:
                    sps = config.training.per_gpu_batch_size * step / (time.time() - t0)
                    tmsg = (f" temp {logit_scale.exp().item():.2f}"
                            if use_infonce else "")
                    logger.info(
                        f"step {step}/{max_steps} | loss {loss.item():.4f} "
                        f"| sat {l_sat.item():.4f} radar {l_radar.item():.4f} "
                        f"| sim {float(l_sim):.4f} ({sim_loss_kind} "
                        f"w={sim_weight} eff={eff_w:.4f}){tmsg} "
                        f"| lr {sched.get_last_lr()[0]:.2e} | {sps:.1f} im/s")

                if step % config.experiment.save_every == 0:
                    save_slot(sat_model, sat_ema, accelerator,
                              f"{out_dir}/sat", "latest", step, logger, use_ema)
                    save_slot(radar_model, radar_ema, accelerator,
                              f"{out_dir}/radar", "latest", step, logger, use_ema)

                if step % config.experiment.eval_every == 0:
                    s_l2, r_l2 = eval_paired_l2(
                        sat_model, radar_model, sat_ema, radar_ema, val_loader,
                        accelerator, crop, clip_tokenizer, clip_encoder,
                        int(config.joint.val_max_batches), use_ema)
                    comb = s_l2 + r_l2
                    logger.info(f"[val] step {step} sat_L2={s_l2:.6f} "
                                f"radar_L2={r_l2:.6f} combined={comb:.6f} "
                                f"(best={best_val:.6f})")
                    if comb < best_val:
                        best_val = comb
                        logger.info(f"[BEST-VAL] new best {comb:.6f} @ {step}")
                        save_slot(sat_model, sat_ema, accelerator,
                                  f"{out_dir}/sat", "best_val", step, logger,
                                  use_ema)
                        save_slot(radar_model, radar_ema, accelerator,
                                  f"{out_dir}/radar", "best_val", step, logger,
                                  use_ema)

    accelerator.wait_for_everyone()
    save_slot(sat_model, sat_ema, accelerator, f"{out_dir}/sat", "final",
              step, logger, use_ema)
    save_slot(radar_model, radar_ema, accelerator, f"{out_dir}/radar", "final",
              step, logger, use_ema)
    logger.info(f"Done. best combined val L2 = {best_val:.6f}")
    # NOTE: no accelerator.init_trackers() in this script, so
    # accelerator.end_training() raises AttributeError('trackers') on
    # accelerate 0.12.0 *after* all work+ckpts are saved. Guard it so the
    # process exits 0 (the earlier overnight run trained fine but exited 1
    # here, producing misleading FAIL markers in the orchestrator log).
    try:
        accelerator.end_training()
    except AttributeError:
        pass


if __name__ == "__main__":
    main()
