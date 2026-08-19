#!/usr/bin/env python3
"""Pure-inference GPU latency timer for FlowTok sat->radar arms.

Measures ONLY the model inference cost (tokenizer encode -> flow/diffusion
sampler loop -> tokenizer decode + CLIP text guidance). NO metrics, NO data I/O,
NO dataset/filelist loading. Synthetic random input tensors of the correct shape
are used (inference latency is input-content-independent).

This script does NOT modify scripts/test_sat2radar_v2v.py. It IMPORTS that module
to reuse its real, module-level helpers (`load_py_config`, `_ae_config`,
`encode_video_with_autoencoder`) and it faithfully replicates the body of the
nested `infer_batch(...)` closure (test_sat2radar_v2v.py:812-960) -- the smallest
real code path that runs encode -> sample -> decode. The sampler itself is the
REAL class `ODEEulerFlowMatchingSolver` (flow arms) / `TokenDiffusion.ddim_sample`
(factddpm), never reimplemented. Model + tokenizer + CLIP loading mirror
test_sat2radar_v2v.py:555-698 line-for-line.

Run under conda env `flowtok`. Required OFFLINE env vars (set these in the PBS
script before launching, exactly as the repo's eval jobs do):

    export HF_HOME="/scratch/kl02/$USER/hf_cache"
    export TRANSFORMERS_CACHE="$HF_HOME"
    export TORCH_HOME="$HF_HOME"
    export XDG_CACHE_HOME="/scratch/kl02/yh0308/hf_cache"
    export HF_HUB_OFFLINE=1
    export WANDB_MODE=disabled
    export OPENCLIP_LOCAL_CKPT="$HF_HOME/hub/models--timm--vit_large_patch14_clip_336.openai/snapshots/81e38efc4637de5023b10e75a7f9bd1c6fa6b010/open_clip_pytorch_model.bin"
    source /scratch/kl02/$USER/miniconda3/etc/profile.d/conda.sh
    conda activate flowtok

Example:
    python3 time_flowtok_infer.py \
        --config configs/Sat2Radar-v2v-cmp-fact-B-bl128-cond3nan1_gadi.py \
        --ckpt   /scratch/kl02/yh0308/Projv2v/Experiments/sat2radar_flowtok_v2v_cmp_fact_B_bl128_cond3nan1/ckpts/600000.ckpt \
        --arm fact-v2v --mode v2v --nfe 20 --out timings.jsonl --reps 5
"""
import argparse
import json
import os
import statistics
import sys
import time

# ── Make FlowTok repo + scripts/ importable, then reuse the REAL eval module. ──
FT_ROOT = os.path.dirname(os.path.abspath(__file__))
if FT_ROOT not in sys.path:
    sys.path.insert(0, FT_ROOT)
SCRIPTS_DIR = os.path.join(FT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Real eval module: gives us load_py_config / _ae_config / encode_video_with_autoencoder
import test_sat2radar_v2v as ev  # noqa: E402

import flow_utils  # noqa: E402
from diffusion.flow_matching import ODEEulerFlowMatchingSolver  # noqa: E402
from libs.flowtitok import FlowTiTok  # noqa: E402
from libs.adapters import AdapterIn, AdapterOut  # noqa: E402
import open_clip  # noqa: E402


def build_args():
    p = argparse.ArgumentParser("FlowTok pure-inference latency timer")
    p.add_argument("--config", required=True, help="configs/*.py with get_config()")
    p.add_argument("--ckpt", required=True,
                   help="TrainState ckpt DIRECTORY, e.g. <exp>/ckpts/600000.ckpt")
    p.add_argument("--arm", required=True, help="Arm label for the JSON row")
    p.add_argument("--mode", required=True, choices=["i2i", "v2v"])
    p.add_argument("--nfe", type=int, required=True,
                   help="Number of function evals for THIS arm's production sampler "
                        "(flow: config.sample.sample_steps; diffusion: config.diffusion.sample_steps)")
    p.add_argument("--out", required=True, help="Append one JSON line here")
    p.add_argument("--reps", type=int, default=5, help="Timed runs (median reported)")
    p.add_argument("--warmup", type=int, default=2, help="Discarded warmup runs")
    p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES override, e.g. '0'")
    p.add_argument("--algo", default="auto", choices=["auto", "flow", "diffusion"],
                   help="Sampler family. 'auto' = read config.generation_algorithm.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = build_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    ev._seed_all(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Config + NFE override (mirrors test_sat2radar_v2v.py:536-547) ──────────
    config = ev.load_py_config(args.config)
    gen_algo = getattr(config, "generation_algorithm", "flow_matching")
    algo = args.algo
    if algo == "auto":
        algo = "diffusion" if gen_algo == "diffusion" else "flow"
    if algo == "diffusion":
        config.diffusion.sample_steps = int(args.nfe)
        print(f"[NFE] diffusion.sample_steps = {args.nfe}")
    else:
        config.sample.sample_steps = int(args.nfe)
        print(f"[NFE] flow sample.sample_steps = {args.nfe}")

    T = 1 if args.mode == "i2i" else int(config.dataset.get("num_frames", 16))
    frames_per_sample = 1 if args.mode == "i2i" else 16

    # ── Backbone (nnet_ema) — mirrors test_sat2radar_v2v.py:555-576 ────────────
    train_state = flow_utils.initialize_train_state(config, device)
    train_state.load(args.ckpt)
    nnet = train_state.nnet.to(device)
    nnet_ema = train_state.nnet_ema.to(device)
    nnet_ema.eval()

    _need_2L_pos_per_frame = (
        (getattr(config, "cond_use_sat_lightning_tokens", False)
         and getattr(config, "cond_token_fusion", "mean") == "seqconcat")
        or getattr(config, "flow_cond_mode", "none") == "token_concat_interleaved"
    )
    if _need_2L_pos_per_frame:
        _pos_override = 2 * int(config.vq_model.num_latent_tokens)
        nnet.pos_n_per_frame = _pos_override
        nnet_ema.pos_n_per_frame = _pos_override
        print(f"[POS] interleaved/seqconcat layout: set pos_n_per_frame={_pos_override}")

    # ── Optional adapters — mirrors test_sat2radar_v2v.py:578-626 ──────────────
    adapter_in_satellite = None
    adapter_out = None
    adapter_in_sat_cfg = getattr(config, "adapter_in_satellite", None)
    if adapter_in_sat_cfg is None:
        adapter_in_sat_cfg = getattr(config, "adapter_in", None)
    if adapter_in_sat_cfg and adapter_in_sat_cfg.get("enabled", False):
        adapter_in_satellite = AdapterIn(
            in_channels=int(adapter_in_sat_cfg.get("in_channels", getattr(config, "sat_in_channels", 3))),
            out_channels=int(getattr(config, "sat_in_channels", 3)),
            mid_channels=int(adapter_in_sat_cfg.get("mid_channels", 32)),
            num_blocks=int(adapter_in_sat_cfg.get("num_blocks", 3)),
        ).to(device)
        p1 = os.path.join(args.ckpt, "adapter_in_satellite.pth")
        p2 = os.path.join(args.ckpt, "adapter_in.pth")
        if os.path.isfile(p1):
            adapter_in_satellite.load_state_dict(torch.load(p1, map_location="cpu"))
            adapter_in_satellite.eval()
        elif os.path.isfile(p2):
            adapter_in_satellite.load_state_dict(torch.load(p2, map_location="cpu"))
            adapter_in_satellite.eval()
        else:
            adapter_in_satellite = None
    if getattr(config, "adapter_out", None) and config.adapter_out.get("enabled", False):
        adapter_out = AdapterOut(
            in_channels=int(getattr(config, "radar_out_channels", 3)),
            out_channels=1,
            mid_channels=int(config.adapter_out.get("mid_channels", 16)),
            num_blocks=int(config.adapter_out.get("num_blocks", 2)),
        ).to(device)
        po = os.path.join(args.ckpt, "adapter_out.pth")
        if os.path.isfile(po):
            adapter_out.load_state_dict(torch.load(po, map_location="cpu"))
            adapter_out.eval()
        else:
            adapter_out = None

    # ── Tokenizers — mirrors test_sat2radar_v2v.py:628-668 ─────────────────────
    sat_in_ch = getattr(config, "sat_in_channels", None)
    sat_out_ch = getattr(config, "sat_out_channels", None)
    radar_in_ch = getattr(config, "radar_in_channels", None)
    radar_out_ch = getattr(config, "radar_out_channels", None)
    if sat_in_ch is None or sat_out_ch is None:
        sat_in_ch, sat_out_ch = 11, 11
    if radar_in_ch is None or radar_out_ch is None:
        radar_in_ch, radar_out_ch = 1, 1

    if getattr(config, "pixel_space", False):
        from libs.patchify_tokenizer import PixelPatchifier
        _P = int(getattr(config, "patch_size", 8))
        _crop = int(getattr(config, "ae_image_size", 128))
        sat_autoencoder = PixelPatchifier(_P, sat_in_ch, crop_size=_crop).to(device)
        radar_autoencoder = PixelPatchifier(_P, radar_in_ch, crop_size=_crop).to(device)
        sat_autoencoder.eval(); sat_autoencoder.requires_grad_(False)
        radar_autoencoder.eval(); radar_autoencoder.requires_grad_(False)
    else:
        sat_ae_config = ev._ae_config(config, sat_in_ch, sat_out_ch)
        radar_ae_config = ev._ae_config(config, radar_in_ch, radar_out_ch)
        sat_autoencoder = FlowTiTok(sat_ae_config).to(device)
        sat_autoencoder.load_state_dict(
            torch.load(config.sat_tokenizer_checkpoint, map_location="cpu"), strict=False)
        sat_autoencoder.eval(); sat_autoencoder.requires_grad_(False)
        radar_autoencoder = FlowTiTok(radar_ae_config).to(device)
        radar_autoencoder.load_state_dict(
            torch.load(config.radar_tokenizer_checkpoint, map_location="cpu"), strict=False)
        radar_autoencoder.eval(); radar_autoencoder.requires_grad_(False)

    # ── CLIP text encoder — mirrors test_sat2radar_v2v.py:670-698 ──────────────
    clip_model_name = "ViT-L-14-336"
    local_clip_ckpt = os.environ.get("OPENCLIP_LOCAL_CKPT", None)
    try:
        if local_clip_ckpt and os.path.isfile(local_clip_ckpt):
            clip_encoder, _, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained=None)
            state_dict = torch.load(local_clip_ckpt, map_location="cpu")
            clip_encoder.load_state_dict(state_dict, strict=False)
        else:
            clip_encoder, _, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained="openai")
        del clip_encoder.visual
        clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
        clip_encoder.transformer.batch_first = False
        clip_encoder.eval(); clip_encoder.requires_grad_(False)
        clip_encoder.to(device)
    except Exception as e:
        clip_encoder = None
        clip_tokenizer = None
        print(f"[WARN] open_clip unavailable, decoder runs without text guidance: {e}")

    num_latent_tokens = config.vq_model.num_latent_tokens
    guidance_scale = config.sample.scale

    # ── build_condition_tokens_from_sat_video — test_sat2radar_v2v.py:704-730 ──
    def build_condition_tokens_from_sat_video(sat_video):
        use_sat_lgt_tokens = getattr(config, "cond_use_sat_lightning_tokens", False)
        if not use_sat_lgt_tokens:
            return ev.encode_video_with_autoencoder(
                sat_autoencoder, sat_video, config.vq_model.scale_factor,
                adapter_in=adapter_in_satellite)
        sat_ir_video = sat_video[:, :, :3, :, :]
        lgt_slice = sat_video[:, :, -1:, :, :]
        lgt_video = lgt_slice.repeat(1, 1, 3, 1, 1)
        sat_ir_tokens = ev.encode_video_with_autoencoder(
            sat_autoencoder, sat_ir_video, config.vq_model.scale_factor, adapter_in=None)
        lgt_tokens = ev.encode_video_with_autoencoder(
            sat_autoencoder, lgt_video, config.vq_model.scale_factor, adapter_in=None)
        fusion = getattr(config, "cond_token_fusion", "mean")
        if fusion == "sum":
            return sat_ir_tokens + lgt_tokens
        return 0.5 * (sat_ir_tokens + lgt_tokens)

    # ── One full inference call — replicates infer_batch:812-960 (no metrics) ──
    @torch.no_grad()
    def run_infer(sat_video, radar_video_gt, radar_paths):
        B, T_max, C_sat, H, W = sat_video.shape
        sat_tokens = build_condition_tokens_from_sat_video(sat_video)

        use_text_vae_encoder = getattr(config, "use_text_vae_encoder", True)
        if use_text_vae_encoder:
            x0, _, _ = nnet_ema(sat_tokens, text_encoder=True)
        else:
            x0 = sat_tokens
        if config.nnet.model_args.noising_type != "none":
            x0 = x0 + torch.randn_like(x0) * config.sample.noise_scale

        if algo == "diffusion":
            from diffusion.token_diffusion import TokenDiffusion
            _dcfg = config.diffusion
            _td = TokenDiffusion(
                train_timesteps=int(_dcfg.get("train_timesteps", 1000)),
                schedule=_dcfg.get("schedule", "linear"),
                target=_dcfg.get("target", "pred_x0"),
                gamma=_dcfg.get("gamma", "ddim"),
                cond_mode=_dcfg.get("cond_mode", "chn_concat"),
            ).to(sat_tokens.device)
            z = _td.ddim_sample(nnet_ema, cond=sat_tokens,
                                sample_steps=int(_dcfg.get("sample_steps", 500)))
        else:
            _flow_cond_mode = getattr(config, "flow_cond_mode", "none")
            _tc_family = ("token_concat", "token_concat_interleaved", "token_concat_modality")
            _uses_cond = _flow_cond_mode in _tc_family or _flow_cond_mode == "cross_attention"
            if _uses_cond:
                _c_out = int(config.nnet.model_args.channels)
                x_T_init = torch.randn(sat_tokens.shape[0], sat_tokens.shape[1], _c_out,
                                       device=sat_tokens.device, dtype=sat_tokens.dtype)
            else:
                x_T_init = x0
            ode_solver = ODEEulerFlowMatchingSolver(
                nnet_ema, step_size_type="step_in_dsigma",
                guidance_scale=guidance_scale, x1_snap_t=-1.0)
            _sample_kwargs = dict(
                x_T=x_T_init, batch_size=B, sample_steps=config.sample.sample_steps,
                unconditional_guidance_scale=guidance_scale,
                has_null_indicator=guidance_scale > 1.0,
                prediction_target=getattr(config, "flow_prediction_target", "velocity"),
                flow_cond_mode=_flow_cond_mode,
                cond_tokens=sat_tokens if _uses_cond else None,
            )
            if _flow_cond_mode == "token_concat_interleaved":
                _sample_kwargs["cond_num_latent_tokens"] = num_latent_tokens
            z, _ = ode_solver.sample(**_sample_kwargs)

        L = z.shape[1]
        assert L % num_latent_tokens == 0
        T_eff = L // num_latent_tokens
        z = z.view(B, T_eff, num_latent_tokens, z.shape[2])
        z = z.view(B * T_eff, num_latent_tokens, z.shape[3])
        z = z.permute(0, 2, 1).unsqueeze(2)

        # CLIP text guidance (dummy but REAL encode path) — infer_batch:895-931
        text_guidance = None
        if clip_encoder is not None and clip_tokenizer is not None:
            texts = []
            for i in range(B):
                for t in range(T_eff):
                    fname = "unknown"
                    if radar_paths and i < len(radar_paths):
                        paths_i = radar_paths[i]
                        if isinstance(paths_i, (list, tuple)) and t < len(paths_i):
                            fname = os.path.basename(str(paths_i[t]))
                    texts.append(f"A radar reflectivity image from {fname}.")
            if len(texts) == B * T_eff:
                try:
                    text_tokens = clip_tokenizer(texts).to(device)
                    cast_dtype = clip_encoder.transformer.get_cast_dtype()
                    text_tokens = clip_encoder.token_embedding(text_tokens).to(cast_dtype)
                    text_tokens = text_tokens + clip_encoder.positional_embedding.to(cast_dtype)
                    text_tokens = text_tokens.permute(1, 0, 2)
                    text_tokens = clip_encoder.transformer(text_tokens, attn_mask=clip_encoder.attn_mask)
                    text_tokens = text_tokens.permute(1, 0, 2)
                    text_tokens = clip_encoder.ln_final(text_tokens)
                    text_guidance = text_tokens
                except Exception as e:
                    print(f"[WARN] CLIP text guidance failed: {e}")
                    text_guidance = None

        radar_pred = radar_autoencoder.decode_tokens(
            z / config.vq_model.scale_factor, text_guidance=text_guidance)
        if adapter_out is not None:
            _, _, _, H_gt0, W_gt0 = radar_video_gt[:, :T_eff].shape
            radar_pred = adapter_out(radar_pred, out_size=(H_gt0, W_gt0))
        else:
            radar_pred = radar_pred[:, 0:1, ...]
        radar_pred = torch.clamp(radar_pred, 0.0, 1.0)
        radar_pred = radar_pred.view(B, T_eff, 1, radar_pred.shape[-2], radar_pred.shape[-1])
        radar_gt = torch.clamp(radar_video_gt[:, :T_eff], 0.0, 1.0)
        _, _, _, H_gt, W_gt = radar_gt.shape
        if radar_pred.shape[-2] != H_gt or radar_pred.shape[-1] != W_gt:
            Bv, Tv, _, H_pred, W_pred = radar_pred.shape
            radar_pred_flat = radar_pred.view(Bv * Tv, 1, H_pred, W_pred)
            radar_pred_flat = F.interpolate(radar_pred_flat, size=(H_gt, W_gt),
                                            mode="bilinear", align_corners=False)
            radar_pred = radar_pred_flat.view(Bv, Tv, 1, H_gt, W_gt)
        return radar_pred

    # ── Synthetic batch (B=1). Correct channel counts: sat=11, radar=1. ────────
    Hs = Ws = int(getattr(config, "ae_image_size", 128))
    C_sat = int(sat_in_ch)  # 11
    sat_video = torch.rand(1, T, C_sat, Hs, Ws, device=device)
    radar_gt = torch.rand(1, T, 1, Hs, Ws, device=device)
    radar_paths = [["synthetic_20200101_000000.npy"] * T]

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[INFO] arm={args.arm} mode={args.mode} algo={algo} nfe={args.nfe} "
          f"T={T} sat_video={tuple(sat_video.shape)} device={gpu_name}")

    # ── Warmup (discarded) ─────────────────────────────────────────────────────
    for _ in range(max(0, args.warmup)):
        run_infer(sat_video, radar_gt, radar_paths)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ── Timed runs ──────────────────────────────────────────────────────────────
    times_ms = []
    for r in range(args.reps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_infer(sat_video, radar_gt, radar_paths)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt_ms)
        print(f"[RUN {r+1}/{args.reps}] {dt_ms:.2f} ms")

    median_ms = float(statistics.median(times_ms))
    row = {
        "arm": args.arm,
        "ms_per_sample": median_ms,
        "ms_per_frame": median_ms / frames_per_sample,
        "frames_per_sample": frames_per_sample,
        "nfe": int(args.nfe),
        "reps": int(args.reps),
        "device": gpu_name,
        "algo": algo,
        "all_ms": times_ms,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[RESULT] {args.arm}: median {median_ms:.2f} ms/sample "
          f"({median_ms / frames_per_sample:.2f} ms/frame, {frames_per_sample} frames) "
          f"nfe={args.nfe} -> appended to {args.out}")


if __name__ == "__main__":
    main()
