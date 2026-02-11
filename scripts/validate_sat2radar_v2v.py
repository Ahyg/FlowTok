import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml_collections import ConfigDict  # noqa: E402

from data.dataset import SatelliteRadarNpyDataset, collate_sat2radar_v2v  # noqa: E402
from diffusion.flow_matching import ODEEulerFlowMatchingSolver  # noqa: E402
from libs.flowtitok import FlowTiTok  # noqa: E402
import flow_utils  # noqa: E402
from scripts.test_sat2radar_i2i import encode_video_with_autoencoder, load_py_config, _ae_config  # noqa: E402


def build_eval_dataloader(config, split: str, batch_size: int, mode: str):
    """
    Build dataloader for I2I (T=1) or V2V (T>=1) for visualization only.
    """
    assert mode in ["i2i", "v2v"]
    if mode == "i2i":
        num_frames = 1
    else:
        num_frames = config.dataset.get("num_frames", 16)

    dataset = SatelliteRadarNpyDataset(
        base_dir=None,
        years=None,
        mode="sat2radar_v2v",
        filelist_path=config.dataset.filelist_path,
        filelist_split=split,
        files=None,
        frame_stride=config.dataset.get("frame_stride", 1),
        num_frames=num_frames,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers_per_gpu,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_sat2radar_v2v,
    )
    return loader


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Visualize sat2radar I2I/V2V predictions (no metrics).")
    parser.add_argument("--config", required=True, help="Python config file, e.g. configs/Sat2Radar-Video-XL.py")
    parser.add_argument("--ckpt", required=True, help="TrainState checkpoint dir, e.g. .../100000.ckpt")
    parser.add_argument("--out_dir", required=True, help="Directory to save visualizations")
    parser.add_argument("--split", default="val", help="Dataset split in filelist (train/val/test)")
    parser.add_argument("--mode", default="i2i", choices=["i2i", "v2v"], help="Visualization mode: i2i or v2v")
    parser.add_argument("--max_batches", type=int, default=10, help="Max number of batches to visualize (-1 for all)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES override, e.g. '0'")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_py_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = build_eval_dataloader(config, split=args.split, batch_size=args.batch_size, mode=args.mode)

    # Load model
    train_state = flow_utils.initialize_train_state(config, device)
    train_state.load(args.ckpt)
    nnet_ema = train_state.nnet_ema.to(device)
    nnet_ema.eval()

    # Load AEs
    sat_ae_config = _ae_config(config, 11, 11)
    radar_ae_config = _ae_config(config, 1, 1)

    sat_autoencoder = FlowTiTok(sat_ae_config).to(device)
    sat_autoencoder.load_state_dict(
        torch.load(config.sat_tokenizer_checkpoint, map_location="cpu")
    )
    sat_autoencoder.eval()
    sat_autoencoder.requires_grad_(False)

    radar_autoencoder = FlowTiTok(radar_ae_config).to(device)
    radar_autoencoder.load_state_dict(
        torch.load(config.radar_tokenizer_checkpoint, map_location="cpu")
    )
    radar_autoencoder.eval()
    radar_autoencoder.requires_grad_(False)

    os.makedirs(args.out_dir, exist_ok=True)

    num_latent_tokens = config.vq_model.num_latent_tokens

    def infer_and_save(batch, batch_idx: int):
        sat_video = batch["sat_video"].to(
            device,
            memory_format=torch.contiguous_format,
            non_blocking=True,
        )  # [B, T_max, 11, H, W]
        radar_video_gt = batch["radar_video"].to(
            device,
            memory_format=torch.contiguous_format,
            non_blocking=True,
        )  # [B, T_max, 1, H, W]
        valid_mask = batch.get("valid_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(device)

        B, T_max, C_sat, H, W = sat_video.shape

        # Encode sat video -> tokens [B, T_eff*L, C_tok]
        sat_tokens = encode_video_with_autoencoder(
            sat_autoencoder, sat_video, config.vq_model.scale_factor
        )

        x0, _, _ = nnet_ema(sat_tokens, text_encoder=True)
        if config.nnet.model_args.noising_type != "none":
            x0 = x0 + torch.randn_like(x0) * config.sample.noise_scale

        ode_solver = ODEEulerFlowMatchingSolver(
            nnet_ema,
            step_size_type="step_in_dsigma",
            guidance_scale=config.sample.scale,
        )
        z, _ = ode_solver.sample(
            x_T=x0,
            batch_size=B,
            sample_steps=config.sample.sample_steps,
            unconditional_guidance_scale=config.sample.scale,
            has_null_indicator=True,
        )

        L = z.shape[1]
        assert L % num_latent_tokens == 0, "Sequence length must be multiple of num_latent_tokens"
        T_eff = L // num_latent_tokens

        z = z.view(B, T_eff, num_latent_tokens, z.shape[2])
        z = z.view(B * T_eff, num_latent_tokens, z.shape[3])
        z = z.permute(0, 2, 1).unsqueeze(2)

        radar_pred = radar_autoencoder.decode_tokens(
            z / config.vq_model.scale_factor, text_guidance=None
        )
        radar_pred = torch.clamp(radar_pred, 0.0, 1.0)
        radar_pred = radar_pred.view(B, T_eff, 1, radar_pred.shape[-2], radar_pred.shape[-1])
        radar_gt = torch.clamp(radar_video_gt[:, :T_eff], 0.0, 1.0)

        # For each sample, save first few frames (or all for small T) as [gt | pred]
        max_frames_to_show = 4 if args.mode == "v2v" else 1

        for i in range(B):
            frames = []
            for t in range(T_eff):
                if valid_mask is not None and not valid_mask[i, t]:
                    continue
                if len(frames) >= max_frames_to_show:
                    break
                grid = torch.cat([radar_gt[i, t], radar_pred[i, t]], dim=-1)  # [1, H, 2W]
                frames.append(grid)

            if not frames:
                continue

            # Stack frames vertically: shape [1, H*k, 2W]
            stacked = torch.cat(frames, dim=-2)
            idx = batch_idx * B + i
            out_path = os.path.join(args.out_dir, f"sample_{idx:06d}.png")
            save_image(stacked, out_path)

    for b_idx, batch in enumerate(loader):
        if args.max_batches >= 0 and b_idx >= args.max_batches:
            break
        infer_and_save(batch, b_idx)

    print(f"Saved sat2radar {args.mode} visualizations to {args.out_dir}")


if __name__ == "__main__":
    main()

