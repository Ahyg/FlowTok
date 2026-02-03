import argparse
import os
from types import SimpleNamespace

import ml_collections
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from libs.flowtitok import FlowTiTok
from data.dataset import SatelliteRadarNpyDataset


def build_config(args):
    config = ml_collections.ConfigDict()
    config.dataset = ml_collections.ConfigDict()
    config.dataset.crop_size = args.image_size

    config.vq_model = ml_collections.ConfigDict()
    config.vq_model.token_size = args.token_size
    config.vq_model.vit_enc_model_size = args.vit_enc_model_size
    config.vq_model.vit_dec_model_size = args.vit_dec_model_size
    config.vq_model.vit_enc_patch_size = args.vit_enc_patch_size
    config.vq_model.vit_dec_patch_size = args.vit_dec_patch_size
    config.vq_model.num_latent_tokens = args.num_latent_tokens
    config.vq_model.use_rmsnorm = args.use_rmsnorm
    config.vq_model.use_swiglu = args.use_swiglu
    config.vq_model.return_quantized = args.return_quantized
    config.vq_model.use_pretrained = False
    config.vq_model.text_context_length = args.text_context_length
    config.vq_model.text_embed_dim = args.text_embed_dim
    config.vq_model.in_channels = args.in_channels
    config.vq_model.out_channels = args.out_channels
    return config


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = build_config(args)

    dataset = SatelliteRadarNpyDataset(
        base_dir=args.data_dir,
        years=args.years.split(",") if args.years else None,
        mode=args.mode,
        ir_band_indices=[int(x) for x in args.ir_band_indices.split(",")] if args.ir_band_indices else None,
        use_lightning=args.use_lightning,
        filelist_path=args.filelist_path,
        filelist_split=args.filelist_split,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    model = FlowTiTok(config).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0

    for epoch in range(args.epochs):
        progress = tqdm(dataloader, desc=f"epoch {epoch}")
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            if args.clamp_input:
                images = images.clamp(-1, 1)

            text_guidance = torch.zeros(
                images.shape[0],
                args.text_context_length,
                args.text_embed_dim,
                device=device,
                dtype=images.dtype,
            )

            z_quantized, post = model.encode(images)
            recon = model.decode(z_quantized, text_guidance)

            recon_loss = torch.mean(torch.abs(recon - images))
            kld_loss = post.kl().mean()
            loss = args.l1_weight * recon_loss + args.kld_weight * kld_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            progress.set_postfix(loss=float(loss), recon=float(recon_loss), kld=float(kld_loss))

            if args.save_every > 0 and global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"step-{global_step}.pth")
                torch.save(model.state_dict(), ckpt_path)

        ckpt_path = os.path.join(args.output_dir, f"epoch-{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FlowTiTok autoencoder on satellite/radar npy frames.")
    parser.add_argument("--data_dir", type=str, required=True, help="Base directory with year subfolders of .npy files.")
    parser.add_argument("--years", type=str, default="2021", help="Comma-separated years (ignored if filelist_path set).")
    parser.add_argument("--filelist_path", type=str, default="", help="Path to dataset_filelist.pkl from DatasetBuilder.")
    parser.add_argument("--filelist_split", type=str, default="train", help="Split to use from filelist: train/val/test.")
    parser.add_argument("--mode", type=str, choices=["satellite", "radar"], required=True, help="Which modality to train.")
    parser.add_argument("--ir_band_indices", type=str, default="", help="Comma-separated IR band indices.")
    parser.add_argument("--use_lightning", action="store_true", help="Include lightning channel for satellite.")
    parser.add_argument("--image_size", type=int, default=128, help="Input image size (H=W).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--output_dir", type=str, required=True, help="Checkpoint output directory.")
    parser.add_argument("--save_every", type=int, default=1000, help="Save every N steps (0 to disable).")
    parser.add_argument("--clamp_input", action="store_true", help="Clamp input to [-1, 1].")

    parser.add_argument("--token_size", type=int, default=16)
    parser.add_argument("--vit_enc_model_size", type=str, default="base")
    parser.add_argument("--vit_dec_model_size", type=str, default="large")
    parser.add_argument("--vit_enc_patch_size", type=int, default=16)
    parser.add_argument("--vit_dec_patch_size", type=int, default=16)
    parser.add_argument("--num_latent_tokens", type=int, default=77)
    parser.add_argument("--use_rmsnorm", action="store_true")
    parser.add_argument("--use_swiglu", action="store_true")
    parser.add_argument("--return_quantized", action="store_true")
    parser.add_argument("--text_context_length", type=int, default=77)
    parser.add_argument("--text_embed_dim", type=int, default=768)
    parser.add_argument("--in_channels", type=int, default=None)
    parser.add_argument("--out_channels", type=int, default=None)
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--kld_weight", type=float, default=0.01)

    args = parser.parse_args()

    if args.in_channels is None or args.out_channels is None:
        if args.mode == "satellite":
            args.in_channels = 11 if args.use_lightning else 10
            args.out_channels = args.in_channels
        else:
            args.in_channels = 1
            args.out_channels = 1

    main(args)
