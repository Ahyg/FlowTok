import torch

ckpt = "/mnt/ssd_1/yghu/Experiments/radar_flowtok_ae_bl77_vae_run1/checkpoint-250000/unwrapped_model/pytorch_model.bin"
state = torch.load(ckpt, map_location="cpu")

bad = []
for k, v in state.items():
    if torch.isnan(v).any() or torch.isinf(v).any():
        bad.append(k)

print("NaN/Inf params:", bad[:10], "count:", len(bad))
