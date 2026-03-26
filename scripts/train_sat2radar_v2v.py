import os
import sys
from pathlib import Path

import accelerate
import torch
import torch.nn.functional as F
from absl import app, flags, logging
from absl import flags
from ml_collections import config_flags, ConfigDict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import cmweather
import cmcrameri
import open_clip

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# Ensure project root is on sys.path so we can import local modules.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import flow_utils
from diffusion.flow_matching import FlowMatching, ODEEulerFlowMatchingSolver
from libs.flowtitok import FlowTiTok
from libs.adapters import AdapterIn, AdapterOut
from data.dataset import SatelliteRadarNpyDataset, collate_sat2radar_v2v
from torch.utils.data import DataLoader


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config",
    None,
    "Config file path for Sat2Radar-Video training.",
    lock_config=False,
)
flags.DEFINE_string(
    "filelist_path",
    None,
    "Optional override for config.dataset.filelist_path.",
)


def encode_video_with_autoencoder(
    autoencoder,
    video,
    scale_factor: float,
    adapter_in=None,
    require_grad_through_encoder: bool = False,
):
    """
    video: [B, T, C, H, W]
    返回: tokens [B, T*L, C_tok]
    """
    B, T, C, H, W = video.shape

    # 将输入 resize 到 AE 预训练时的分辨率（例如 512x512），
    # 避免 FlowTiTok 的 RoPE 位置编码与 checkpoint 中的形状不一致。
    ae_ds_cfg = getattr(getattr(autoencoder, "config", None), "dataset", None)
    if isinstance(ae_ds_cfg, dict):
        target_size = ae_ds_cfg.get("crop_size", H)
    else:
        target_size = getattr(getattr(autoencoder, "config", None), "ae_image_size", H)

    video = video.view(B * T, C, H, W)

    if adapter_in is not None:
        # 由可学习 AdapterIn 完成通道/分辨率适配（例如 4ch@128 -> 3ch@512）
        video = adapter_in(video, target_size)
    else:
        # 通道对齐：FlowTiTok_512 预训练是 3 通道 in / 3 通道 out。
        # 如果 AE 期望 3 通道而当前是 1 通道（雷达），则将单通道复制为 3 通道。
        expected_in_ch = getattr(getattr(autoencoder, "config", None), "vq_model", {}).get(
            "in_channels", C
        )
        if C == 1 and expected_in_ch == 3:
            video = video.repeat(1, 3, 1, 1)  # [B*T, 3, H, W]

        if H != target_size or W != target_size:
            video = F.interpolate(
                video,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
    if require_grad_through_encoder:
        # Enable gradient flow to adapter input (AE params are frozen by requires_grad_(False)).
        z = autoencoder.encode(video)[0] * scale_factor  # [B*T, C_tok, 1, L]
    else:
        with torch.no_grad():
            # FlowTiTok.encode 返回 (z, dict)
            z = autoencoder.encode(video)[0] * scale_factor  # [B*T, C_tok, 1, L]
    z = z.squeeze(2).permute(0, 2, 1)  # [B*T, L, C_tok]
    L = z.shape[1]
    # interpolate 等操作可能导致非 contiguous，view 会报错，改用 reshape 更安全
    z = z.reshape(B, T * L, z.shape[2])   # [B, T*L, C_tok]
    return z


def _cmap_or_fallback(name, fallback="viridis"):
    """Return a valid colormap name, falling back if needed."""
    try:
        plt.get_cmap(name)
        return name
    except ValueError:
        print(f"[WARN] Colormap '{name}' not found, fallback to '{fallback}'")
        return fallback


def _apply_cmap(img2d, cmap_name, vmin, vmax):
    """Apply matplotlib colormap to a 2D array and return RGB image."""
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    return cmap(norm(img2d))[..., :3]


def _save_three_panel_sat_radar(
    sat_2d,
    gt_2d,
    pred_2d,
    cmap_sat,
    cmap_rad,
    ir_min,
    ir_max,
    z_min,
    z_max,
    out_path,
):
    """
    Save a 3-panel composite [sat_IR | radar_gt | radar_pred] using AE-style colormaps.
    """
    sat_rgb = _apply_cmap(sat_2d, cmap_sat, ir_min, ir_max)
    gt_rgb = _apply_cmap(gt_2d, cmap_rad, z_min, z_max)
    pred_rgb = _apply_cmap(pred_2d, cmap_rad, z_min, z_max)
    composite = np.concatenate([sat_rgb, gt_rgb, pred_rgb], axis=1)
    plt.imsave(out_path, composite)


def build_dataloader(config, mode: str, accelerator: accelerate.Accelerator):
    """Build dataloader for real-time v2v (aligned sat↔radar, variable T, T=1 => i2i)."""
    assert mode in ["train", "val"], f"mode must be 'train' or 'val', got {mode}"

    split = "train" if mode == "train" else "val"
    use_v2v = config.dataset.get("v2v", True)

    # 可选：只使用部分 IR 通道、是否拼接闪电通道
    ir_band_indices = config.dataset.get("ir_band_indices", None)
    use_lightning = config.dataset.get("use_lightning", True)

    is_train = (mode == "train")
    aug_cfg = config.dataset.get("augment", None)
    do_augment = is_train and aug_cfg is not None and aug_cfg.get("enabled", False)

    if use_v2v:
        dataset = SatelliteRadarNpyDataset(
            base_dir=None,
            years=None,
            mode="sat2radar_v2v",
            filelist_path=config.dataset.filelist_path,
            filelist_split=split,
            files=None,
            frame_stride=config.dataset.get("frame_stride", 1),
            num_frames=config.dataset.get("num_frames", (1, 8)),  # (min_t, max_t) => i2i + v2v
            ir_band_indices=ir_band_indices,
            use_lightning=use_lightning,
            augment=do_augment,
            augment_hflip=aug_cfg.get("hflip", True) if aug_cfg else True,
            augment_vflip=aug_cfg.get("vflip", True) if aug_cfg else True,
        )
        collate_fn = collate_sat2radar_v2v
    else:
        dataset = SatelliteRadarNpyDataset(
            base_dir=None,
            years=None,
            mode="sat2radar_video",
            filelist_path=config.dataset.filelist_path,
            filelist_split=split,
            files=None,
            history_frames=config.dataset.get("history_frames", None),
            future_frames=config.dataset.get("future_frames", None),
            frame_stride=config.dataset.get("frame_stride", 1),
            ir_band_indices=ir_band_indices,
            use_lightning=use_lightning,
            augment=do_augment,
            augment_hflip=aug_cfg.get("hflip", True) if aug_cfg else True,
            augment_vflip=aug_cfg.get("vflip", True) if aug_cfg else True,
        )
        collate_fn = None

    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size // accelerator.num_processes,
        shuffle=True if mode == "train" else False,
        num_workers=config.dataset.num_workers_per_gpu,
        pin_memory=True,
        drop_last=True if mode == "train" else False,
        collate_fn=collate_fn,
    )
    
    # Log dataset info
    if accelerator.is_main_process:
        dataset_mode = "sat2radar_v2v" if use_v2v else "sat2radar_video"
        num_samples = len(dataset)
        num_batches = len(dataloader)
        logging.info(f"Dataset [{mode.upper()}]:")
        logging.info(f"  Split: {split}")
        logging.info(f"  Mode: {dataset_mode}")
        logging.info(f"  Samples: {num_samples}")
        logging.info(f"  Batches: {num_batches}")
        logging.info(f"  Batch size (per GPU): {config.train.batch_size // accelerator.num_processes}")
        if use_v2v:
            num_frames_cfg = config.dataset.get("num_frames", (1, 8))
            logging.info(f"  Frame stride: {config.dataset.get('frame_stride', 1)}")
            if isinstance(num_frames_cfg, tuple):
                logging.info(f"  Num frames: {num_frames_cfg[0]}-{num_frames_cfg[1]} (variable)")
            else:
                logging.info(f"  Num frames: {num_frames_cfg}")
        else:
            logging.info(f"  History frames: {config.dataset.get('history_frames', None)}")
            logging.info(f"  Future frames: {config.dataset.get('future_frames', None)}")
            logging.info(f"  Frame stride: {config.dataset.get('frame_stride', 1)}")
        logging.info(f"  Filelist: {config.dataset.filelist_path}")
        if do_augment:
            logging.info(f"  Augmentation: ON (hflip={dataset.augment_hflip}, vflip={dataset.augment_vflip})")
        else:
            logging.info(f"  Augmentation: OFF")
    
    return dataloader


def train(config):
    torch.autograd.set_detect_anomaly(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    accelerator = accelerate.Accelerator(split_batches=False)
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f"Process {accelerator.process_index} using device: {device}")

    config.mixed_precision = accelerator.mixed_precision
    config = config.copy_and_resolve_references()

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        flow_utils.set_logger(
            log_level="info", fname=os.path.join(config.workdir, "output.log")
        )
        logging.info(config)
    else:
        flow_utils.set_logger(log_level="error")
    logging.info(f"Run on {accelerator.num_processes} devices")

    # ========= Data =========
    train_dataloader = build_dataloader(config, mode="train", accelerator=accelerator)
    val_dataloader = build_dataloader(config, mode="val", accelerator=accelerator)

    # Dataset info is already logged in build_dataloader

    # ========= FlowTok backbone & optimizer =========
    train_state = flow_utils.initialize_train_state(config, device)

    # ========= Lightweight adapters =========
    adapter_in_satellite = None
    adapter_in_radar = None
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
        )
    if getattr(config, "adapter_out", None) and config.adapter_out.get("enabled", False):
        adapter_out = AdapterOut(
            in_channels=int(getattr(config, "radar_out_channels", 3)),
            out_channels=1,
            mid_channels=int(config.adapter_out.get("mid_channels", 16)),
            num_blocks=int(config.adapter_out.get("num_blocks", 2)),
        )
    if getattr(config, "adapter_in_radar", None) and config.adapter_in_radar.get("enabled", False):
        adapter_in_radar = AdapterIn(
            in_channels=int(config.adapter_in_radar.get("in_channels", 1)),
            out_channels=int(getattr(config, "radar_in_channels", 3)),
            mid_channels=int(config.adapter_in_radar.get("mid_channels", 16)),
            num_blocks=int(config.adapter_in_radar.get("num_blocks", 2)),
        )

    # 将 adapter 参数加入优化器，与主干一起训练
    if adapter_in_satellite is not None:
        train_state.optimizer.add_param_group(
            {"params": list(adapter_in_satellite.parameters())}
        )
    if adapter_in_radar is not None:
        train_state.optimizer.add_param_group({"params": list(adapter_in_radar.parameters())})
    if adapter_out is not None:
        train_state.optimizer.add_param_group({"params": list(adapter_out.parameters())})
    # NOTE:
    # LambdaLR captures param-group count at creation time (inside base_lrs).
    # Since adapters are added after initialize_train_state(), we must rebuild
    # scheduler so its internal lr list length matches optimizer.param_groups.
    train_state.lr_scheduler = flow_utils.get_lr_scheduler(
        train_state.optimizer, **config.lr_scheduler
    )

    prepare_items = [
        train_state.nnet,
        train_state.nnet_ema,
        train_state.optimizer,
        train_dataloader,
        val_dataloader,
    ]
    if adapter_in_satellite is not None:
        prepare_items.append(adapter_in_satellite)
    if adapter_in_radar is not None:
        prepare_items.append(adapter_in_radar)
    if adapter_out is not None:
        prepare_items.append(adapter_out)
    prepared = accelerator.prepare(*prepare_items)
    nnet, nnet_ema, optimizer, train_dataloader, val_dataloader = (
        prepared[0],
        prepared[1],
        prepared[2],
        prepared[3],
        prepared[4],
    )
    idx_prepared = 5
    if adapter_in_satellite is not None:
        adapter_in_satellite = prepared[idx_prepared]
        idx_prepared += 1
    if adapter_in_radar is not None:
        adapter_in_radar = prepared[idx_prepared]
        idx_prepared += 1
    if adapter_out is not None:
        adapter_out = prepared[idx_prepared]

    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)

    # 如使用 adapter，尝试从同一步 ckpt 目录额外恢复其权重（若不存在则忽略）
    if train_state.step > 0:
        ckpt_path = os.path.join(config.ckpt_root, f"{train_state.step}.ckpt")
        if adapter_in_satellite is not None:
            adapter_in_satellite_path = os.path.join(
                ckpt_path, "adapter_in_satellite.pth"
            )
            legacy_adapter_in_path = os.path.join(ckpt_path, "adapter_in.pth")
            if os.path.isfile(adapter_in_satellite_path):
                accelerator.unwrap_model(adapter_in_satellite).load_state_dict(
                    torch.load(adapter_in_satellite_path, map_location="cpu")
                )
                logging.info(f"Loaded adapter_in_satellite from {adapter_in_satellite_path}")
            elif os.path.isfile(legacy_adapter_in_path):
                accelerator.unwrap_model(adapter_in_satellite).load_state_dict(
                    torch.load(legacy_adapter_in_path, map_location="cpu")
                )
                logging.info(f"Loaded legacy adapter_in from {legacy_adapter_in_path}")
        if adapter_in_radar is not None:
            adapter_in_radar_path = os.path.join(ckpt_path, "adapter_in_radar.pth")
            if os.path.isfile(adapter_in_radar_path):
                accelerator.unwrap_model(adapter_in_radar).load_state_dict(
                    torch.load(adapter_in_radar_path, map_location="cpu")
                )
                logging.info(f"Loaded adapter_in_radar from {adapter_in_radar_path}")
        if adapter_out is not None:
            adapter_out_path = os.path.join(ckpt_path, "adapter_out.pth")
            if os.path.isfile(adapter_out_path):
                accelerator.unwrap_model(adapter_out).load_state_dict(
                    torch.load(adapter_out_path, map_location="cpu")
                )
                logging.info(f"Loaded adapter_out from {adapter_out_path}")

    # ========= Pretrained FlowTiTok autoencoders =========
    # I2I/V2V pipeline: sat image -> sat tokenizer (77 tokens/frame); DiT -> radar tokens -> radar decoder (1ch).
    # 这里根据 config 中是否显式指定 in/out_channels 来决定 AE 的通道数：
    #   - 旧配置（Sat2Radar-Video-XL.py）默认 11ch sat / 1ch radar
    #   - 新配置（Sat2Radar-FlowTiTok-XL.py）通过 Dataset 仅使用 0/2/6 三个 IR 通道作为 sat 输入
    def _ae_config(base_config, role: str):
        vq = dict(base_config.vq_model)
        if role == "sat":
            in_ch = getattr(base_config, "sat_in_channels", None)
            out_ch = getattr(base_config, "sat_out_channels", None)
            if in_ch is None or out_ch is None:
                # 旧版：11 通道卫星
                in_ch, out_ch = 11, 11
        else:
            in_ch = getattr(base_config, "radar_in_channels", None)
            out_ch = getattr(base_config, "radar_out_channels", None)
            if in_ch is None or out_ch is None:
                # 旧版：1 通道雷达
                in_ch, out_ch = 1, 1
        vq["in_channels"] = in_ch
        vq["out_channels"] = out_ch
        cfg = ConfigDict(dict(base_config))
        cfg.vq_model = ConfigDict(vq)

        # 为 AE 单独设置预训练时的图像尺寸（例如 512x512），与 FlowTiTok_512 checkpoint 对齐
        ae_img_size = getattr(base_config, "ae_image_size", None)
        if ae_img_size is not None:
            ds = dict(getattr(base_config, "dataset", {}))
            ds["crop_size"] = ae_img_size
            cfg.dataset = ConfigDict(ds)
            # 也记录到 config 上，encode 时可读取
            cfg.ae_image_size = ae_img_size
        return cfg

    sat_ae_config = _ae_config(config, role="sat")
    radar_ae_config = _ae_config(config, role="radar")

    sat_autoencoder = FlowTiTok(sat_ae_config)
    sat_autoencoder.load_state_dict(
        torch.load(config.sat_tokenizer_checkpoint, map_location="cpu")
    )
    sat_autoencoder.eval()
    sat_autoencoder.requires_grad_(False)
    sat_autoencoder.to(device)

    radar_autoencoder = FlowTiTok(radar_ae_config)
    radar_autoencoder.load_state_dict(
        torch.load(config.radar_tokenizer_checkpoint, map_location="cpu")
    )
    radar_autoencoder.eval()
    radar_autoencoder.requires_grad_(False)
    radar_autoencoder.to(device)

    # ========= Text guidance encoder for FlowTiTok decoder（基于文件名的弱描述）=========
    clip_model_name = "ViT-L-14-336"
    local_clip_ckpt = os.environ.get("OPENCLIP_LOCAL_CKPT", None)
    try:
        if local_clip_ckpt is not None and os.path.isfile(local_clip_ckpt):
            logging.info(
                f"Initializing open_clip '{clip_model_name}' from local checkpoint: {local_clip_ckpt}"
            )
            # 不通过 HuggingFace hub，直接本地 load_state_dict，避免在计算节点访问外网。
            clip_encoder, _, _ = open_clip.create_model_and_transforms(
                clip_model_name, pretrained=None
            )
            state_dict = torch.load(local_clip_ckpt, map_location="cpu")
            missing_keys, unexpected_keys = clip_encoder.load_state_dict(
                state_dict, strict=False
            )
            if missing_keys or unexpected_keys:
                logging.info(
                    f"Loaded open_clip from local ckpt with missing_keys={len(missing_keys)}, "
                    f"unexpected_keys={len(unexpected_keys)}"
                )
        else:
            # 回退到默认行为：通过 tag='openai' 走 hub（本地/在线均可）
            logging.info(
                f"OPENCLIP_LOCAL_CKPT not set or file not found, "
                f"fallback to pretrained tag 'openai' for '{clip_model_name}'."
            )
            clip_encoder, _, _ = open_clip.create_model_and_transforms(
                clip_model_name, pretrained="openai"
            )

        # 只保留文本分支
        del clip_encoder.visual
        clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
        clip_encoder.transformer.batch_first = False
        clip_encoder.eval()
        clip_encoder.requires_grad_(False)
        clip_encoder.to(device)
    except Exception as e:
        clip_encoder = None
        clip_tokenizer = None
        logging.warning(
            f"open_clip not available or failed to initialize, "
            f"FlowTiTok decoder will run without text guidance. Error: {e}"
        )

    flow_matching_model = FlowMatching(
        noising_type=config.nnet.model_args.noising_type,
        noising_scale=config.nnet.model_args.noising_scale,
    )

    num_latent_tokens = config.vq_model.num_latent_tokens

    # Whether to backprop through frozen AE encoder to train AdapterIn.
    train_adapter_in_with_encoder_grad = bool(
        getattr(config, "train_adapter_in_with_encoder_grad", True)
    )
    debug_cfg = getattr(config, "debug", None)
    debug_enabled = bool(debug_cfg.get("enabled", False)) if debug_cfg is not None else False
    debug_max_steps = int(debug_cfg.get("max_steps", 5)) if debug_cfg is not None else 5
    debug_log_every = max(
        int(debug_cfg.get("log_every", 1)) if debug_cfg is not None else 1, 1
    )
    debug_grad_eps_on = (
        float(debug_cfg.get("grad_eps_on", 1e-12)) if debug_cfg is not None else 1e-12
    )
    debug_grad_eps_off = (
        float(debug_cfg.get("grad_eps_off", 1e-14)) if debug_cfg is not None else 1e-14
    )

    def _first_trainable_param(module):
        if module is None:
            return None
        for p in module.parameters():
            if p.requires_grad:
                return p
        return None

    def _module_grad_norm(module):
        if module is None:
            return 0.0
        grad_sq = 0.0
        has_grad = False
        for p in module.parameters():
            if p.grad is not None:
                g = p.grad.detach().float()
                grad_sq += float((g * g).sum().item())
                has_grad = True
        return float(grad_sq**0.5) if has_grad else 0.0

    def build_condition_tokens_from_sat_video(sat_video, require_grad=False):
        """
        Build condition tokens for FlowMatching.

        Default behavior:
          cond = sat_tokens (existing pipeline)

        Optional (config.cond_use_sat_lightning_tokens=True):
          - sat IR(3ch) -> tokenizer
          - lightning(1ch) repeat to 3ch -> tokenizer
          - fuse two token streams and feed to FlowMatching.
        """
        use_sat_lgt_tokens = getattr(config, "cond_use_sat_lightning_tokens", False)
        if not use_sat_lgt_tokens:
            return encode_video_with_autoencoder(
                sat_autoencoder,
                sat_video,
                config.vq_model.scale_factor,
                adapter_in=adapter_in_satellite,
                require_grad_through_encoder=require_grad,
            )

        sat_ir_video = sat_video[:, :, :3, :, :]
        # Dataset output layout: [selected IR bands..., lightning].
        # Lightning is the last channel after preprocessing (not second-to-last).
        lgt_slice = sat_video[:, :, -1:, :, :]
        lgt_video = lgt_slice.repeat(1, 1, 3, 1, 1)

        # For dual-token mode we intentionally bypass satellite AdapterIn and
        # use tokenizer-compatible 3ch inputs directly for both branches.
        sat_ir_tokens = encode_video_with_autoencoder(
            sat_autoencoder,
            sat_ir_video,
            config.vq_model.scale_factor,
            adapter_in=None,
            require_grad_through_encoder=require_grad,
        )
        lgt_tokens = encode_video_with_autoencoder(
            sat_autoencoder,
            lgt_video,
            config.vq_model.scale_factor,
            adapter_in=None,
            require_grad_through_encoder=require_grad,
        )

        fusion = getattr(config, "cond_token_fusion", "mean")
        if fusion == "sum":
            return sat_ir_tokens + lgt_tokens
        # default: mean
        return 0.5 * (sat_ir_tokens + lgt_tokens)

    @torch.no_grad()
    def _val_forward_batch(batch):
        """Same loss as training forward (flow + optional adapter_out recon), no backward."""
        metrics = {}
        sat_video = batch["sat_video"].to(
            accelerator.device,
            memory_format=torch.contiguous_format,
            non_blocking=True,
        )
        radar_video = batch["radar_video"].to(
            accelerator.device,
            memory_format=torch.contiguous_format,
            non_blocking=True,
        )
        sat_tokens = build_condition_tokens_from_sat_video(
            sat_video, require_grad=False
        )
        radar_tokens = encode_video_with_autoencoder(
            radar_autoencoder,
            radar_video,
            config.vq_model.scale_factor,
            adapter_in=adapter_in_radar,
            require_grad_through_encoder=False,
        )
        valid_mask = batch.get("valid_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(
                accelerator.device,
                non_blocking=True,
            )
            token_mask = valid_mask.repeat_interleave(num_latent_tokens, dim=1)
        else:
            token_mask = None

        loss, loss_dict = flow_matching_model(
            x=radar_tokens,
            nnet=nnet,
            cond=sat_tokens,
            all_config=config,
            batch_img_clip=None,
            valid_mask=token_mask,
        )
        total_loss = loss

        losses_cfg = getattr(config, "losses", None)
        adapter_out_recon_weight = (
            float(losses_cfg.get("adapter_out_recon_weight", 0.0))
            if losses_cfg is not None
            else 0.0
        )
        adapter_out_recon_loss = loss.new_zeros([])
        if adapter_out is not None and adapter_out_recon_weight > 0:
            Bv, Tv, _, H_gt, W_gt = radar_video.shape
            C_tok = radar_tokens.shape[-1]
            assert radar_tokens.shape[1] == Tv * num_latent_tokens
            radar_tok = radar_tokens.view(Bv, Tv, num_latent_tokens, C_tok)
            radar_tok = radar_tok.reshape(Bv * Tv, num_latent_tokens, C_tok)
            radar_tok = radar_tok.permute(0, 2, 1).unsqueeze(2)
            radar_decoded = radar_autoencoder.decode_tokens(
                radar_tok / config.vq_model.scale_factor,
                text_guidance=None,
            )
            radar_pred_aux = adapter_out(radar_decoded, out_size=(H_gt, W_gt))
            radar_pred_aux = torch.clamp(radar_pred_aux, 0.0, 1.0)
            radar_gt_aux = radar_video.reshape(Bv * Tv, 1, H_gt, W_gt)
            if valid_mask is not None:
                per_frame_mse = (radar_pred_aux - radar_gt_aux).pow(2).mean(
                    dim=[1, 2, 3]
                )
                vm_flat = valid_mask.reshape(Bv * Tv).float()
                adapter_out_recon_loss = (
                    (per_frame_mse * vm_flat).sum() / vm_flat.sum().clamp_min(1.0)
                )
            else:
                adapter_out_recon_loss = F.mse_loss(radar_pred_aux, radar_gt_aux)
            total_loss = total_loss + adapter_out_recon_weight * adapter_out_recon_loss
            metrics["adapter_out_recon_loss"] = accelerator.gather(
                adapter_out_recon_loss.detach()
            ).mean()

        metrics["total_loss"] = accelerator.gather(total_loss.detach()).mean()
        metrics["loss"] = accelerator.gather(loss.detach()).mean()
        for key, val in loss_dict.items():
            metrics[key] = accelerator.gather(val.detach()).mean()
        return metrics

    def run_validation_logging():
        """Average val losses over part or all of val loader; log to the same output.log as train."""
        val_max_batches = getattr(config.train, "val_max_batches", 64)
        if val_max_batches is None:
            val_max_batches = 64
        val_max_batches = int(val_max_batches)
        # Negative or zero: use full val set (all batches).
        if val_max_batches <= 0:
            val_max_batches = 10**12
        nnet.eval()
        if adapter_in_satellite is not None:
            adapter_in_satellite.eval()
        if adapter_in_radar is not None:
            adapter_in_radar.eval()
        if adapter_out is not None:
            adapter_out.eval()

        sums: dict = {}
        nb = 0
        for bi, vbatch in enumerate(val_dataloader):
            if bi >= val_max_batches:
                break
            m = _val_forward_batch(vbatch)
            for k, v in m.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach().item())
            nb += 1

        nnet.train()
        if adapter_in_satellite is not None:
            adapter_in_satellite.train()
        if adapter_in_radar is not None:
            adapter_in_radar.train()
        if adapter_out is not None:
            adapter_out.train()

        if nb == 0 or not accelerator.is_main_process:
            return
        avgs = {k: sums[k] / nb for k in sums}
        log_d = {"step": train_state.step}
        for k, v in avgs.items():
            log_d["val_" + k] = v
        logging.info(flow_utils.dct2str(log_d))

    def train_step(batch):
        metrics = dict()
        optimizer.zero_grad()

        sat_video = batch["sat_video"].to(
            accelerator.device,
            memory_format=torch.contiguous_format,
            non_blocking=True,
        )  # [B, T_max, C_sat, H, W] (may contain padding)
        radar_video = batch["radar_video"].to(
            accelerator.device,
            memory_format=torch.contiguous_format,
            non_blocking=True,
        )  # [B, T_max, 1, H, W]

        # tokens: [B, T_max*L, C_tok]
        sat_tokens = build_condition_tokens_from_sat_video(
            sat_video, require_grad=train_adapter_in_with_encoder_grad
        )
        radar_tokens = encode_video_with_autoencoder(
            radar_autoencoder,
            radar_video,
            config.vq_model.scale_factor,
            adapter_in=adapter_in_radar,
            require_grad_through_encoder=(
                train_adapter_in_with_encoder_grad and (adapter_in_radar is not None)
            ),
        )

        # Variable-length v2v: mask padded positions in loss (valid_mask [B, T_max] -> token-level)
        valid_mask = batch.get("valid_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(
                accelerator.device,
                non_blocking=True,
            )
            token_mask = valid_mask.repeat_interleave(num_latent_tokens, dim=1)  # [B, T_max*L]
        else:
            token_mask = None

        # x_start = radar tokens, cond = sat tokens
        loss, loss_dict = flow_matching_model(
            x=radar_tokens,
            nnet=nnet,
            cond=sat_tokens,
            all_config=config,
            batch_img_clip=None,
            valid_mask=token_mask,
        )
        total_loss = loss

        # Auxiliary loss for AdapterOut so it receives direct training signal.
        losses_cfg = getattr(config, "losses", None)
        adapter_out_recon_weight = (
            float(losses_cfg.get("adapter_out_recon_weight", 0.0))
            if losses_cfg is not None
            else 0.0
        )
        adapter_out_recon_loss = loss.new_zeros([])
        if adapter_out is not None and adapter_out_recon_weight > 0:
            Bv, Tv, _, H_gt, W_gt = radar_video.shape
            C_tok = radar_tokens.shape[-1]
            assert (
                radar_tokens.shape[1] == Tv * num_latent_tokens
            ), "radar token length must equal T*num_latent_tokens"
            radar_tok = radar_tokens.view(Bv, Tv, num_latent_tokens, C_tok)
            radar_tok = radar_tok.reshape(Bv * Tv, num_latent_tokens, C_tok)
            radar_tok = radar_tok.permute(0, 2, 1).unsqueeze(2)  # [B*T, C_tok, 1, L]
            # Keep AE frozen; only train adapter_out from this branch.
            with torch.no_grad():
                radar_decoded = radar_autoencoder.decode_tokens(
                    radar_tok / config.vq_model.scale_factor,
                    text_guidance=None,
                )  # [B*T, 3, H_dec, W_dec]
            radar_pred_aux = adapter_out(radar_decoded, out_size=(H_gt, W_gt))
            radar_pred_aux = torch.clamp(radar_pred_aux, 0.0, 1.0)
            radar_gt_aux = radar_video.reshape(Bv * Tv, 1, H_gt, W_gt)
            if valid_mask is not None:
                per_frame_mse = (radar_pred_aux - radar_gt_aux).pow(2).mean(dim=[1, 2, 3])
                vm_flat = valid_mask.reshape(Bv * Tv).float()
                adapter_out_recon_loss = (
                    (per_frame_mse * vm_flat).sum() / vm_flat.sum().clamp_min(1.0)
                )
            else:
                adapter_out_recon_loss = F.mse_loss(radar_pred_aux, radar_gt_aux)
            total_loss = total_loss + adapter_out_recon_weight * adapter_out_recon_loss
            metrics["adapter_out_recon_loss"] = accelerator.gather(
                adapter_out_recon_loss.detach()
            ).mean()

        if (
            debug_enabled
            and accelerator.is_main_process
            and train_state.step < debug_max_steps
            and train_state.step % debug_log_every == 0
        ):
            flow_loss_scalar = loss.mean()
            recon_loss_scalar = (adapter_out_recon_weight * adapter_out_recon_loss).mean()
            total_loss_scalar = total_loss.mean()
            probe_params = {
                "nnet": _first_trainable_param(nnet),
                "adapter_in_satellite": _first_trainable_param(adapter_in_satellite),
                "adapter_in_radar": _first_trainable_param(adapter_in_radar),
                "adapter_out": _first_trainable_param(adapter_out),
            }
            grad_probe = {}
            for name, p in probe_params.items():
                if p is None:
                    grad_probe[f"{name}_from_flow"] = 0.0
                    grad_probe[f"{name}_from_recon"] = 0.0
                    grad_probe[f"{name}_from_total"] = 0.0
                    continue
                g_flow = torch.autograd.grad(
                    flow_loss_scalar, p, retain_graph=True, allow_unused=True
                )[0]
                g_recon = torch.autograd.grad(
                    recon_loss_scalar, p, retain_graph=True, allow_unused=True
                )[0]
                g_total = torch.autograd.grad(
                    total_loss_scalar, p, retain_graph=True, allow_unused=True
                )[0]
                grad_probe[f"{name}_from_flow"] = (
                    float(g_flow.detach().norm().item()) if g_flow is not None else 0.0
                )
                grad_probe[f"{name}_from_recon"] = (
                    float(g_recon.detach().norm().item()) if g_recon is not None else 0.0
                )
                grad_probe[f"{name}_from_total"] = (
                    float(g_total.detach().norm().item()) if g_total is not None else 0.0
                )
            recon_branch_enabled = adapter_out is not None and adapter_out_recon_weight > 0
            debug_failures = []

            def _expect_gt(key, eps, note):
                if grad_probe.get(key, 0.0) <= eps:
                    debug_failures.append(f"{key}<={eps:.1e} ({note})")

            def _expect_le(key, eps, note):
                if grad_probe.get(key, 0.0) > eps:
                    debug_failures.append(f"{key}>{eps:.1e} ({note})")

            # Expected gradient routing:
            # flow -> nnet/adapter_in_* ; recon -> adapter_out only ; total -> union.
            _expect_gt("nnet_from_flow", debug_grad_eps_on, "nnet should get flow grad")
            if adapter_in_satellite is not None:
                _expect_gt(
                    "adapter_in_satellite_from_flow",
                    debug_grad_eps_on,
                    "adapter_in_satellite should get flow grad",
                )
            if adapter_in_radar is not None:
                _expect_gt(
                    "adapter_in_radar_from_flow",
                    debug_grad_eps_on,
                    "adapter_in_radar should get flow grad",
                )
            if adapter_out is not None:
                _expect_le(
                    "adapter_out_from_flow",
                    debug_grad_eps_off,
                    "adapter_out should not get flow grad",
                )
                if recon_branch_enabled:
                    _expect_gt(
                        "adapter_out_from_recon",
                        debug_grad_eps_on,
                        "adapter_out should get recon grad",
                    )
                else:
                    _expect_le(
                        "adapter_out_from_recon",
                        debug_grad_eps_off,
                        "adapter_out recon is disabled",
                    )
            _expect_le(
                "nnet_from_recon",
                debug_grad_eps_off,
                "nnet should not get recon grad",
            )
            if adapter_in_satellite is not None:
                _expect_le(
                    "adapter_in_satellite_from_recon",
                    debug_grad_eps_off,
                    "adapter_in_satellite should not get recon grad",
                )
            if adapter_in_radar is not None:
                _expect_le(
                    "adapter_in_radar_from_recon",
                    debug_grad_eps_off,
                    "adapter_in_radar should not get recon grad",
                )
            debug_status = "PASS" if len(debug_failures) == 0 else "FAIL"
            logging.info(
                "DEBUG_LOSS_PATH_%s step=%d flow=%.6e recon_w=%.6e total=%.6e probes=%s failures=%s",
                debug_status,
                int(train_state.step),
                float(flow_loss_scalar.detach().item()),
                float(recon_loss_scalar.detach().item()),
                float(total_loss_scalar.detach().item()),
                grad_probe,
                debug_failures,
            )

        metrics["total_loss"] = accelerator.gather(total_loss.detach()).mean()
        metrics["loss"] = accelerator.gather(loss.detach()).mean()
        for key, val in loss_dict.items():
            metrics[key] = accelerator.gather(val.detach()).mean()

        accelerator.backward(total_loss.mean())
        if (
            debug_enabled
            and accelerator.is_main_process
            and train_state.step < debug_max_steps
            and train_state.step % debug_log_every == 0
        ):
            logging.info(
                "DEBUG_GRAD_NORM step=%d nnet=%.6e adapter_in_satellite=%.6e "
                "adapter_in_radar=%.6e adapter_out=%.6e",
                int(train_state.step),
                _module_grad_norm(nnet),
                _module_grad_norm(adapter_in_satellite),
                _module_grad_norm(adapter_in_radar),
                _module_grad_norm(adapter_out),
            )

        optimizer.step()
        lr_scheduler.step()

        train_state.ema_update(config.get("ema_rate", 0.9999))

        train_state.step += 1
        return {
            "lr": train_state.optimizer.param_groups[0]["lr"],
            "total_loss": metrics["total_loss"],
            **{k: v for k, v in metrics.items() if k != "total_loss"},
        }

    def ode_fm_solver_sample(nnet_ema_local, batch, use_adapter_out: bool = True):
        """
        简单的采样示例：给一个 batch 的 sat_video，生成对应的 radar_video。
        """
        with torch.no_grad():
            sat_video = batch["sat_video"].to(
                accelerator.device,
                memory_format=torch.contiguous_format,
                non_blocking=True,
            )
            B = sat_video.shape[0]
            sat_tokens = build_condition_tokens_from_sat_video(
                sat_video, require_grad=False
            )  # [B, L, C]

            # 采样阶段与训练阶段保持一致：
            # - 默认：使用 textVAE，对 sat_tokens 做编码得到 x0 作为 flow 起点；
            # - 可选（config.use_text_vae_encoder == False）：直接使用 sat_tokens 作为起点，实现
            #   “从卫星 tokens → 雷达 tokens” 的显式 flow，而不经过 textVAE。
            use_text_vae_encoder = getattr(config, "use_text_vae_encoder", True)
            if use_text_vae_encoder:
                x0, _, _ = nnet_ema_local(sat_tokens, text_encoder=True)
            else:
                x0 = sat_tokens

            if config.nnet.model_args.noising_type != "none":
                x0 = x0 + torch.randn_like(x0) * config.sample.noise_scale

            guidance_scale = config.sample.scale
            has_null_indicator = True

            ode_solver = ODEEulerFlowMatchingSolver(
                nnet_ema_local,
                step_size_type="step_in_dsigma",
                guidance_scale=guidance_scale,
            )
            z, _ = ode_solver.sample(
                x_T=x0,
                batch_size=B,
                sample_steps=config.sample.sample_steps,
                unconditional_guidance_scale=guidance_scale,
                has_null_indicator=has_null_indicator,
            )
            # z: [B, L, C_tok] -> [B*T, C_tok, 1, L_frame] 再 decode
            # 这里假设 T 与训练时一致，可以由 L 和 num_latent_tokens 反推
            L = z.shape[1]
            num_latent_tokens = config.vq_model.num_latent_tokens
            assert (
                L % num_latent_tokens == 0
            ), "Sequence length must be multiple of num_latent_tokens"
            T = L // num_latent_tokens

            z = z.view(B, T, num_latent_tokens, z.shape[2])  # [B, T, Lf, C]
            z = z.view(B * T, num_latent_tokens, z.shape[3])  # [B*T, Lf, C]
            z = z.permute(0, 2, 1).unsqueeze(2)  # [B*T, C, 1, Lf]

            # 构造基于文件名的弱文本描述，作为 FlowTiTok decoder 的 text_guidance
            text_guidance = None
            if clip_encoder is not None and clip_tokenizer is not None:
                radar_paths = batch.get("radar_paths")
                texts = []
                for i in range(B):
                    for t in range(T):
                        fname = "unknown"
                        if radar_paths and i < len(radar_paths):
                            paths_i = radar_paths[i]
                            if isinstance(paths_i, (list, tuple)) and t < len(paths_i):
                                fname = os.path.basename(str(paths_i[t]))
                        desc = f"A radar reflectivity image from {fname}."
                        texts.append(desc)

                if len(texts) == B * T:
                    try:
                        text_tokens = clip_tokenizer(texts).to(device)
                        cast_dtype = clip_encoder.transformer.get_cast_dtype()
                        text_tokens = clip_encoder.token_embedding(text_tokens).to(
                            cast_dtype
                        )  # [B*T, n_ctx, d_model]
                        text_tokens = (
                            text_tokens
                            + clip_encoder.positional_embedding.to(cast_dtype)
                        )
                        text_tokens = text_tokens.permute(1, 0, 2)  # NLD -> LND
                        text_tokens = clip_encoder.transformer(
                            text_tokens, attn_mask=clip_encoder.attn_mask
                        )
                        text_tokens = text_tokens.permute(1, 0, 2)  # LND -> NLD
                        text_tokens = clip_encoder.ln_final(
                            text_tokens
                        )  # [B*T, n_ctx, d_model]
                        text_guidance = text_tokens
                    except Exception as e:
                        logging.warning(f"Failed to build CLIP text guidance: {e}")
                        text_guidance = None

            radar_video_pred = radar_autoencoder.decode_tokens(
                z / config.vq_model.scale_factor, text_guidance=text_guidance
            )  # [B*T, C_out, H, W]，此处 C_out=3
            if adapter_out is not None and use_adapter_out:
                # 适配到数据集雷达分辨率（若 batch 中可用）
                radar_ref = batch.get("radar_video")
                if radar_ref is not None:
                    H_gt = radar_ref.shape[-2]
                    W_gt = radar_ref.shape[-1]
                    radar_video_pred = adapter_out(radar_video_pred, out_size=(H_gt, W_gt))
                else:
                    radar_video_pred = adapter_out(
                        radar_video_pred,
                        out_size=(radar_video_pred.shape[-2], radar_video_pred.shape[-1]),
                    )
                radar_video_pred = torch.clamp(radar_video_pred, 0.0, 1.0)
            else:
                # 由于雷达物理上是单通道，这里只取第一个通道作为雷达图
                radar_video_pred = radar_video_pred[:, 0:1, ...]  # [B*T, 1, H, W]
            radar_video_pred = radar_video_pred.view(
                B, T, 1, radar_video_pred.shape[-2], radar_video_pred.shape[-1]
            )
            return radar_video_pred

    logging.info(
        f"Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}"
    )

    while train_state.step < config.train.n_steps:
        for batch in train_dataloader:
            nnet.train()
            metrics = train_step(batch)

            nnet.eval()
            if (
                accelerator.is_main_process
                and train_state.step % config.train.log_interval == 0
            ):
                logging.info(flow_utils.dct2str(dict(step=train_state.step, **metrics)))
                logging.info(config.workdir)

            # 保存一些可视化样例
            if train_state.step % config.train.eval_interval == 0:
                torch.cuda.empty_cache()
                run_validation_logging()
                logging.info(
                    "Save a grid of [sat_IR | sat_lightning | radar_gt | radar_pred] "
                    "(first frame only) with AE-style colormaps..."
                )
                with torch.no_grad():
                    # 预测雷达视频
                    radar_video_pred = ode_fm_solver_sample(nnet_ema, batch)  # [B, T, 1, H, W]
                    radar_video_pred_no_adapterout = None
                    if adapter_out is not None:
                        radar_video_pred_no_adapterout = ode_fm_solver_sample(
                            nnet_ema, batch, use_adapter_out=False,
                        )  # [B, T, 1, H, W]
                    B, T_eff, _, H, W = radar_video_pred.shape

                    # 从 batch 中取出对应的 sat / radar GT（只看第 1 帧）
                    sat_video = batch["sat_video"].to(
                        accelerator.device,
                        memory_format=torch.contiguous_format,
                        non_blocking=True,
                    )  # [B, T, C_sat, H, W]
                    radar_video_gt = batch["radar_video"].to(
                        accelerator.device,
                        memory_format=torch.contiguous_format,
                        non_blocking=True,
                    )  # [B, T, 1, H_gt, W_gt]

                    # 如果 AE 解码出来的雷达分辨率与原始 GT 不一致（例如 AE 在 512x512 上预训练，
                    # 而数据集裁剪为 128x128），则在可视化前把预测 resize 回 GT 的分辨率，
                    # 以便后续拼接 [sat_IR | sat_lightning | radar_gt | radar_pred] 时尺寸一致。
                    _, _, _, H_gt, W_gt = radar_video_gt.shape
                    if radar_video_pred.shape[-2] != H_gt or radar_video_pred.shape[-1] != W_gt:
                        radar_video_pred_flat = radar_video_pred.view(B * T_eff, 1, H, W)
                        radar_video_pred_flat = F.interpolate(
                            radar_video_pred_flat,
                            size=(H_gt, W_gt),
                            mode="bilinear",
                            align_corners=False,
                        )
                        radar_video_pred = radar_video_pred_flat.view(
                            B, T_eff, 1, H_gt, W_gt
                        )
                    if radar_video_pred_no_adapterout is not None:
                        T_eff_no_adapter = radar_video_pred_no_adapterout.shape[1]
                        H_no_adapter = radar_video_pred_no_adapterout.shape[-2]
                        W_no_adapter = radar_video_pred_no_adapterout.shape[-1]
                        if (
                            H_no_adapter != H_gt
                            or W_no_adapter != W_gt
                        ):
                            radar_video_pred_no_adapterout_flat = radar_video_pred_no_adapterout.view(
                                B * T_eff_no_adapter, 1, H_no_adapter, W_no_adapter
                            )
                            radar_video_pred_no_adapterout_flat = F.interpolate(
                                radar_video_pred_no_adapterout_flat,
                                size=(H_gt, W_gt),
                                mode="bilinear",
                                align_corners=False,
                            )
                            radar_video_pred_no_adapterout = radar_video_pred_no_adapterout_flat.view(
                                B, T_eff_no_adapter, 1, H_gt, W_gt
                            )

                    # 只可视化每个样本的第 1 帧，使用物理范围和 colormap：
                    # sat: IR ch0 使用 [200,320] + 'cmc.batlow_r'
                    #       lightning 通道使用 [0.1,50] + 'Reds'
                    # radar: [0,60] dBZ + 'HomeyerRainbow'
                    ir_min, ir_max = 200.0, 320.0
                    l_min, l_max = 0.1, 50.0
                    z_min, z_max = 0.0, 60.0
                    cmap_ir = _cmap_or_fallback("cmc.batlow_r", fallback="viridis")
                    cmap_lgt = _cmap_or_fallback("Reds", fallback="Reds")
                    cmap_rad = _cmap_or_fallback("HomeyerRainbow", fallback="viridis")
                    ir_band_indices = getattr(config.dataset, "ir_band_indices", None)
                    lgt_channel_idx = (
                        len(ir_band_indices)
                        if getattr(config.dataset, "use_lightning", False)
                        and ir_band_indices is not None
                        else None
                    )

                    max_samples = min(B, 8)
                    rows_np = []
                    for i in range(max_samples):
                        # 第 1 帧
                        sat_ir = sat_video[i, 0, 0] * (ir_max - ir_min) + ir_min  # [H_gt, W_gt]
                        radar_gt_2d = (
                            radar_video_gt[i, 0, 0] * (z_max - z_min) + z_min
                        )  # [H, W]
                        radar_pred_2d = (
                            radar_video_pred[i, 0, 0] * (z_max - z_min) + z_min
                        )  # [H, W]

                        sat_ir_np = sat_ir.detach().cpu().numpy()
                        radar_gt_np = radar_gt_2d.detach().cpu().numpy()
                        radar_pred_np = radar_pred_2d.detach().cpu().numpy()

                        # Lightning channel index follows dataset output:
                        # [selected IR bands] + [lightning (optional)].
                        if (
                            lgt_channel_idx is not None
                            and sat_video.shape[2] > lgt_channel_idx
                        ):
                            sat_lgt = sat_video[i, 0, lgt_channel_idx] * (l_max - l_min) + l_min
                            sat_lgt_np = sat_lgt.detach().cpu().numpy()
                        else:
                            sat_lgt_np = np.zeros_like(sat_ir_np)

                        sat_ir_rgb = _apply_cmap(sat_ir_np, cmap_ir, ir_min, ir_max)
                        sat_lgt_rgb = _apply_cmap(sat_lgt_np, cmap_lgt, l_min, l_max)
                        gt_rgb = _apply_cmap(radar_gt_np, cmap_rad, z_min, z_max)
                        pred_rgb = _apply_cmap(radar_pred_np, cmap_rad, z_min, z_max)

                        row = np.concatenate(
                            [sat_ir_rgb, sat_lgt_rgb, gt_rgb, pred_rgb], axis=1
                        )  # [H, 4W, 3]
                        rows_np.append(row)

                    if rows_np:
                        # 竖直堆叠多行，得到 [H*max_samples, 4W, 3]
                        stacked = np.concatenate(rows_np, axis=0)

                        if accelerator.is_main_process:
                            save_path = os.path.join(
                                config.sample_dir,
                                f"{train_state.step}_sat_lgt_gt_pred.png",
                            )
                            plt.imsave(save_path, stacked)

                    # v2v 训练时，额外保存一段随时间演化的 GIF 视频，
                    # 每一帧与上面的 PNG 类似，都是多样本堆叠 + 四联图：[sat_IR | sat_lightning | radar_gt | radar_pred]。
                    if accelerator.is_main_process and getattr(getattr(config, "dataset", {}), "v2v", False) and HAS_IMAGEIO:
                        video_frames = []
                        for t in range(T_eff):
                            rows_t = []
                            for i in range(max_samples):
                                # 物理量缩放与上面保持一致
                                sat_ir_t = sat_video[i, t, 0] * (ir_max - ir_min) + ir_min
                                radar_gt_t = (
                                    radar_video_gt[i, t, 0] * (z_max - z_min) + z_min
                                )
                                radar_pred_t = (
                                    radar_video_pred[i, t, 0] * (z_max - z_min) + z_min
                                )

                                sat_ir_np_t = sat_ir_t.detach().cpu().numpy()
                                radar_gt_np_t = radar_gt_t.detach().cpu().numpy()
                                radar_pred_np_t = radar_pred_t.detach().cpu().numpy()

                                if (
                                    lgt_channel_idx is not None
                                    and sat_video.shape[2] > lgt_channel_idx
                                ):
                                    sat_lgt_t = sat_video[i, t, lgt_channel_idx] * (l_max - l_min) + l_min
                                    sat_lgt_np_t = sat_lgt_t.detach().cpu().numpy()
                                else:
                                    sat_lgt_np_t = np.zeros_like(sat_ir_np_t)

                                sat_ir_rgb_t = _apply_cmap(
                                    sat_ir_np_t, cmap_ir, ir_min, ir_max
                                )
                                sat_lgt_rgb_t = _apply_cmap(
                                    sat_lgt_np_t, cmap_lgt, l_min, l_max
                                )
                                gt_rgb_t = _apply_cmap(
                                    radar_gt_np_t, cmap_rad, z_min, z_max
                                )
                                pred_rgb_t = _apply_cmap(
                                    radar_pred_np_t, cmap_rad, z_min, z_max
                                )

                                row_t = np.concatenate(
                                    [sat_ir_rgb_t, sat_lgt_rgb_t, gt_rgb_t, pred_rgb_t],
                                    axis=1,
                                )  # [H, 4W, 3]
                                rows_t.append(row_t)

                            if not rows_t:
                                continue

                            # 与 PNG 一致：按样本在竖直方向堆叠，形成一帧
                            frame = np.concatenate(rows_t, axis=0)  # [H*max_samples, 4W, 3]
                            # 转为 uint8，确保 GIF 兼容
                            frame_uint8 = (np.clip(frame, 0.0, 1.0) * 255).astype(
                                np.uint8
                            )
                            video_frames.append(frame_uint8)

                        if accelerator.is_main_process and video_frames:
                            video_path = os.path.join(
                                config.sample_dir,
                                f"{train_state.step}_v2v.gif",
                            )
                            try:
                                imageio.mimsave(video_path, video_frames, fps=4)
                            except Exception as e:
                                logging.warning(f"Failed to save v2v GIF: {e}")

                        # Additional comparison GIF: same visualization but bypass adapter_out.
                        if (
                            accelerator.is_main_process
                            and radar_video_pred_no_adapterout is not None
                        ):
                            video_frames_no_adapter = []
                            T_eff_no_adapter = radar_video_pred_no_adapterout.shape[1]
                            for t in range(T_eff_no_adapter):
                                rows_t = []
                                for i in range(max_samples):
                                    sat_ir_t = sat_video[i, t, 0] * (ir_max - ir_min) + ir_min
                                    radar_gt_t = (
                                        radar_video_gt[i, t, 0] * (z_max - z_min) + z_min
                                    )
                                    radar_pred_t = (
                                        radar_video_pred_no_adapterout[i, t, 0] * (z_max - z_min) + z_min
                                    )

                                    sat_ir_np_t = sat_ir_t.detach().cpu().numpy()
                                    radar_gt_np_t = radar_gt_t.detach().cpu().numpy()
                                    radar_pred_np_t = radar_pred_t.detach().cpu().numpy()

                                    if (
                                        lgt_channel_idx is not None
                                        and sat_video.shape[2] > lgt_channel_idx
                                    ):
                                        sat_lgt_t = sat_video[i, t, lgt_channel_idx] * (l_max - l_min) + l_min
                                        sat_lgt_np_t = sat_lgt_t.detach().cpu().numpy()
                                    else:
                                        sat_lgt_np_t = np.zeros_like(sat_ir_np_t)

                                    sat_ir_rgb_t = _apply_cmap(
                                        sat_ir_np_t, cmap_ir, ir_min, ir_max
                                    )
                                    sat_lgt_rgb_t = _apply_cmap(
                                        sat_lgt_np_t, cmap_lgt, l_min, l_max
                                    )
                                    gt_rgb_t = _apply_cmap(
                                        radar_gt_np_t, cmap_rad, z_min, z_max
                                    )
                                    pred_rgb_t = _apply_cmap(
                                        radar_pred_np_t, cmap_rad, z_min, z_max
                                    )

                                    row_t = np.concatenate(
                                        [sat_ir_rgb_t, sat_lgt_rgb_t, gt_rgb_t, pred_rgb_t],
                                        axis=1,
                                    )
                                    rows_t.append(row_t)

                                if not rows_t:
                                    continue
                                frame = np.concatenate(rows_t, axis=0)
                                frame_uint8 = (np.clip(frame, 0.0, 1.0) * 255).astype(
                                    np.uint8
                                )
                                video_frames_no_adapter.append(frame_uint8)

                            if video_frames_no_adapter:
                                video_path_no_adapter = os.path.join(
                                    config.sample_dir,
                                    f"{train_state.step}_v2v_no_adapterout.gif",
                                )
                                try:
                                    imageio.mimsave(
                                        video_path_no_adapter, video_frames_no_adapter, fps=4
                                    )
                                except Exception as e:
                                    logging.warning(
                                        f"Failed to save no-adapterout v2v GIF: {e}"
                                    )
                accelerator.wait_for_everyone()
                torch.cuda.empty_cache()

            # 保存 checkpoint
            if (
                train_state.step % config.train.save_interval == 0
                or train_state.step == config.train.n_steps
            ):
                torch.cuda.empty_cache()
                logging.info(f"Save checkpoint {train_state.step}...")
                if accelerator.is_main_process:
                    ckpt_save_path = os.path.join(config.ckpt_root, f"{train_state.step}.ckpt")
                    train_state.save(ckpt_save_path)
                    if adapter_in_satellite is not None:
                        torch.save(
                            accelerator.unwrap_model(adapter_in_satellite).state_dict(),
                            os.path.join(ckpt_save_path, "adapter_in_satellite.pth"),
                        )
                    if adapter_in_radar is not None:
                        torch.save(
                            accelerator.unwrap_model(adapter_in_radar).state_dict(),
                            os.path.join(ckpt_save_path, "adapter_in_radar.pth"),
                        )
                    if adapter_out is not None:
                        torch.save(
                            accelerator.unwrap_model(adapter_out).state_dict(),
                            os.path.join(ckpt_save_path, "adapter_out.pth"),
                        )
                accelerator.wait_for_everyone()
                torch.cuda.empty_cache()

            if train_state.step >= config.train.n_steps:
                break
        accelerator.wait_for_everyone()

    logging.info(f"Finish fitting, step={train_state.step}")
    accelerator.wait_for_everyone()


def main(_):
    config = FLAGS.config
    if FLAGS.filelist_path:
        config.dataset.filelist_path = FLAGS.filelist_path
        logging.info("Override filelist_path: %s", FLAGS.filelist_path)
    train(config)


if __name__ == "__main__":
    app.run(main)

