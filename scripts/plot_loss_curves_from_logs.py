import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Series:
    steps: List[int]
    values: List[float]


LOSS_KEYS_DEFAULT = ["total_loss"]

# Keys emitted by training (train_step) and validate (run_validation_logging -> val_*).
_KEYS_TO_PARSE = [
    "total_loss",
    "loss",
    "diff_loss",
    "contrastive_loss",
    "kld_loss",
    "val_total_loss",
    "val_loss",
    "val_diff_loss",
    "val_contrastive_loss",
    "val_kld_loss",
    "val_adapter_out_recon_loss",
]

_LINE_MARKERS = tuple(f"'{k}'" for k in _KEYS_TO_PARSE)


def _extract_step_and_losses(line: str) -> Optional[Tuple[int, Dict[str, float]]]:
    """
    Parse a log line containing a dict-like payload:
        {'step': '100', 'lr': '8e-06', 'total_loss': '1.30051', 'diff_loss': '...', ...}
        {'step': '2000', 'val_total_loss': '0.42', 'val_diff_loss': '...', ...}
    Returns (step, dict_of_available_losses) or None if not parseable.
    """
    if "{" not in line or "'step'" not in line:
        return None
    if not any(m in line for m in _LINE_MARKERS):
        return None

    m_step = re.search(r"'step'\s*:\s*'(\d+)'", line)
    if not m_step:
        return None
    step = int(m_step.group(1))

    out: Dict[str, float] = {}
    for k in _KEYS_TO_PARSE:
        mk = re.search(rf"'{re.escape(k)}'\s*:\s*'([0-9eE.+-]+)'", line)
        if mk:
            out[k] = float(mk.group(1))
    if not out:
        return None
    return step, out


def parse_log_file(log_path: Path, keys: Sequence[str]) -> Dict[str, Series]:
    """
    Parse a training log file and return step-aligned series for requested keys.
    If multiple entries exist for the same step, keeps the last seen value.
    """
    # step -> key -> value
    per_step: Dict[int, Dict[str, float]] = {}
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = _extract_step_and_losses(line)
            if parsed is None:
                continue
            step, losses = parsed
            # keep only requested keys to reduce clutter
            filtered = {k: losses[k] for k in keys if k in losses}
            if filtered:
                per_step.setdefault(step, {}).update(filtered)

    steps_sorted = sorted(per_step.keys())
    series: Dict[str, Series] = {}
    for k in keys:
        vals = [per_step[s].get(k, np.nan) for s in steps_sorted]
        # remove points where k was never present at that step
        steps2: List[int] = []
        vals2: List[float] = []
        for s, v in zip(steps_sorted, vals):
            if not np.isnan(v):
                steps2.append(s)
                vals2.append(float(v))
        series[k] = Series(steps=steps2, values=vals2)
    return series


def moving_average(y: Sequence[float], window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(y, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < window:
        return y
    kernel = np.ones(window, dtype=float) / float(window)
    # "same" keeps array length
    return np.convolve(y, kernel, mode="same")


def main():
    parser = argparse.ArgumentParser("Plot loss curves from training output.log")
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        help="Path to output.log (can be given multiple times).",
    )
    parser.add_argument(
        "--labels",
        default=None,
        type=str,
        help="Comma-separated labels matching --log order. If omitted, uses log filename.",
    )
    parser.add_argument(
        "--keys",
        default="total_loss",
        type=str,
        help=(
            "Comma-separated loss keys to plot. Supported: total_loss, loss, diff_loss, "
            "contrastive_loss, kld_loss, val_total_loss, val_loss, val_diff_loss, "
            "val_contrastive_loss, val_kld_loss, val_adapter_out_recon_loss"
        ),
    )
    parser.add_argument(
        "--smooth-window",
        default=0,
        type=int,
        help="Moving average window (steps) for smoothing. 0 disables.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help="Output image path, e.g. /scratch/.../loss_curve.png",
    )
    parser.add_argument(
        "--ylog",
        action="store_true",
        help="Plot y on log scale.",
    )
    args = parser.parse_args()

    logs: List[Path] = [Path(p) for p in args.log]
    if args.labels is None:
        labels = [p.parent.name + "/" + p.name for p in logs]
    else:
        labels = [s.strip() for s in args.labels.split(",")]
        if len(labels) != len(logs):
            raise ValueError(f"--labels count {len(labels)} != --log count {len(logs)}")

    keys = [s.strip() for s in args.keys.split(",") if s.strip()]
    if not keys:
        raise ValueError("No --keys provided.")

    # Create one subplot per key
    fig, axes = plt.subplots(nrows=len(keys), ncols=1, figsize=(10, 4 * len(keys)), dpi=150, sharex=True)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        for log_path, label in zip(logs, labels):
            if not log_path.exists():
                raise FileNotFoundError(str(log_path))
            series = parse_log_file(log_path, keys=[key])
            s = series[key]
            if len(s.steps) == 0:
                print(f"[WARN] key '{key}' not found in {log_path}")
                continue
            y = np.asarray(s.values, dtype=float)
            if args.smooth_window and args.smooth_window > 1:
                y = moving_average(y, window=args.smooth_window)
            ax.plot(s.steps, y, label=label, linewidth=1.5)
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
        if args.ylog:
            ax.set_yscale("log")
        ax.legend(loc="best")

    axes[-1].set_xlabel("step")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[OK] Saved loss curve to: {out_path}")


if __name__ == "__main__":
    main()

