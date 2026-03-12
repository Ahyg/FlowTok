import os
import sys

from absl import app, flags
from ml_collections import ConfigDict

# Ensure this script can import train_sat2radar_v2v from the same folder.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import train_sat2radar_v2v as base_train


FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "debug_enabled",
    True,
    "Enable loss/gradient path debug logging in training script.",
)
flags.DEFINE_integer(
    "debug_steps",
    5,
    "Run debug checks for the first N training steps.",
)
flags.DEFINE_integer(
    "debug_log_every",
    1,
    "Log debug information every N steps during debug window.",
)
flags.DEFINE_float(
    "debug_grad_eps_on",
    1e-12,
    "Minimum gradient norm considered as 'has gradient'.",
)
flags.DEFINE_float(
    "debug_grad_eps_off",
    1e-14,
    "Maximum gradient norm considered as 'no gradient leak'.",
)


def main(_):
    config = FLAGS.config
    if config is None:
        raise ValueError("--config is required.")

    debug_cfg = ConfigDict()
    debug_cfg.enabled = bool(FLAGS.debug_enabled)
    debug_cfg.max_steps = int(FLAGS.debug_steps)
    debug_cfg.log_every = int(FLAGS.debug_log_every)
    debug_cfg.grad_eps_on = float(FLAGS.debug_grad_eps_on)
    debug_cfg.grad_eps_off = float(FLAGS.debug_grad_eps_off)
    config.debug = debug_cfg

    base_train.train(config)


if __name__ == "__main__":
    app.run(main)
