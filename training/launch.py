# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
import os

# Ensure repository root is on sys.path so local packages (e.g. `vggt`) can be imported
# This is required when the entrypoint lives in a subdirectory (training/) and torchrun
# spawns subprocesses with a different import context.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="housecat_default",
        help="Name of the config file without .yaml extension"
    )
    
    # Parse known args to separate --config from Hydra overrides
    args, hydra_overrides = parser.parse_known_args()

    with initialize(version_base=None, config_path="config"):
        # Pass the overrides to Hydra's compose function
        cfg = compose(config_name=args.config, overrides=hydra_overrides)

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()

