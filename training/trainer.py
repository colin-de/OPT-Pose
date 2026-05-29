# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os


# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


import contextlib
import gc
import json
import logging
import math
import time
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from train_utils.checkpoint import DDPCheckpointSaver
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils.optimizer import construct_optimizers


class Trainer:
    """
    A generic trainer for DDP training. This should naturally support multi-node training.

    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (TensorBoard, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
            accum_steps: Number of steps to accumulate gradients before an optimizer step.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()

        # Store Hydra configurations
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # Store hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value
        
        # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        self.where = 0.0
        
        # Global step counter for monotonic W&B logging
        self.global_step = 0

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."

        # Instantiate components (model, loss, etc.)
        self._setup_components()
        self._setup_dataloaders()

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # Load checkpoint if available or specified
        if self.checkpoint_conf.resume_checkpoint_path is not None:
            self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)
        else:   
            ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
            if ckpt_path is not None:
                self._load_resuming_checkpoint(ckpt_path)

        # Wrap the model with DDP
        self._setup_ddp_distributed_training(distributed, device)
        
        # Barrier to ensure all processes are synchronized before starting
        dist.barrier()

    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        # Initialize the DDP process group
        dist.init_process_group(
            backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins)
        )
        self.rank = dist.get_rank()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        
        # Load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        
        # Filter out mismatched parameters
        current_model_dict = self.model.state_dict()
        filtered_state_dict = {}
        mismatched_keys = []
        
        for key, value in model_state_dict.items():
            if key in current_model_dict:
                if current_model_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    mismatched_keys.append(
                        f"{key}: checkpoint {value.shape} vs model {current_model_dict[key].shape}"
                    )
            else:
                filtered_state_dict[key] = value
        
        if self.rank == 0 and mismatched_keys:
            logging.warning(f"Skipping {len(mismatched_keys)} mismatched parameters:")
            for mk in mismatched_keys:
                logging.warning(f"  {mk}")
        
        missing, unexpected = self.model.load_state_dict(
            filtered_state_dict, strict=self.checkpoint_conf.strict
        )
        if self.rank == 0:
            logging.info(f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")

        # Load optimizer state if available and in training mode
        load_optim = getattr(self.checkpoint_conf, "load_optimizer", True)
        if (
            load_optim
            and "optimizer" in checkpoint
            and self.mode != "val"
            and self.optims
        ):
            logging.info(f"Loading optimizer state dict (rank {self.rank})")
            opt_state = checkpoint["optimizer"]
            try:
                if isinstance(opt_state, list):
                    # Multiple optimizers saved
                    if len(opt_state) != len(self.optims):
                        logging.warning(
                            f"Optimizer count mismatch: checkpoint has {len(opt_state)} states, "
                            f"but trainer has {len(self.optims)} optimizers. Will load the min count."
                        )
                    for i, optim in enumerate(self.optims[: len(opt_state)]):
                        optim.optimizer.load_state_dict(opt_state[i])
                elif isinstance(opt_state, dict):
                    # Single optimizer saved
                    self.optims[0].optimizer.load_state_dict(opt_state)
                else:
                    logging.warning(
                        f"Unexpected optimizer state type: {type(opt_state)}. Skipping optimizer load."
                    )
            except Exception as e:
                logging.error(f"Failed to load optimizer state: {e}")

        # Load training progress if enabled
        load_training_progress = getattr(self.checkpoint_conf, "load_training_progress", True)
        if load_training_progress:
            # Backward-compatible epoch restore: prefer 'epoch', fallback to 'prev_epoch'
            if "epoch" in checkpoint:
                self.epoch = checkpoint["epoch"]
            elif "prev_epoch" in checkpoint:
                self.epoch = checkpoint["prev_epoch"]
            self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}
            self.global_step = checkpoint.get("global_step", sum(self.steps.values()))
            self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)
            logging.info(f"Training progress loaded: epoch={self.epoch}, steps={self.steps}, global_step={self.global_step}")
        else:
            # Keep initial values when not loading training progress  
            logging.info("Skipping training progress load - starting from initial state")
            logging.info(f"Current training state: epoch={self.epoch}, steps={self.steps}, global_step={self.global_step}")

        # Load AMP scaler state if available (tied to optimizer load to avoid mismatches)
        if load_optim and self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        """Initializes all core training components using Hydra configs."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}
        self.global_step = 0

        # Instantiate components from configs
        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.optim_conf.amp.enabled)

        # Freeze specified model parameters if any
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

        # Log model summary on rank 0
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    def _setup_dataloaders(self):
        """Initializes train and validation datasets and dataloaders."""
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(
                self.data_conf.get('val', None), _recursive_=False
            )
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        assert isinstance(self.model, torch.nn.Module)

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """
        Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: A list of names for the checkpoint file (e.g., "checkpoint_latest").
                              If None, saves "checkpoint" and "checkpoint_{epoch}" on frequency.
        """
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            "epoch": epoch,
            "steps": self.steps,
            "global_step": self.global_step,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
        }
        
        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # Save the checkpoint for DDP only
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module

        saver.save_checkpoint(
            model=model,
            ema_models = None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )




    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            self.run_train()
            # Optionally run a final validation after all training is done
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def run_train(self):
        """Runs the main training loop over all epochs."""
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)
            
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
            self.train_epoch(dataloader)
            
            # Save checkpoint after each training epoch
            self.save_checkpoint(self.epoch)

            # Clean up memory
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Run validation at the specified frequency
            # Skips validation after the last training epoch, as it can be run separately.
            if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
                self.run_val()
            
            self.epoch += 1
        
        self.epoch -= 1

    def run_val(self):
        """Runs a full validation epoch if a validation dataset is available."""
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
        self.val_epoch(dataloader)
        
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


    @torch.no_grad()
    def val_epoch(self, val_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'val'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        limit_val_batches = (
            iters_per_epoch
            if self.limit_val_batches is None
            else self.limit_val_batches
        )

        for data_iter, batch in enumerate(val_loader):
            if data_iter > limit_val_batches:
                break
            
            # Skip None batches (empty batches filtered by collate function)
            if batch is None:
                continue
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)
            
            with torch.amp.autocast("cuda", enabled=False):
                batch = self._process_batch(batch)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            amp_type = self.optim_conf.amp.amp_dtype
            assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
            if amp_type == "bfloat16":
                amp_type = torch.bfloat16
            else:
                amp_type = torch.float16
            
            # compute output
            with torch.no_grad():
                with torch.amp.autocast(
                    "cuda",
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    val_loss_dict = self._step(
                        batch, self.model, phase, loss_meters
                    )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)


        return True

    def train_epoch(self, train_loader):        
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'train'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        for config in self.gradient_clipper.configs: 
            param_names = ",".join(config['module_names'])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")


        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )
        
        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            self.gradient_clipper.setup_clipping(self.model)

        for data_iter, batch in enumerate(train_loader):
            if data_iter > limit_train_batches:
                break
            
            # Skip None batches (empty batches filtered by collate function)
            if batch is None:
                continue
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            
            with torch.amp.autocast("cuda", enabled=False):
                batch = self._process_batch(batch)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            accum_steps = self.accum_steps

            if accum_steps==1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            self._run_steps_on_batch_chunks(
                chunked_batches, phase, loss_meters
            )

            # compute gradient and do SGD step
            assert data_iter <= limit_train_batches  # allow for off by one errors
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs
            
            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                )
                    
            # Log schedulers
            if self.steps[phase] % self.logging_conf.log_freq == 0:
                # Use global_step for W&B, phase-specific step for TensorBoard
                log_step = self.global_step if hasattr(self.tb_writer, 'log_visuals') else self.steps[phase]
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else (
                                    "" + f"{j}_"
                                    if len(optim.optimizer.param_groups) > 1
                                    else ""
                                )
                            )
                            self.tb_writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                log_step,
                            )
                self.tb_writer.log(
                    os.path.join("Optim", "where"),
                    self.where,
                    log_step,
                )

            # Clipping gradients and detecting diverging gradients
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():
                    loss_meters[f"Grad/{key}"].update(grad_norm)

            # Optimizer step
            for optim in self.optims:   
                self.scaler.step(optim.optimizer)
            self.scaler.update()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return True

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ):
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """        
        
        for optim in self.optims:   
            optim.zero_grad(set_to_none=True)

        # Filter out any empty chunks defensively and compute effective accumulation steps
        chunked_batches = [cb for cb in chunked_batches if _get_batch_size(cb) > 0]
        accum_steps = len(chunked_batches)
        if accum_steps == 0:
            # Nothing to do for this batch (e.g., global batch < accum_steps on this rank)
            logging.debug("Skipping batch with no non-empty chunks after accumulation split")
            return

        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        if amp_type == "bfloat16":
            amp_type = torch.bfloat16
        else:
            amp_type = torch.float16
        
        for i, chunked_batch in enumerate(chunked_batches):
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )

            with ddp_context:
                with torch.amp.autocast(
                    "cuda",
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    loss_dict = self._step(
                        chunked_batch, self.model, phase, loss_meters
                    )


                loss = loss_dict["objective"]
                loss_key = f"Loss/{phase}_loss_objective"
                batch_size = _get_batch_size(chunked_batch)

                if not math.isfinite(loss.item()):
                    error_msg = f"Loss is {loss.item()}, attempting to stop training"
                    logging.error(error_msg)
                    return

                loss /= accum_steps
                self.scaler.scale(loss).backward()
                loss_meters[loss_key].update(loss.item(), batch_size)


    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        """
        Applies a data augmentation by concatenating the original batch with a
        flipped version of itself.
        """
        tensor_keys = [
            "images", "depths", "depths_sensor", "extrinsics", "extrinsics_sym",
            "intrinsics", "cam_points", "world_points", "point_masks",
            "nocs", "inst_masks", "sizes", "choose_indices", "seq_id",
        ]
        # Per-sample labels: duplicate only (no flip) — symmetry is unchanged by image flip
        duplicate_only_keys = ["sym_codes"]
        string_keys = ["seq_name", "category_name"]

        def _repeat(t: torch.Tensor) -> torch.Tensor:
            if isinstance(t, torch.Tensor):
                if t.ndim > 1:
                    return torch.concatenate([t, torch.flip(t, dims=[1])], dim=0)
                else:
                    return torch.concatenate([t, t], dim=0)
            return t

        def _duplicate_only(t: torch.Tensor) -> torch.Tensor:
            if isinstance(t, torch.Tensor):
                return torch.concatenate([t, t], dim=0)
            return t

        for key in tensor_keys:
            if key in batch and batch[key] is not None:
                batch[key] = _repeat(batch[key])

        for key in duplicate_only_keys:
            if key in batch and batch[key] is not None:
                batch[key] = _duplicate_only(batch[key])

        # 1D label tensors (e.g., [B])
        if "cat_id" in batch and batch["cat_id"] is not None:
            batch["cat_id"] = torch.concatenate([batch["cat_id"], batch["cat_id"]], dim=0)

        for key in string_keys:
            if key in batch and batch[key] is not None:
                batch[key] = batch[key] * 2

        return batch

    def _process_batch(self, batch: Mapping):      
        if self.data_conf.train.common_config.repeat_batch:
            batch = self._apply_batch_repetition(batch)
        
        # Normalize camera extrinsics and points. The function returns new tensors.
        normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths, avg_scale = \
            normalize_camera_extrinsics_and_points_batch(
                extrinsics=batch["extrinsics"],
                cam_points=batch["cam_points"],
                world_points=batch["world_points"],
                depths=batch["depths"],
                point_masks=batch["point_masks"],
            )

        # normalized_size = (batch["sizes"] / avg_scale).clamp(min=1e-6)
        # Replace the original values in the batch with the normalized ones.
        batch["abs_extrinsics"] = batch["extrinsics_sym"].clone()  # [B, S, 3, 4] symmetric-aware pose
        batch["avg_scale"] = avg_scale
        batch["sizes"] = batch["sizes"].clone()
        batch["extrinsics"] = normalized_extrinsics
        batch["cam_points"] = normalized_cam_points
        batch["world_points"] = normalized_world_points
        batch["depths"] = normalized_depths

        return batch

    def _step(self, batch, model: nn.Module, phase: str, loss_meters: dict):
        """
        Performs a single forward pass, computes loss, and logs results.
        
        Returns:
            A dictionary containing the computed losses.
        """
        # Forward pass
        y_hat = model(
            images=batch["images"], 
            cat_labels=batch.get("cat_id"),  # Pass category IDs if available
            choose_indices=batch.get("choose_indices"),
            nocs_gt=batch.get("nocs"),  # Pass NOCS ground truth if available
            depth_sensor=batch.get("depths_sensor"),
            category_names=batch.get("category_name"),  # Pass category names for text prototype loss
            intrinsics=batch.get("intrinsics"),
            crop_boxes=batch.get("crop_boxes"),
            masks=batch.get("inst_masks"),
        )
        
        # Loss computation
        loss_dict = self.loss(y_hat, batch)
        
        # Combine all data for logging
        log_data = {**y_hat, **loss_dict, **batch}
        if "world_points" in y_hat:
            log_data['world_points'] = y_hat["world_points"]
            log_data['world_points_gt'] = batch["world_points"]
        if "nocs" in y_hat:
            log_data['nocs'] = y_hat["nocs"]
            if "nocs" in batch:
                log_data['nocs_gt'] = batch["nocs"]

        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        # Route visuals based on logger type
        try:
            from train_utils.wb_writer import WandbLogger as _WBLogger
            if isinstance(self.tb_writer, _WBLogger):
                self._log_wandb_visuals(log_data, phase, self.global_step)
            else:
                self._log_tb_visuals(log_data, phase, self.steps[phase])
        except Exception as e:
            # Print and log the exception once (rank 0) then fallback to TensorBoard
            if getattr(self, "rank", 0) == 0:
                print(f"Wandb visualization logging failed: {e}")
                logging.exception("Wandb visualization logging failed; falling back to TensorBoard")
            self._log_tb_visuals(log_data, phase, self.steps[phase])

        self.steps[phase] += 1
        self.global_step += 1
        return loss_dict

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """Updates average meters and logs scalar values to TensorBoard."""
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = data['extrinsics'].shape[0]
        
        for key in keys_to_log:
            if key in data:
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                value = self._apply_logging_weight(key, value)
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    # Use global_step for W&B, phase-specific step for TensorBoard
                    log_step = self.global_step if hasattr(self.tb_writer, 'log_visuals') else step
                    self.tb_writer.log(f"Values/{phase}/{key}", value, log_step)

    def _get_conf_value(self, conf: Optional[Mapping], key: str, default: float) -> float:
        if conf is None:
            return default
        try:
            return conf.get(key, default)
        except Exception:
            return getattr(conf, key, default)

    def _apply_logging_weight(self, key: str, value: float) -> float:
        """Apply configured loss weights to logged scalars (only where applicable)."""
        loss_conf = self.loss_conf or {}

        pose_conf = self._get_conf_value(loss_conf, "pose", None)
        if key == "loss_abs_pose":
            return value * self._get_conf_value(pose_conf, "weight", 1.0)
        if key == "loss_pose_R":
            return value * self._get_conf_value(pose_conf, "weight", 1.0) * self._get_conf_value(pose_conf, "rotation_weight", 1.0)
        if key == "loss_pose_t":
            return value * self._get_conf_value(pose_conf, "weight", 1.0) * self._get_conf_value(pose_conf, "translation_weight", 1.0)
        if key == "loss_pose_s":
            return value * self._get_conf_value(pose_conf, "weight", 1.0) * self._get_conf_value(pose_conf, "size_weight", 1.0)

        cc_conf = self._get_conf_value(loss_conf, "cycle_consistency", None)
        if key == "loss_cycle_consistency" and cc_conf is not None:
            return value * self._get_conf_value(cc_conf, "weight", 1.0)
        if key == "loss_cycle_R" and cc_conf is not None:
            return value * self._get_conf_value(cc_conf, "weight", 1.0) * self._get_conf_value(cc_conf, "rotation_weight", 1.0)
        if key == "loss_cycle_t" and cc_conf is not None:
            return value * self._get_conf_value(cc_conf, "weight", 1.0) * self._get_conf_value(cc_conf, "translation_weight", 0.1)

        nocs_conf = self._get_conf_value(loss_conf, "nocs", None)
        if key in ("loss_conf_nocs", "loss_reg_nocs", "loss_grad_nocs"):
            component_weights = self._get_conf_value(nocs_conf, "component_weights", {}) or {}
            comp_key = {
                "loss_conf_nocs": "conf",
                "loss_reg_nocs": "reg",
                "loss_grad_nocs": "grad",
            }[key]
            return value * self._get_conf_value(nocs_conf, "weight", 1.0) * component_weights.get(comp_key, 1.0)
        if key in ("loss_diversity", "loss_cd"):
            keypoint_weight = self._get_conf_value(nocs_conf, "keypoint_weight", 1.0)
            keypoint_loss_weights = self._get_conf_value(nocs_conf, "keypoint_loss_weights", {}) or {}
            kpt_key = "diversity" if key == "loss_diversity" else "cd"
            return value * keypoint_weight * keypoint_loss_weights.get(kpt_key, 1.0)

        infonce_conf = self._get_conf_value(loss_conf, "infonce", None)
        if key == "loss_infonce":
            return value * self._get_conf_value(infonce_conf, "weight", 1.0)
        if key == "loss_instance_consistency":
            return value * self._get_conf_value(infonce_conf, "instance_weight", 1.0)

        scale_conf = self._get_conf_value(loss_conf, "scale", None)
        if key == "loss_absolute_scale":
            return value * self._get_conf_value(scale_conf, "weight", 1.0)
        if key == "loss_abs_translation":
            return value * self._get_conf_value(scale_conf, "weight", 1.0) * self._get_conf_value(scale_conf, "t_weight", 1.0)
        if key == "loss_abs_size":
            return value * self._get_conf_value(scale_conf, "weight", 1.0) * self._get_conf_value(scale_conf, "s_weight", 1.0)

        rel_scale_conf = self._get_conf_value(loss_conf, "rel_scale", None)
        if key in ("loss_rel_scale", "loss_rel_scale_logspace"):
            return value * self._get_conf_value(rel_scale_conf, "weight", 1.0)

        recon_conf = self._get_conf_value(loss_conf, "recon", None)
        if key == "loss_recon":
            return value * self._get_conf_value(recon_conf, "weight", 1.0)
        if key == "loss_delta":
            return value * self._get_conf_value(recon_conf, "delta_weight", 1.0)

        return value

    def _get_effective_visual_freq(self, phase: str) -> Optional[int]:
        """Returns the effective visualization frequency for a phase.

        Falls back to the training frequency when the given phase
        is not explicitly configured. Returns None if not available.
        """
        try:
            freq_cfg = self.logging_conf.log_visual_frequency
        except Exception:
            return None

        if phase in freq_cfg and freq_cfg[phase] is not None:
            return int(freq_cfg[phase])
        if "train" in freq_cfg and freq_cfg["train"] is not None:
            return int(freq_cfg["train"])  # fallback to train
        return None

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs image or video visualizations to TensorBoard."""
        # visuals_keys_to_log may be omitted in config; guard against OmegaConf struct errors
        try:
            visuals_keys_cfg = self.logging_conf.visuals_keys_to_log
        except Exception:
            visuals_keys_cfg = None

        eff_freq = self._get_effective_visual_freq(phase)
        if not (
            self.logging_conf.log_visuals
            and (eff_freq is not None)
            and (eff_freq > 0)
            and (step % eff_freq == 0)
            and (visuals_keys_cfg is not None)
        ):
            return
        # Allow reusing training visual keys when val keys are not configured
        cfg_phase = phase if (visuals_keys_cfg is not None and phase in visuals_keys_cfg) else (
            "train" if (visuals_keys_cfg is not None and "train" in visuals_keys_cfg) else None
        )
        if cfg_phase is None:
            return

        keys_to_log = visuals_keys_cfg[cfg_phase]["keys_to_log"]
        assert len(keys_to_log) > 0, "Need to include some visual keys to log"
        modality = visuals_keys_cfg[cfg_phase]["modality"]
        assert modality in [
            "image",
            "video",
        ], "Currently only support video or image logging"

        name = f"Visuals/{phase}"

        visuals_to_log = torchvision.utils.make_grid(
            [
                torchvision.utils.make_grid(
                    batch[key][0],  # Ensure batch[key][0] is tensor and has at least 3 dimensions
                    nrow=self.logging_conf.visuals_per_batch_to_log,
                )
                for key in keys_to_log if key in batch and batch[key][0].dim() >= 3
            ],
            nrow=1,
        ).clamp(-1, 1)

        visuals_to_log = visuals_to_log.cpu()
        if visuals_to_log.dtype == torch.bfloat16:
            visuals_to_log = visuals_to_log.to(torch.float16)
        visuals_to_log = visuals_to_log.numpy()

        self.tb_writer.log_visuals(
            name, visuals_to_log, step, self.logging_conf.video_logging_fps
        )

    def _log_wandb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs ref + 3 random query views (image, depth, point map) to WandbLogger."""
        eff_freq = self._get_effective_visual_freq(phase)
        if not (
            self.logging_conf.log_visuals
            and (eff_freq is not None)
            and (eff_freq > 0)
            and (step % eff_freq == 0)
        ):
            return

        try:
            import numpy as np
        except Exception:
            return

        if ("images" not in batch):
            return

        images = batch["images"]  # [B, S, 3, H, W]
        # Only visualize depth when model predicts it; do not use GT-only depth as prediction
        depths = batch.get("depth", None)  # predicted depth: [B, S, H, W] or [B, S, H, W, 1]
        depths_gt = batch["depths"].unsqueeze(-1) if ("depths" in batch) else None
        pts3d = batch.get("world_points", None)  # [B, S, H, W, 3]
        pts3d_gt = batch.get("world_points_gt", None)
        point_masks = batch.get("point_masks", None)  # [B, S, H, W]

        if not torch.is_tensor(images):
            return

        B, S = images.shape[0], images.shape[1]
        if B < 1 or S < 1:
            return

        ref_idx = 0
        if S > 1:
            num_queries = min(3, S - 1)
            perm = torch.randperm(S - 1, device=images.device)[:num_queries] + 1
            query_indices = [int(i) for i in perm.tolist()]
        else:
            query_indices = []

        selected_indices = [ref_idx] + query_indices

        if pts3d is None and depths is not None and ("extrinsics" in batch) and ("intrinsics" in batch):
            try:
                from opt.utils.geometry import unproject_depth_map_to_point_map
                dep0 = depths[0]
                if dep0.ndim == 3:
                    dep0 = dep0[..., None]
                extr0 = batch["extrinsics"][0]
                intr0 = batch["intrinsics"][0]
                pts_np = unproject_depth_map_to_point_map(dep0, extr0, intr0)
                pts3d = torch.from_numpy(pts_np).to(images.device)
                pts3d = pts3d.unsqueeze(0)
            except Exception:
                pts3d = None

        # Compute global min/max for point maps across selected views, ignoring background
        if pts3d is not None:
            pts_selected = torch.stack([pts3d[0, i] for i in selected_indices], dim=0)  # [K, H, W, 3]
            if point_masks is not None and torch.is_tensor(point_masks):
                mask_selected = torch.stack([point_masks[0, i] for i in selected_indices], dim=0)  # [K, H, W]
                mask_selected = mask_selected.bool()
                # If no valid points in mask, fall back to unmasked behavior
                if mask_selected.any():
                    valid_pts = pts_selected[mask_selected]  # [N, 3]
                    valid_pts = torch.nan_to_num(valid_pts, nan=0.0, posinf=0.0, neginf=0.0)
                    pts_min = valid_pts.min(dim=0).values
                    pts_max = valid_pts.max(dim=0).values
                else:
                    pts_min = torch.nan_to_num(pts_selected.view(-1, 3), nan=0.0).min(dim=0).values
                    pts_max = torch.nan_to_num(pts_selected.view(-1, 3), nan=0.0).max(dim=0).values
            else:
                pts_min = torch.nan_to_num(pts_selected.view(-1, 3), nan=0.0).min(dim=0).values
                pts_max = torch.nan_to_num(pts_selected.view(-1, 3), nan=0.0).max(dim=0).values
            pts_range = torch.clamp(pts_max - pts_min, min=1e-6)

        panels = []
        for vidx in selected_indices:
            img = images[0, vidx]
            H, W = img.shape[-2], img.shape[-1]

            img_vis = img.detach().float().clamp(0.0, 1.0)
            # Apply background mask to image if available (keep bg black)
            if point_masks is not None and torch.is_tensor(point_masks):
                im_mask = point_masks[0, vidx]
                if im_mask.ndim == 2:
                    img_vis = img_vis * im_mask.float().unsqueeze(0)
            img_vis = img_vis.cpu().numpy()
            img_vis = np.transpose(img_vis, (1, 2, 0))

            # Build row dynamically: always image, then optional depth (pred+gt), then optional pointmaps (pred+gt)
            row_parts = [img_vis]

            if depths is not None and depths_gt is not None:
                dep = depths[0, vidx]
                dep_gt = depths_gt[0, vidx]
                if dep.ndim == 2:
                    pass
                elif dep.ndim == 3 and dep.shape[-1] == 1:
                    dep = dep[..., 0]
                    dep_gt = dep_gt[..., 0]
                else:
                    dep = dep.squeeze(-1)
                    dep_gt = dep_gt.squeeze(-1)
                dep = dep.detach().float()
                dep_gt = dep_gt.detach().float()
                dep = torch.nan_to_num(dep, nan=0.0, posinf=0.0, neginf=0.0)
                dep_gt = torch.nan_to_num(dep_gt, nan=0.0, posinf=0.0, neginf=0.0)

                # Use point mask to ignore background for normalization
                if point_masks is not None and torch.is_tensor(point_masks):
                    dmask = point_masks[0, vidx].bool()
                    # If mask shape mismatches depth due to preprocessing, skip masking to avoid crashes
                    if dmask.shape != dep.shape:
                        dmask = None
                    if dmask is not None and dmask.any():
                        dep_vals = dep[dmask]
                        dmin = torch.quantile(dep_vals, 0.01)
                        dmax = torch.quantile(dep_vals, 0.99)
                        dep_norm = (dep - dmin) / torch.clamp(dmax - dmin, min=1e-6)
                        dep_norm = torch.clamp(dep_norm, 0.0, 1.0)
                        dep_norm = dep_norm * dmask.float()

                        dep_gt_vals = dep_gt[dmask]
                        dmin_gt = torch.quantile(dep_gt_vals, 0.01)
                        dmax_gt = torch.quantile(dep_gt_vals, 0.99)
                        dep_gt_norm = (dep_gt - dmin_gt) / torch.clamp(dmax_gt - dmin_gt, min=1e-6)
                        dep_gt_norm = torch.clamp(dep_gt_norm, 0.0, 1.0)
                        dep_gt_norm = dep_gt_norm * dmask.float()
                    else:
                        dep_norm = torch.zeros_like(dep)
                        dep_gt_norm = torch.zeros_like(dep_gt)
                else:
                    dmin = torch.quantile(dep.flatten(), 0.01)
                    dmax = torch.quantile(dep.flatten(), 0.99)
                    dep_norm = (dep - dmin) / torch.clamp(dmax - dmin, min=1e-6)
                    dep_norm = torch.clamp(dep_norm, 0.0, 1.0)

                    dmin_gt = torch.quantile(dep_gt.flatten(), 0.01)
                    dmax_gt = torch.quantile(dep_gt.flatten(), 0.99)
                    dep_gt_norm = (dep_gt - dmin_gt) / torch.clamp(dmax_gt - dmin_gt, min=1e-6)
                    dep_gt_norm = torch.clamp(dep_gt_norm, 0.0, 1.0)

                # Resize to image size if needed
                if dep_norm.shape != (H, W):
                    dep_norm = F.interpolate(dep_norm.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest').squeeze(0).squeeze(0)
                if dep_gt_norm.shape != (H, W):
                    dep_gt_norm = F.interpolate(dep_gt_norm.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest').squeeze(0).squeeze(0)

                dep_vis = dep_norm.cpu().numpy()
                dep_vis = np.stack([dep_vis, dep_vis, dep_vis], axis=-1)

                dep_gt_vis = dep_gt_norm.cpu().numpy()
                dep_gt_vis = np.stack([dep_gt_vis, dep_gt_vis, dep_gt_vis], axis=-1)

                row_parts += [dep_vis, dep_gt_vis]

            if pts3d is not None and pts3d_gt is not None:
                pm = pts3d[0, vidx].detach().float()  # [H, W, 3]
                pm = torch.nan_to_num(pm, nan=0.0, posinf=0.0, neginf=0.0)
                pm_norm = torch.clamp((pm - pts_min) / pts_range, 0.0, 1.0)
                if point_masks is not None and torch.is_tensor(point_masks):
                    pmask = point_masks[0, vidx].bool()
                    if pmask.shape == pm_norm.shape[:2]:
                        pm_norm = pm_norm * pmask.unsqueeze(-1).float()
                # Resize to image size if needed
                if pm_norm.shape[0] != H or pm_norm.shape[1] != W:
                    pm_norm_rs = pm_norm.permute(2, 0, 1).unsqueeze(0)
                    pm_norm_rs = F.interpolate(pm_norm_rs, size=(H, W), mode='bilinear', align_corners=False)
                    pm_norm = pm_norm_rs.squeeze(0).permute(1, 2, 0)
                pm_vis = pm_norm.detach().cpu().numpy()

                pm_gt = pts3d_gt[0, vidx].detach().float()
                pm_gt = torch.nan_to_num(pm_gt, nan=0.0, posinf=0.0, neginf=0.0)
                pm_gt_norm = torch.clamp((pm_gt - pts_min) / pts_range, 0.0, 1.0)
                if point_masks is not None and torch.is_tensor(point_masks):
                    pmask = point_masks[0, vidx].bool()
                    if pmask.shape == pm_gt_norm.shape[:2]:
                        pm_gt_norm = pm_gt_norm * pmask.unsqueeze(-1).float()
                if pm_gt_norm.shape[0] != H or pm_gt_norm.shape[1] != W:
                    pm_gt_norm_rs = pm_gt_norm.permute(2, 0, 1).unsqueeze(0)
                    pm_gt_norm_rs = F.interpolate(pm_gt_norm_rs, size=(H, W), mode='bilinear', align_corners=False)
                    pm_gt_norm = pm_gt_norm_rs.squeeze(0).permute(1, 2, 0)
                pm_gt_vis = pm_gt_norm.detach().cpu().numpy()
                row_parts += [pm_vis, pm_gt_vis]

            # NOCS visualization (map [-0.5, 0.5] -> [0,1]) if available
            nocs_pred = batch.get("nocs", None)
            nocs_gt = batch.get("nocs_gt", None)
            fps_indices = batch.get("fps_indices", None)
            inst_masks = batch.get("inst_masks", None)
            
            if nocs_gt is not None:
                # Handle GT NOCS visualization (might be sparse or dense)
                if len(nocs_gt.shape) == 4:
                    # Sparse GT
                    ngt_dense = torch.zeros(H * W, 3, device=nocs_gt.device, dtype=nocs_gt.dtype)
                    if fps_indices is not None and vidx < fps_indices.shape[1]:
                        sparse_nocs_gt = nocs_gt[0, vidx].detach().float()  # [K, 3]
                        indices = fps_indices[0, vidx]  # [K]
                        ngt_dense[indices] = sparse_nocs_gt
                        ngt_dense = ngt_dense.view(H, W, 3)
                    else:
                        ngt_dense = ngt_dense.view(H, W, 3)
                    ngt = ngt_dense
                else:
                    # Dense GT
                    ngt = nocs_gt[0, vidx].detach().float()
                
                ngt = torch.nan_to_num(ngt, nan=0.0, posinf=0.0, neginf=0.0)
                ngt_norm = torch.clamp((ngt + 0.5), 0.0, 1.0)
                if inst_masks is not None and torch.is_tensor(inst_masks):
                    imask = inst_masks[0, vidx].bool()
                    if imask.shape == ngt_norm.shape[:2]:
                        ngt_norm = ngt_norm * imask.unsqueeze(-1).float()
                if ngt_norm.shape[0] != H or ngt_norm.shape[1] != W:
                    ngt_norm_rs = ngt_norm.permute(2, 0, 1).unsqueeze(0)
                    ngt_norm_rs = F.interpolate(ngt_norm_rs, size=(H, W), mode='bilinear', align_corners=False)
                    ngt_norm = ngt_norm_rs.squeeze(0).permute(1, 2, 0)
                nocs_gt_vis = ngt_norm.detach().cpu().numpy()
                row_parts.append(nocs_gt_vis)

            # Visual feature map visualization using PCA if available
            visual_feats = batch.get("visual_feats", None)
            
            if visual_feats is not None:
                # visual_feats: [B, S, C, H_feat, W_feat]
                vf = visual_feats[0, vidx].detach().float()  # [C, H_feat, W_feat]
                C_feat, H_feat, W_feat = vf.shape
                
                # Get instance mask and resize to feature map size
                fg_mask = None
                if inst_masks is not None and torch.is_tensor(inst_masks):
                    imask = inst_masks[0, vidx].bool()
                    if imask.shape[0] != H_feat or imask.shape[1] != W_feat:
                        imask_t = imask.unsqueeze(0).unsqueeze(0).float()
                        imask_t = F.interpolate(imask_t, size=(H_feat, W_feat), mode='nearest')
                        fg_mask = imask_t.squeeze(0).squeeze(0).bool()
                    else:
                        fg_mask = imask
                
                # Reshape to [H_feat*W_feat, C] for PCA
                vf_reshaped = vf.permute(1, 2, 0).reshape(-1, C_feat)  # [H_feat*W_feat, C]
                
                # Apply PCA only on foreground region
                from sklearn.decomposition import PCA
                if fg_mask is not None:
                    fg_mask_flat = fg_mask.reshape(-1).cpu().numpy()
                    vf_fg = vf_reshaped[torch.from_numpy(fg_mask_flat).to(vf.device)].cpu().numpy()
                    if vf_fg.shape[0] > 3:  # Need at least some foreground pixels
                        pca = PCA(n_components=3)
                        vf_fg_pca = pca.fit_transform(vf_fg)  # [N_fg, 3]
                        # Transform all pixels using fitted PCA
                        vf_pca = pca.transform(vf_reshaped.cpu().numpy())  # [H_feat*W_feat, 3]
                    else:
                        vf_pca = np.zeros((vf_reshaped.shape[0], 3))
                else:
                    pca = PCA(n_components=3)
                    vf_pca = pca.fit_transform(vf_reshaped.cpu().numpy())  # [H_feat*W_feat, 3]
                
                # Normalize to [0, 1] for visualization using foreground range
                if fg_mask is not None:
                    fg_mask_flat = fg_mask.reshape(-1).cpu().numpy()
                    vf_pca_fg = vf_pca[fg_mask_flat]
                    vf_pca_min = vf_pca_fg.min(axis=0, keepdims=True)
                    vf_pca_max = vf_pca_fg.max(axis=0, keepdims=True)
                else:
                    vf_pca_min = vf_pca.min(axis=0, keepdims=True)
                    vf_pca_max = vf_pca.max(axis=0, keepdims=True)
                vf_pca_norm = (vf_pca - vf_pca_min) / np.clip(vf_pca_max - vf_pca_min, 1e-6, None)
                
                # Reshape back to [H_feat, W_feat, 3] and apply mask
                vf_vis = vf_pca_norm.reshape(H_feat, W_feat, 3)
                if fg_mask is not None:
                    vf_vis = vf_vis * fg_mask.cpu().numpy()[..., None]
                
                # Resize to image size if needed
                if H_feat != H or W_feat != W:
                    vf_vis_t = torch.from_numpy(vf_vis).permute(2, 0, 1).unsqueeze(0).float()
                    vf_vis_t = F.interpolate(vf_vis_t, size=(H, W), mode='bilinear', align_corners=False)
                    vf_vis = vf_vis_t.squeeze(0).permute(1, 2, 0).numpy()
                
                row_parts.append(vf_vis)

            # If neither depth nor points were available, row_parts will just contain the image
            panel = np.concatenate(row_parts, axis=1)
            panels.append(panel)

        grid = np.concatenate(panels, axis=0)

        name = f"Visuals/{phase}/ref_plus{len(query_indices)}"
        # Use the global step for monotonic W&B logging
        self.tb_writer.log_visuals(name, grid, step, self.logging_conf.video_logging_fps)


def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Splits a batch into smaller chunks for gradient accumulation.

    Ensures no zero-sized chunks are produced and remainders are distributed to the
    earliest chunks. If the global batch size is smaller than the requested number
    of accumulation steps, it reduces the effective steps accordingly.
    """
    if accum_steps <= 1:
        return [batch]

    batch_size = _get_batch_size(batch)
    if batch_size <= 0:
        return []

    effective_steps = min(accum_steps, batch_size)
    if effective_steps == 1:
        return [batch]

    base = batch_size // effective_steps
    rem = batch_size % effective_steps

    chunks: List[Mapping] = []
    start = 0
    for i in range(effective_steps):
        extra = 1 if i < rem else 0
        end = start + base + extra
        if end > start:
            chunks.append(_slice_batch(batch, start, end))
        start = end

    return chunks

def is_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of primitive types (str, int, float, bool)."""
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )

def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """
    Recursively splits tensors and sequences within a data structure into chunks.

    Args:
        data: The data structure to split (e.g., a dictionary of tensors).
        chunk_id: The index of the chunk to retrieve.
        num_chunks: The total number of chunks to split the data into.

    Returns:
        A chunk of the original data structure.
    """
    if isinstance(data, torch.Tensor):
        # Handle 0-dimensional tensors (scalars) - they can't be chunked
        if data.dim() == 0:
            return data
        # Handle multi-dimensional tensors with remainder distribution
        n = len(data)
        base = n // num_chunks
        rem = n % num_chunks
        start = base * chunk_id + min(chunk_id, rem)
        end = start + base + (1 if chunk_id < rem else 0)
        return data[start:end]
    elif is_sequence_of_primitives(data):
        # Handle sequences of primitive objects with remainder distribution
        n = len(data)
        base = n // num_chunks
        rem = n % num_chunks
        start = base * chunk_id + min(chunk_id, rem)
        end = start + base + (1 if chunk_id < rem else 0)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, str):
        # NOTE: this is a hack to support string keys in the batch
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data

def _get_batch_size(batch: Mapping) -> int:
    """Returns the batch dimension (0th dim) using a known tensor key, or 0 if unknown."""
    try:
        if isinstance(batch, Mapping):
            if "images" in batch and torch.is_tensor(batch["images"]):
                return int(batch["images"].shape[0])
            # Fallback: find first tensor with a batch dimension
            for v in batch.values():
                if torch.is_tensor(v) and v.dim() >= 1:
                    return int(v.shape[0])
        return 0
    except Exception:
        return 0

def _slice_batch(data: Any, start: int, end: int) -> Any:
    """Slices a nested batch structure along the first dimension [start:end]."""
    if isinstance(data, torch.Tensor):
        if data.dim() == 0:
            return data
        return data[start:end]
    elif is_sequence_of_primitives(data):
        return data[start:end]
    elif isinstance(data, Mapping):
        return {k: _slice_batch(v, start, end) for k, v in data.items()}
    elif isinstance(data, str):
        return data
    elif isinstance(data, Sequence):
        return [_slice_batch(v, start, end) for v in data]
    else:
        return data
