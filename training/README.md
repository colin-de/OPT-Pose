# HouseCat6D Training

This directory contains the supported training path for the code release. The only release config is:

```text
training/config/housecat_default.yaml
```

Run training with explicit dataset and checkpoint paths:

```bash
torchrun --nproc_per_node=<N> training/launch.py \
  --config housecat_default \
  data.train.dataset.dataset_configs.0.data_root=<HouseCat6D> \
  checkpoint.resume_checkpoint_path=<checkpoint>
```

If training from scratch or handling checkpoint loading yourself, omit the checkpoint override and leave `checkpoint.resume_checkpoint_path=null`.
