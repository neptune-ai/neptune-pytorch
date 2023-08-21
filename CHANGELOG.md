## neptune-pytorch 2.0.0

### Changes
- Rename `save_model` to `log_model` and `save_checkpoint` to `log_checkpoint`. (https://github.com/neptune-ai/neptune-pytorch/pull/9)
- Prefix private methods with underscore. (https://github.com/neptune-ai/neptune-pytorch/pull/12)
- Add docstrings for `log_model` and `log_checkpoint`. (https://github.com/neptune-ai/neptune-pytorch/pull/11)


## neptune-pytorch 1.1.0 (YANKED)

### Fixes
- Rename `save_model` to `log_model` and `save_checkpoint` to `log_checkpoint`. (https://github.com/neptune-ai/neptune-pytorch/pull/9)

## neptune-pytorch 1.0.1

### Fixes
- Make `torchviz` optional dependency. (https://github.com/neptune-ai/neptune-pytorch/pull/8)

## neptune-pytorch 1.0.0

### Fixes
- Change where `checkpoints` are logged. Previously they we logged under `base_namespace/model` but now they will be logged under `base_namespace/model/checkpoints` (https://github.com/neptune-ai/neptune-pytorch/pull/5)
- Add warning if `dot` is not installed instead of hard error. Also, improve clean-up of visualization files (https://github.com/neptune-ai/neptune-pytorch/pull/6)
### Features
- Create `NeptuneLogger` for logging metadata (https://github.com/neptune-ai/neptune-pytorch/pull/1)

## neptune-pytorch 0.2.0

### Fixes
- Change where `checkpoints` are logged. Previously they we logged under `base_namespace/model` but now they will be logged under `base_namespace/model/checkpoints` (https://github.com/neptune-ai/neptune-pytorch/pull/5)
- Add warning if `dot` is not installed instead of hard error. Also, improve clean-up of visualization files (https://github.com/neptune-ai/neptune-pytorch/pull/6)


## neptune-pytorch 0.1.0 (initial release)

### Features
- Create `NeptuneLogger` for logging metadata (https://github.com/neptune-ai/neptune-pytorch/pull/1)
