## neptune-pytorch 3.0.0

### Breaking Changes
- **Neptune 3.x only**: This version only supports Neptune 3.x and requires a Neptune 3.x API Key.
- **Dropped support for logging model binaries and checkpoints**: The `log_model()` and `log_checkpoint()` methods have been removed.
- **Removed gradient and parameter tracking flags from NeptuneLogger constructor**: The `log_gradients`, `log_parameters`, and `log_freq` parameters have been removed from the NeptuneLogger constructor. Use `log_model_internals()` parameters instead.
- **Updated namespace structure**: Model internals are now logged under `{base_namespace}/model/internals/{prefix}/{metric_type}/{layer}/{statistic}`. For example, `pytorch_experiment/model/internals/train/activations/conv1/mean`.
- **Changed base_namespace default**: The `base_namespace` parameter now defaults to `None` instead of `"training"`.
- **Dropped support for Python <=3.9**
-
### New Features
- **Enhanced model monitoring**: Added comprehensive model internals tracking with configurable statistics (mean, std, norm, min, max, var, abs_mean, hist).
- **Flexible layer filtering**: Added `track_layers` parameter to specify which layer types to track.
- **Configurable statistics**: Added `tensor_stats` parameter to customize which statistics to compute for tracked tensors.

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
