# Neptune - PyTorch integration

<div align="center">

[![PyPI version](https://badge.fury.io/py/neptune-pytorch.svg)](https://badge.fury.io/py/neptune-pytorch)
[![neptune_scale](https://img.shields.io/badge/neptune__scale-0.14.0+-orange.svg)](https://pypi.org/project/neptune-scale/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

The **Neptune-PyTorch** integration simplifies tracking your PyTorch experiments with Neptune by providing automated tracking of PyTorch model internals including activations, gradients, and parameters.

## Installation

```bash
pip install -U neptune-pytorch
```

## Requirements

- **Neptune 3.x**: Requires a Neptune 3.x account. See the [Getting Started Guide](https://docs.neptune.ai/setup) for setup instructions.
- **Python 3.10+**: Minimum Python version requirement
- **PyTorch 1.11+**: For tensor operations and model support
- **NumPy 1.20+**: For numerical computations

## Quickstart

The below quickstart example logs the following data to Neptune:

- **Model architecture**: Visual diagram and summary of the neural network
- **Training metrics**: Loss curves and epoch progress
- **Layer activations**: Mean, std, norm, histograms for each layer
- **Gradient analysis**: Gradient statistics to detect vanishing/exploding gradients
- **Parameter tracking**: Weight and bias distributions over time

```python
import torch
import torch.nn as nn
import torch.optim as optim
from neptune_scale import Run
from neptune_pytorch import NeptuneLogger

# Initialize Neptune run
run = Run(project="your-project/experiment-tracking")

# Create your PyTorch model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Initialize Neptune logger with model tracking
neptune_logger = NeptuneLogger(
    run=run,
    model=model,
    base_namespace="mnist_classification",  # Organizes all metrics under this folder
    log_model_diagram=True,  # Generates model architecture diagram
)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop with comprehensive tracking
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log training metrics to Neptune
        run.log_metrics({
            f"{neptune_logger.base_namespace}/batch/loss": loss.item(),
            f"{neptune_logger.base_namespace}/epoch": epoch,
        })

        # Track model internals every 10 steps
        if batch_idx % 10 == 0:
            neptune_logger.log_model_internals(
                step=batch_idx,
                prefix="train",
                track_activations=True,   # Monitor activation patterns
                track_gradients=True,     # Track gradient flow
                track_parameters=True     # Log parameter statistics
            )
```

## Advanced configuration

The below example demonstrates the following additional features:

- **Layer filtering**: Only track Conv2d and Linear layers (reduces overhead)
- **Custom statistics**: Use mean, std, hist instead of all 8 statistics
- **Phase-specific tracking**: Different tracking strategies for train/validation
- **Frequency control**: Track every 20 steps in training, every 50 in validation

```python
import torch
import torch.nn as nn
from neptune_scale import Run
from neptune_pytorch import NeptuneLogger

# Initialize Neptune run
run = Run(project="your-project/advanced-tracking")

# Create a more complex model (e.g., CNN for image classification)
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNNModel()

# Advanced Neptune logger configuration
neptune_logger = NeptuneLogger(
    run=run,
    model=model,
    base_namespace="cnn_experiment",  # Custom organization folder
    track_layers=[nn.Conv2d, nn.Linear],  # Only track conv and linear layers
    tensor_stats=["mean", "norm", "hist"],  # Custom statistics (faster than default)
    log_model_diagram=True,  # Log model summary and diagram
)

# Training with phase-specific tracking
for epoch in range(num_epochs):
    # Training phase - comprehensive tracking
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # ... your training code ...
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Track everything during training
        if batch_idx % 20 == 0:  # Every 20 steps
            neptune_logger.log_model_internals(
                step=batch_idx,
                prefix="train",
                track_activations=True,   # Monitor activation patterns
                track_gradients=True,     # Track gradient flow
                track_parameters=True     # Log parameter statistics
            )

    # Validation phase - lightweight tracking
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            # ... your validation code ...
            output = model(data)
            val_loss = criterion(output, target)

            # Only track activations during validation (faster)
            if batch_idx % 50 == 0:  # Every 50 steps
                neptune_logger.log_model_internals(
                    step=batch_idx,
                    prefix="validation",
                    track_activations=True,   # Monitor activation patterns
                    track_gradients=False,    # Skip gradients (no backward pass)
                    track_parameters=False    # Skip parameters (expensive)
                )
```


## Features

### Model monitoring

- **Layer activations**: Track activation patterns across all layers with 8 different statistics
- **Gradient analysis**: Monitor gradient flow and detect vanishing/exploding gradients
- **Parameter tracking**: Log parameter statistics and distributions for model analysis
- **Custom statistics**: Choose from mean, std, norm, min, max, var, abs_mean, and hist

### Configuration options

- **Layer filtering**: Track only specific layer types (Conv2d, Linear, etc.)
- **Phase organization**: Separate tracking for training/validation phases with custom prefixes
- **Custom namespaces**: Organize experiments with custom folder structures

### Visualizations

- **Model architecture**: Automatic model diagram generation with torchviz
- **Distribution histograms**: 50-bin histograms for all tracked metrics
- **Real-time monitoring**: Live tracking during training with Neptune
- **Comparative analysis**: Easy comparison across experiments and runs

### Integration

- **Minimal setup**: Simple integration with existing code
- **PyTorch native**: Works with existing PyTorch workflows

### Performance optimization

Since parameter logging can be expensive for large models, you can control the frequency explicitly:

```python
for step in range(num_steps):
    # ... training code ...

    # Log lightweight metrics every step
    neptune_logger.log_model_internals(
        step=step,
        track_activations=True,
        track_gradients=True,
        track_parameters=False  # Skip expensive parameter logging
    )

    # Log expensive parameters less frequently
    if step % 100 == 0:
        neptune_logger.log_model_internals(
            step=step,
            track_activations=False,
            track_gradients=False,
            track_parameters=True
        )
```

### Namespace structure

The integration organizes all logged data under a clear hierarchical and customizable namespace structure:

```
{base_namespace}/                   # Optional custom top-level folder
├── batch/
│   └── loss                        # Training loss per batch (logged by the user)
├── model/
│   ├── summary                     # Model architecture (if log_model_diagram=True)
│   └── internals/                  # Model internals tracking
│       └── {prefix}/               # Optional prefix (e.g., "train", "validation")
│           ├── activations/        # Layer activations
│           │   └── {layer_name}/
│           │       ├── mean        # Mean activation value
│           │       ├── std         # Standard deviation
│           │       ├── norm        # L2 norm
│           │       ├── min         # Minimum value
│           │       ├── max         # Maximum value
│           │       ├── var         # Variance
│           │       ├── abs_mean    # Mean of absolute values
│           │       └── hist        # Histogram (50 bins)
│           ├── gradients/          # Layer gradients
│           │   └── {layer_name}/
│           │       └── {statistic} # Same statistics as activations
│           └── parameters/         # Model parameters
│               └── {layer_name}/
│                   └── {statistic} # Same statistics as activations
```

**Example namespaces:**

With `base_namespace="my_experiment"`:

- `my_experiment/batch/loss` - Training loss
- `my_experiment/model/summary` - Model architecture
- `my_experiment/model/internals/activations/conv/1/mean` - Mean activation (no prefix)
- `my_experiment/model/internals/train/activations/conv/1/mean` - Mean activation (with "train" prefix)
- `my_experiment/model/internals/validation/gradients/linear1/norm` - L2 norm of gradients (with "validation" prefix)

With `base_namespace=None`:

- `batch/loss` - Training loss
- `model/summary` - Model architecture
- `model/internals/activations/conv/1/mean` - Mean activation (no prefix)
- `model/internals/train/activations/conv/1/mean` - Mean activation (with "train" prefix)
- `model/internals/validation/gradients/linear1/norm` - L2 norm of gradients (with "validation" prefix)

**Layer name handling:**

- Dots in layer names are automatically replaced with forward slashes for proper namespace organization
- Example: `seq_model.0.weight` becomes `seq_model/0/weight` in the namespace
- Example: `module.submodule.layer` becomes `module/submodule/layer` in the namespace

**Available statistics:** `mean`, `std`, `norm`, `min`, `max`, `var`, `abs_mean`, `hist`

## API reference

### NeptuneLogger

```python
NeptuneLogger(
    run: Run,
    model: torch.nn.Module,
    base_namespace: Optional[str] = None,
    track_layers: Optional[List[Type[nn.Module]]] = None,
    tensor_stats: Optional[List[TensorStatType]] = None,
    log_model_diagram: bool = False
)
```

**Parameters:**

- `run`: Neptune run object for logging
- `model`: PyTorch model to track
- `base_namespace`: Optional top-level folder for organization (default: `None`)
- `track_layers`: List of layer types to track (default: `None` = all layers)
- `tensor_stats`: Statistics to compute (default: `["mean", "norm", "hist"]`)
- `log_model_diagram`: Log the model summary and diagram (default: `False`)

### log_model_internals()

```python
log_model_internals(
    step: int,
    track_activations: bool = True,
    track_gradients: bool = True,
    track_parameters: bool = False,
    prefix: Optional[str] = None
)
```

**Parameters:**

- `step`: Current training step for logging
- `track_activations`: Track layer activations (default: `True`)
- `track_gradients`: Track layer gradients (default: `True`)
- `track_parameters`: Track model parameters (default: `False`)
- `prefix`: Optional phase identifier (e.g., "train", "validation")

### Available statistics

| Statistic  | Description             | Use case                              |
| ---------- | ----------------------- | ------------------------------------- |
| `mean`     | Mean value              | Monitor activation levels             |
| `std`      | Standard deviation      | Detect activation variance            |
| `norm`     | L2 norm                 | Monitor gradient/activation magnitude |
| `min`      | Minimum value           | Detect dead neurons                   |
| `max`      | Maximum value           | Detect saturation                     |
| `var`      | Variance                | Monitor activation spread             |
| `abs_mean` | Mean of absolute values | Monitor activation strength           |
| `hist`     | 50-bin histogram        | Visualize distributions               |

## Contributing

Contributions to neptune-pytorch are welcome. Here's how you can help:

### Report issues

- Found a bug? [Open an issue](https://github.com/neptune-ai/neptune-pytorch/issues)
- Include Python version, PyTorch version, and error traceback
- Provide a minimal reproducible example

### Suggest features

- Have an idea? [Create a feature request](https://github.com/neptune-ai/neptune-pytorch/issues)
- Describe the use case and expected behavior
- Check existing issues first to avoid duplicates

### Contribute code

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run tests: `pytest tests/`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to remote: `git push origin feature/amazing-feature`
7. Open a Pull Request

## Support

### Get help

<!--- 📖 **Documentation**: [Neptune PyTorch Docs](https://docs.neptune.ai/integrations/pytorch/)-->

- 🔧 **Troubleshooting**: [Common Issues Guide](https://docs.neptune.ai/troubleshooting)
- 🎫 **Support Portal**: [Reach out to us](https://support.neptune.ai)

### Resources

- [Neptune Documentation](https://docs.neptune.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Neptune Examples](https://github.com/neptune-ai/scale-examples)
- [Neptune Blog](https://neptune.ai/blog/)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ by the Neptune team**

</div>
