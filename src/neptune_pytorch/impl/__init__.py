#
# Copyright (c) 2023, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
__all__ = ["__version__", "NeptuneLogger"]

import os
import uuid
import warnings
from typing import (
    List,
    Optional,
    Type,
)

import torch
import torch.nn as nn
from neptune_scale import Run

from neptune_pytorch.impl._torchwatcher import (
    TensorStatType,
    _TorchWatcher,
)
from neptune_pytorch.impl.version import __version__

_IS_TORCHVIZ_AVAILABLE = True
try:
    import torchviz
    from graphviz import ExecutableNotFound
except ImportError:
    _IS_TORCHVIZ_AVAILABLE = False

_INTEGRATION_VERSION_KEY = "source_code/integrations/neptune-pytorch"


class NeptuneLogger:
    """Captures model training metadata and logs them to Neptune.

    This logger provides comprehensive model monitoring capabilities including:
    - Model diagram and summary
    - Layer activations, gradients, and parameters tracking
    - Configurable statistics computation (mean, std, norm, histograms, etc.)
    - Flexible layer filtering and parameter logging frequency control

    Args:
        run: Neptune run object.
        model: PyTorch model whose metadata will be tracked.
        base_namespace: Optional custom top-level folder for organizing logged data.
            If None, metrics are logged under a "model" folder at the root level.
        track_layers: List of PyTorch layer types to track. If None, tracks all layers.
        tensor_stats: List of statistics to compute for tracked tensors.
            Available options: "mean", "std", "norm", "min", "max", "var", "abs_mean", "hist".
            Defaults to ["mean", "norm", "hist"].
        log_model_diagram: Whether to save the model summary and diagram.
            Requires torchviz to be installed: https://pypi.org/project/torchviz/

    Example:
        from neptune_scale import Run
        from neptune_pytorch import NeptuneLogger
        import torch.nn as nn
        import torch.nn.functional as F

        run = Run()

        # Basic usage with default settings
        neptune_callback = NeptuneLogger(run=run, model=model)

        # Advanced usage with custom configuration
        neptune_callback = NeptuneLogger(
            run=run,
            model=model,
            track_layers=[nn.Conv2d, nn.Linear],  # Only track specific layer types
            tensor_stats=["mean", "norm", "hist"],  # Custom statistics
        )

        for epoch in range(1, 4):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)

                loss.backward()
                optimizer.step()

                # Log training metrics
                run.log_metrics({f"{neptune_callback.base_namespace}/batch/loss": loss.item()})

                # Log model internals (activations, gradients, parameters)
                neptune_callback.log_model_internals(
                    step=batch_idx,
                    prefix="train",
                    track_activations=True,
                    track_gradients=True,
                    track_parameters=True
                )

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/pytorch/
        API reference: https://docs.neptune.ai/api/integrations/pytorch/
    """

    def __init__(
        self,
        run: Run,
        *,
        model: torch.nn.Module,
        base_namespace: Optional[str] = None,
        log_model_diagram: bool = False,
        track_layers: Optional[List[Type[nn.Module]]] = None,
        tensor_stats: Optional[List[TensorStatType]] = None,
    ):
        if not isinstance(run, Run):
            raise ValueError("run must be a Neptune Run object")
        if not isinstance(model, torch.nn.Module):
            raise ValueError("model must be a PyTorch model")

        self.run = run
        self.model = model
        self._base_namespace = base_namespace
        self.log_model_diagram = log_model_diagram
        self.ckpt_number = 1

        self._is_diagram_saved = False
        self._diagram_hook_handler = None
        if log_model_diagram:
            summary_key = f"{self._base_namespace}/model/summary" if self._base_namespace else "model/summary"
            self.run.log_configs({summary_key: str(model)})
            self._add_diagram_hook()

        # Initialize TorchWatcher for model internals tracking
        self._torch_watcher = _TorchWatcher(
            model=model,
            run=run,
            base_namespace=base_namespace or "",
            track_layers=track_layers,
            tensor_stats=tensor_stats,
        )

        # Log integration version
        self.run.log_configs({_INTEGRATION_VERSION_KEY: __version__})

    def _add_diagram_hook(self):
        if not _IS_TORCHVIZ_AVAILABLE:
            msg = "Skipping model diagram because no torchviz installation was found."
            warnings.warn(msg)
            return

        def hook(module, input, output):
            if not self._is_diagram_saved:
                dot = torchviz.make_dot(output, params=dict(module.named_parameters()))
                dot.format = "png"
                # generate unique name so that multiple concurrent runs
                # don't over-write each other.
                viz_name = f"{str(uuid.uuid4())}.png"
                try:
                    dot.render(outfile=viz_name, cleanup=True)
                    _safe_upload_diagram(self.run, f"{self._base_namespace}/model/diagram", viz_name)
                except ExecutableNotFound:
                    # This errors because `dot` renderer is not found even
                    # if python binding of `graphviz` are available.
                    warnings.warn("Skipping model diagram because no dot (graphviz) installation was found.")

                self._is_diagram_saved = True

        self._diagram_hook_handler = self.model.register_forward_hook(hook)

    @property
    def base_namespace(self):
        return self._base_namespace

    def log_model_internals(
        self,
        step: int,
        track_activations: bool = True,
        track_gradients: bool = True,
        track_parameters: bool = False,
        prefix: Optional[str] = None,
    ):
        """
        Log model internals using TorchWatcher.

        Args:
            step: Logging step
            track_activations: Whether to track activations. Defaults to True.
            track_gradients: Whether to track gradients. Defaults to True.
            track_parameters: Whether to track parameters. Defaults to False.
            prefix: Optional prefix for phase organization (e.g., "train", "validation").
                If provided, internal metrics will be logged under {base_namespace}/model/internals/{prefix}/...
        """
        self._torch_watcher.watch(
            step=step,
            track_activations=track_activations,
            track_gradients=track_gradients,
            track_parameters=track_parameters,
            prefix=prefix,
        )

    def __del__(self):
        # Remove hooks
        if self._diagram_hook_handler is not None:
            self._diagram_hook_handler.remove()

        # Clean up TorchWatcher
        self._torch_watcher.hm.remove_hooks()


def _safe_upload_diagram(run: Run, name: str, file_name: str):
    # Function to safely upload a file and
    # delete the file on completion of upload.
    try:
        # Upload the file
        run.assign_files({f"{name}": file_name})
        # Wait for the upload to complete
        run.wait_for_processing()
    finally:
        # Clean up the files after upload is complete
        if os.path.exists(file_name):
            os.remove(file_name)
        # Also remove graphviz intermediate file.
        gv_file = file_name.replace(".png", ".gv")
        if os.path.exists(gv_file):
            os.remove(gv_file)
