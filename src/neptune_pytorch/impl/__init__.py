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
import weakref
from typing import (
    Optional,
    Union,
)

import torch

from neptune_pytorch.impl.version import __version__

try:
    # neptune-client>=1.0.0 package structure
    from neptune import Run
    from neptune.handler import Handler
    from neptune.internal.utils import verify_type
    from neptune.types import File
except ImportError:
    from neptune.new import Run
    from neptune.new.handler import Handler
    from neptune.new.integrations.utils import verify_type
    from neptune.new.types import File

IS_TORCHVIZ_AVAILABLE = True
try:
    import torchviz
    from graphviz import ExecutableNotFound
except ImportError:
    IS_TORCHVIZ_AVAILABLE = False

INTEGRATION_VERSION_KEY = "source_code/integrations/neptune-pytorch"


class NeptuneLogger:
    """Captures model training metadata and logs them to Neptune.

    Args:
        run: Neptune run object. You can also pass a namespace handler object;
            for example, run["test"], in which case all metadata is logged under
            the "test" namespace inside the run.
        model: PyTorch model whose metadata will be tracked.
        base_namespace: Namespace where all metadata logged by the callback is stored.
        log_gradients: Whether to track the frobenius-order norm of the gradients.
        log_parameters: Whether to track the frobenius-order norm of the parameters.
        log_freq: How often to log the parameters/gradients norm. Applicable only
            if `log_parameters` or `log_gradients` is True.
        log_model_diagram: Whether to save the model visualization.
            Requires torchviz to be installed: https://pypi.org/project/torchviz/

    Example:
        import neptune
        from neptune.integrations.pytorch import NeptuneLogger
        run = neptune.init_run()
        neptune_callback = NeptuneLogger(run=run, model=model)

        for epoch in range(1, 4):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)

                loss.backward()
                optimizer.step()

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/pytorch/
        API reference: https://docs.neptune.ai/api/integrations/pytorch/
    """

    def __init__(
        self,
        run: Union[Run, Handler],
        *,
        model: torch.nn.Module,
        base_namespace="training",
        log_model_diagram: bool = False,
        log_gradients: bool = False,
        log_parameters: bool = False,
        log_freq: int = 100,
    ):
        verify_type("run", run, (Run, Handler))

        self.run = run
        self.model = model
        self._base_namespace = base_namespace
        self.log_model_diagram = log_model_diagram
        self.ckpt_number = 1
        self.log_freq = log_freq
        self._namespace_handler = self.run[base_namespace]

        self._is_viz_saved = False
        self._vis_hook_handler = None
        if log_model_diagram:
            self.run[self._base_namespace]["model"]["summary"] = str(model)
            self._add_visualization_hook()

        self.log_gradients = log_gradients
        self._gradients_iter_tracker = {}
        self._gradients_hook_handler = {}
        if self.log_gradients:
            self._add_hooks_for_grads()

        self.log_parameters = log_parameters
        self._params_iter_tracker = 0
        self._params_hook_handler = None
        if self.log_parameters:
            self._add_hooks_for_params()

        # Log integration version
        root_obj = self.run
        if isinstance(self.run, Handler):
            root_obj = self.run.get_root_obj()

        root_obj[INTEGRATION_VERSION_KEY] = __version__

    def _add_hooks_for_grads(self):
        for name, parameter in self.model.named_parameters():
            self._gradients_iter_tracker[name] = 0

            def hook(grad, name=name):
                self._gradients_iter_tracker[name] += 1
                if self._gradients_iter_tracker[name] % self.log_freq == 0:
                    self._namespace_handler["plots"]["gradients"][name].append(grad.norm())

            self._gradients_hook_handler[name] = parameter.register_hook(hook)

    def _add_visualization_hook(self):
        if not IS_TORCHVIZ_AVAILABLE:
            msg = "Skipping model visualization because no torchviz installation was found."
            warnings.warn(msg)
            return

        def hook(module, input, output):
            if not self._is_viz_saved:
                dot = torchviz.make_dot(output, params=dict(module.named_parameters()))
                dot.format = "png"
                # generate unique name so that multiple concurrent runs
                # don't over-write each other.
                viz_name = str(uuid.uuid4()) + ".png"
                try:
                    dot.render(outfile=viz_name)
                    safe_upload_visualization(self._namespace_handler["model"], "visualization", viz_name)
                except ExecutableNotFound:
                    # This errors because `dot` renderer is not found even
                    # if python binding of `graphviz` are available.
                    warnings.warn("Skipping model visualization because no dot (graphviz) installation was found.")

                self._is_viz_saved = True

        self._vis_hook_handler = self.model.register_forward_hook(hook)

    def _add_hooks_for_params(self):
        def hook(module, inp, output):
            self._params_iter_tracker += 1
            if self._params_iter_tracker % self.log_freq == 0:
                for name, param in module.named_parameters():
                    self._namespace_handler["plots"]["parameters"][name].append(param.norm())

        self._params_hook_handler = self.model.register_forward_hook(hook)

    @property
    def base_namespace(self):
        return self._base_namespace

    def log_model(self, model_name: Optional[str] = None):
        """Uploads the model to Neptune.

        The model is saved in a namespace called "model" nested under the base namespace of the run, which is
        "training" by default.

        The filename is set to `model.pt` by default, but can be customized.

        Args:
            model_name: Name for the logged model file. The extension `.pt` is added automatically.

        Example:
            from neptune_pytorch import NeptuneLogger
            neptune_logger = NeptuneLogger()
            ...
            neptune_logger.log_model()

        For more, see the docs:
            Tutorial: https://docs.neptune.ai/integrations/pytorch/
            API reference: https://docs.neptune.ai/api/integrations/pytorch/
        """
        if model_name is None:
            # Default model name
            model_name = "model.pt"
        else:
            # User is not expected to add extension
            model_name = model_name + ".pt"

        safe_upload_model(self._namespace_handler["model"], model_name, self.model)

    def log_checkpoint(self, checkpoint_name: Optional[str] = None):
        """Uploads a model checkpoint to Neptune.

        The checkpoint is saved in a namespace called "model/checkpoints" nested under the base namespace of the run,
        which is "training" by default.

        The filename is set to `checkpoint_<checkpoint number>.pt` by default, but can be customized.

        Args:
            checkpoint_name: Name for the logged checkpoint file.
                If left empty, the checkpoint number used for the file name starts from 1 and is incremented
                automatically on each call. The extension `.pt` is added automatically.

        Example:
            from neptune_pytorch import NeptuneLogger
            neptune_logger = NeptuneLogger()
            ...
            for epoch in range(parameters["epochs"]):
                ...
                neptune_logger.log_checkpoint()

        For more, see the docs:
            Tutorial: https://docs.neptune.ai/integrations/pytorch/
            API reference: https://docs.neptune.ai/api/integrations/pytorch/
        """
        if checkpoint_name is None:
            # Default checkpoint name
            checkpoint_name = f"checkpoint_{self.ckpt_number}.pt"
            self.ckpt_number += 1
        else:
            # User is not expected to add extension
            checkpoint_name = checkpoint_name + ".pt"

        safe_upload_model(self._namespace_handler["model"]["checkpoints"], checkpoint_name, self.model)

    def __del__(self):
        # Remove hooks
        if self._params_hook_handler is not None:
            self._params_hook_handler.remove()

        for name, handler in self._gradients_hook_handler.items():
            if handler is not None:
                handler.remove()

        if self._vis_hook_handler is not None:
            self._vis_hook_handler.remove()


def safe_upload_visualization(run: Run, name: str, file_name: str):
    # Function to safely upload a file and
    # delete the file on completion of upload.
    # We utilise the weakref.finalize to remove
    # the file once the stream object goes out-of-scope.

    def remove(file_name):
        os.remove(file_name)
        # Also remove graphviz intermediate file.
        os.remove(file_name.replace(".png", ".gv"))

    with open(file_name, "rb") as f:
        weakref.finalize(f, remove, file_name)
        run[name].upload(File.from_stream(f, extension="png"))


def safe_upload_model(run: Run, name: str, model: torch.nn.Module):
    # Function to safely upload a file and
    # delete the file on completion of upload.
    # We utilise the weakref.finalize to remove
    # the file once the stream object goes out-of-scope.

    torch.save(model.state_dict(), name)

    def remove(file_name):
        os.remove(file_name)

    with open(name, "rb") as f:
        weakref.finalize(f, remove, name)
        run[name].upload(File.from_stream(f))
