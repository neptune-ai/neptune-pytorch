"""
Comprehensive test suite for _TorchWatcher functionality.
"""

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from neptune_scale import Run

from neptune_pytorch.impl._torchwatcher import (
    TENSOR_STATS,
    _HookManager,
    _TorchWatcher,
)


class TestNet(nn.Module):
    """Simple test network for testing TorchWatcher."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


@pytest.fixture
def mock_run():
    """Mock Neptune run object."""
    run = Mock(spec=Run)
    run.log_metrics = Mock()
    run.log_histograms = Mock()
    run.log_configs = Mock()
    run.assign_files = Mock()
    run.wait_for_processing = Mock()
    return run


@pytest.fixture
def test_model():
    """Test PyTorch model."""
    return TestNet()


@pytest.fixture
def test_data():
    """Test input data."""
    return torch.rand(2, 1, 28, 28)


class TestHookManager:
    """Test _HookManager functionality."""

    def test_hook_manager_initialization(self, test_model):
        """Test HookManager initialization with valid model."""
        hm = _HookManager(test_model)
        assert hm.model == test_model
        assert hm.track_layers is None
        assert len(hm.hooks) == 0
        assert len(hm.activations) == 0
        assert len(hm.gradients) == 0

    def test_hook_manager_with_specific_layers(self, test_model):
        """Test HookManager with specific layer types."""
        track_layers = [nn.Conv2d, nn.Linear]
        hm = _HookManager(test_model, track_layers)
        assert hm.track_layers == track_layers

    def test_hook_manager_invalid_model(self):
        """Test HookManager with invalid model type."""
        with pytest.raises(TypeError, match="The model must be a PyTorch model"):
            _HookManager("not_a_model")

    def test_hook_manager_invalid_layer_types(self, test_model):
        """Test HookManager with invalid layer types."""
        with pytest.raises(ValueError, match="Invalid layer type"):
            _HookManager(test_model, [str, int])

    def test_register_hooks(self, test_model):
        """Test hook registration."""
        hm = _HookManager(test_model)
        hm.register_hooks(track_activations=True, track_gradients=True)

        # Should have hooks for all layers except the model itself
        expected_hooks = len([name for name, _ in test_model.named_modules() if name != ""])
        assert len(hm.hooks) == expected_hooks * 2  # activations + gradients

    def test_register_hooks_activations_only(self, test_model):
        """Test hook registration for activations only."""
        hm = _HookManager(test_model)
        hm.register_hooks(track_activations=True, track_gradients=False)

        expected_hooks = len([name for name, _ in test_model.named_modules() if name != ""])
        assert len(hm.hooks) == expected_hooks

    def test_register_hooks_gradients_only(self, test_model):
        """Test hook registration for gradients only."""
        hm = _HookManager(test_model)
        hm.register_hooks(track_activations=False, track_gradients=True)

        expected_hooks = len([name for name, _ in test_model.named_modules() if name != ""])
        assert len(hm.hooks) == expected_hooks

    def test_remove_hooks(self, test_model):
        """Test hook removal."""
        hm = _HookManager(test_model)
        hm.register_hooks()

        hm.remove_hooks()
        assert len(hm.hooks) == 0

    def test_clear_data(self, test_model):
        """Test clearing stored data."""
        hm = _HookManager(test_model)
        hm.activations["test"] = torch.tensor([1.0])
        hm.gradients["test"] = torch.tensor([2.0])

        hm.clear()
        assert len(hm.activations) == 0
        assert len(hm.gradients) == 0

    def test_hook_manager_with_specific_layers_filtering(self, test_model):
        """Test that specific layer filtering works correctly."""
        track_layers = [nn.Conv2d]
        hm = _HookManager(test_model, track_layers)
        hm.register_hooks(track_activations=True, track_gradients=True)

        # Should only have hooks for Conv2d layers
        conv_layers = [
            name for name, module in test_model.named_modules() if isinstance(module, nn.Conv2d) and name != ""
        ]
        expected_hooks = len(conv_layers) * 2  # activations + gradients
        assert len(hm.hooks) == expected_hooks


class TestTorchWatcher:
    """Test _TorchWatcher functionality."""

    def test_torch_watcher_initialization(self, mock_run, test_model):
        """Test TorchWatcher initialization with valid parameters."""
        tw = _TorchWatcher(
            model=test_model,
            run=mock_run,
            base_namespace="test",
            track_layers=None,
            tensor_stats=["mean", "norm"],
        )

        assert tw.model == test_model
        assert tw.run == mock_run
        assert tw.base_namespace == "test"
        assert "mean" in tw.tensor_stats
        assert "norm" in tw.tensor_stats

    def test_torch_watcher_default_parameters(self, mock_run, test_model):
        """Test TorchWatcher with default parameters."""
        tw = _TorchWatcher(model=test_model, run=mock_run, base_namespace="test")

        assert tw.track_layers is None
        assert tw.tensor_stats == {stat: TENSOR_STATS[stat] for stat in ["mean", "norm", "hist"]}

    def test_torch_watcher_invalid_model(self, mock_run):
        """Test TorchWatcher with invalid model type."""
        with pytest.raises(TypeError, match="The model must be a PyTorch model"):
            _TorchWatcher(model="not_a_model", run=mock_run, base_namespace="test")

    def test_torch_watcher_invalid_tensor_stats(self, mock_run, test_model):
        """Test TorchWatcher with invalid tensor statistics."""
        with pytest.raises(ValueError, match="Invalid statistics requested"):
            _TorchWatcher(
                model=test_model,
                run=mock_run,
                base_namespace="test",
                tensor_stats=["invalid_stat", "mean"],
            )

    def test_track_activations(self, mock_run, test_model, test_data):
        """Test activation tracking."""
        tw = _TorchWatcher(model=test_model, run=mock_run, base_namespace="test", tensor_stats=["mean", "norm"])

        # Forward pass to generate activations
        _ = test_model(test_data)

        metrics = tw.track_activations()

        # Check that activations were tracked
        assert len(metrics) > 0
        for key in metrics:
            assert key.startswith("test/model/internals/activations/")
            assert key.endswith(("/mean", "/norm"))

    def test_track_gradients(self, mock_run, test_model, test_data):
        """Test gradient tracking."""
        tw = _TorchWatcher(model=test_model, run=mock_run, base_namespace="test", tensor_stats=["mean", "norm"])

        # Forward and backward pass to generate gradients
        output = test_model(test_data)
        loss = F.nll_loss(output, torch.randint(0, 10, (2,)))
        loss.backward()

        metrics = tw.track_gradients()

        # Check that gradients were tracked
        assert len(metrics) > 0
        for key in metrics:
            assert key.startswith("test/model/internals/gradients/")
            assert key.endswith(("/mean", "/norm"))

    def test_track_parameters(self, mock_run, test_model):
        """Test parameter tracking."""
        tw = _TorchWatcher(
            model=test_model,
            run=mock_run,
            base_namespace="test",
            tensor_stats=["mean", "norm"],
        )

        metrics = tw.track_parameters()

        # Check that parameters were tracked
        assert len(metrics) > 0
        for key in metrics:
            assert key.startswith("test/model/internals/parameters/")
            assert key.endswith(("/mean", "/norm"))

    def test_track_parameters_always_logs(self, mock_run, test_model):
        """Test parameter tracking always logs when called."""
        tw = _TorchWatcher(
            model=test_model,
            run=mock_run,
            base_namespace="test",
            tensor_stats=["mean"],
        )

        # Should log every time
        metrics = tw.track_parameters()
        assert len(metrics) > 0

    def test_watch_method(self, mock_run, test_model, test_data):
        """Test the main watch method."""
        tw = _TorchWatcher(model=test_model, run=mock_run, base_namespace="test", tensor_stats=["mean", "norm"])

        # Forward and backward pass
        output = test_model(test_data)
        loss = F.nll_loss(output, torch.randint(0, 10, (2,)))
        loss.backward()

        tw.watch(step=0, track_activations=True, track_gradients=True, track_parameters=False)

        # Check that metrics were logged
        mock_run.log_metrics.assert_called_once()
        call_args = mock_run.log_metrics.call_args
        assert "step" in call_args.kwargs
        assert call_args.kwargs["step"] == 0

        # Verify that the logged data contains the expected metrics
        logged_data = call_args.kwargs["data"]
        assert len(logged_data) > 0, "Should have logged some metrics"

        # Check that all logged metrics have the correct namespace structure
        for metric_name in logged_data.keys():
            assert metric_name.startswith(
                "test/model/internals/"
            ), f"Metric {metric_name} should start with correct namespace"
            assert any(
                metric_type in metric_name for metric_type in ["activations", "gradients"]
            ), f"Metric {metric_name} should contain activations or gradients"
            assert metric_name.endswith(("/mean", "/norm")), f"Metric {metric_name} should end with /mean or /norm"

    def test_watch_method_with_prefix(self, mock_run, test_model, test_data):
        """Test watch method with prefix."""
        tw = _TorchWatcher(model=test_model, run=mock_run, base_namespace="test", tensor_stats=["mean"])

        # Forward pass
        _ = test_model(test_data)

        tw.watch(step=0, track_activations=True, prefix="train")

        # Check that metrics were logged with prefix
        mock_run.log_metrics.assert_called_once()
        call_args = mock_run.log_metrics.call_args
        metrics = call_args.kwargs["data"]

        # All metrics should have the prefix
        for key in metrics:
            assert key.startswith("test/model/internals/train/")

    def test_histogram_processing(self, mock_run, test_model, test_data):
        """Test histogram processing in watch method."""
        tw = _TorchWatcher(model=test_model, run=mock_run, base_namespace="test", tensor_stats=["hist"])

        # Forward pass
        _ = test_model(test_data)

        tw.watch(step=0, track_activations=True)

        # Should call both log_metrics and log_histograms
        mock_run.log_metrics.assert_called_once()
        mock_run.log_histograms.assert_called_once()

        # Verify log_histograms was called with correct parameters
        hist_call_args = mock_run.log_histograms.call_args
        assert "step" in hist_call_args.kwargs
        assert hist_call_args.kwargs["step"] == 0

        # Verify that histograms contain the expected structure
        histograms = hist_call_args.kwargs["histograms"]
        assert len(histograms) > 0, "Should have logged some histograms"

        for hist_name, hist_data in histograms.items():
            assert hist_name.startswith(
                "test/model/internals/activations/"
            ), f"Histogram {hist_name} should start with correct namespace"
            assert hist_name.endswith("/hist"), f"Histogram {hist_name} should end with /hist"
            assert hasattr(hist_data, "bin_edges"), "Histogram should have bin_edges"
            assert hasattr(hist_data, "counts"), "Histogram should have counts"

    def test_hook_cleanup_on_destruction(self, mock_run, test_model):
        """Test that hooks are cleaned up when TorchWatcher is destroyed."""
        tw = _TorchWatcher(model=test_model, run=mock_run, base_namespace="test")
        assert len(tw.hm.hooks) > 0

        # Manually call remove_hooks to test cleanup
        tw.hm.remove_hooks()
        assert len(tw.hm.hooks) == 0

    def test_safe_tensor_stats(self, mock_run, test_model):
        """Test safe tensor statistics computation."""
        tw = _TorchWatcher(
            model=test_model,
            run=mock_run,
            base_namespace="test",
            tensor_stats=["mean", "hist", "norm"],
        )

        test_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        stats = tw._safe_tensor_stats(test_tensor)

        assert "mean" in stats
        assert "hist" in stats
        assert "norm" in stats
        assert isinstance(stats["mean"], float)
        assert isinstance(stats["hist"], torch.return_types.histogram)
        assert isinstance(stats["norm"], float)

    def test_track_metric_namespace_construction(self, mock_run, test_model):
        """Test proper namespace construction in _track_metric."""
        tw = _TorchWatcher(model=test_model, run=mock_run, base_namespace="test", tensor_stats=["mean"])

        # Test without prefix
        test_data = {"layer1": torch.tensor([1.0, 2.0])}
        metrics = tw._track_metric("activations", test_data)

        expected_key = "test/model/internals/activations/layer1/mean"
        assert expected_key in metrics

        # Test with prefix
        metrics = tw._track_metric("gradients", test_data, prefix="train")

        expected_key = "test/model/internals/train/gradients/layer1/mean"
        assert expected_key in metrics


class TestIntegration:
    """Integration tests for TorchWatcher with NeptuneLogger."""

    def test_neptune_logger_with_torch_watcher(self, mock_run, test_model, test_data):
        """Test NeptuneLogger integration with TorchWatcher."""
        from neptune_pytorch.impl import NeptuneLogger

        logger = NeptuneLogger(
            run=mock_run,
            model=test_model,
            base_namespace="integration_test",
            tensor_stats=["mean", "norm"],
        )

        # Forward and backward pass
        output = test_model(test_data)
        loss = F.nll_loss(output, torch.randint(0, 10, (2,)))
        loss.backward()

        # Log model internals
        logger.log_model_internals(
            step=0,
            prefix="test",
            track_activations=True,
            track_gradients=True,
            track_parameters=True,
        )

        # Check that TorchWatcher was used
        assert logger._torch_watcher is not None
        mock_run.log_metrics.assert_called()

    def test_neptune_logger_without_tracking(self, mock_run, test_model):
        """Test NeptuneLogger with tracking disabled via log_model_internals."""
        from neptune_pytorch.impl import NeptuneLogger

        logger = NeptuneLogger(
            run=mock_run,
            model=test_model,
        )

        # Should have TorchWatcher
        assert logger._torch_watcher is not None

        # log_model_internals with all tracking disabled
        logger.log_model_internals(step=0, track_activations=False, track_gradients=False, track_parameters=False)
        # Should still call log_metrics but with no data
        mock_run.log_metrics.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
