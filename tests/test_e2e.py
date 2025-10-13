from __future__ import print_function

from unittest.mock import Mock

import torch
import torch.nn.functional as F
import torch.optim as optim
from neptune_scale import Run

from neptune_pytorch import NeptuneLogger


def test_e2e(model, dataset):
    # Training settings
    device = torch.device("cpu")
    train_kwargs = {"batch_size": 8}

    dataset = torch.utils.data.TensorDataset(*dataset)
    train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)

    model = model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.001)

    # Mock the Neptune run object
    run = Mock(spec=Run)
    run.log_metrics = Mock()
    run.log_histograms = Mock()
    run.log_configs = Mock()
    run.assign_files = Mock()
    run.wait_for_processing = Mock()

    npt_logger = NeptuneLogger(
        run,
        model=model,
        base_namespace="test_experiment",
        log_model_diagram=True,
    )

    for _ in range(1, 4):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)

            loss.backward()
            optimizer.step()

            run.log_metrics({f"{npt_logger.base_namespace}/batch/loss": loss.item()})
            npt_logger.log_model_internals(
                step=0,
                prefix="train",
                track_activations=True,
                track_gradients=True,
                track_parameters=True,
            )

    # Verify that the correct logging methods were called
    # Check that log_metrics was called for batch loss
    assert run.log_metrics.call_count >= 1

    # Check that log_configs was called for model summary and integration version
    assert run.log_configs.call_count >= 2

    # Verify model summary was logged
    log_configs_calls = run.log_configs.call_args_list
    model_summary_calls = [
        call for call in log_configs_calls if f"{npt_logger.base_namespace}/model/summary" in str(call)
    ]
    assert model_summary_calls, "Model summary should be logged"

    # Verify model summary content and namespace
    for call in model_summary_calls:
        call_kwargs = call.kwargs
        if "data" in call_kwargs:
            summary_data = call_kwargs["data"]
            expected_namespace = f"{npt_logger.base_namespace}/model/summary"
            assert expected_namespace in summary_data, f"Model summary should be logged under {expected_namespace}"
            # The summary should contain the model's string representation
            assert isinstance(summary_data[expected_namespace], str), "Model summary should be a string"

    # Verify diagram behavior
    assign_files_calls = run.assign_files.call_args_list
    diagram_calls = [call for call in assign_files_calls if "diagram" in str(call)]

    assert run.assign_files.call_count >= 1, "Model diagram should be uploaded"
    assert run.wait_for_processing.call_count >= 1, "Should wait for upload to complete"
    assert diagram_calls, "Model diagram should be uploaded with correct namespace"

    # Verify that model internals logging was called
    # The TorchWatcher should have logged metrics for activations, gradients, and parameters
    log_metrics_calls = run.log_metrics.call_args_list

    # Check that we have calls for both batch loss and model internals
    batch_loss_calls = [call for call in log_metrics_calls if f"{npt_logger.base_namespace}/batch/loss" in str(call)]
    model_internals_calls = [call for call in log_metrics_calls if "model/internals" in str(call)]

    assert batch_loss_calls, "Batch loss should be logged"
    assert model_internals_calls, "Model internals should be logged"

    # Verify that the model internals calls contain the expected namespace structure
    for call in model_internals_calls:
        call_kwargs = call.kwargs
        if "data" in call_kwargs:
            metrics = call_kwargs["data"]
            for metric_name in metrics.keys():
                assert metric_name.startswith(
                    f"{npt_logger.base_namespace}/model/internals/"
                ), f"Metric {metric_name} should start with correct namespace"
                assert any(
                    metric_type in metric_name for metric_type in ["activations", "gradients", "parameters"]
                ), f"Metric {metric_name} should contain one of: activations, gradients, parameters"
