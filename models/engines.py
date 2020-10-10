# definition of the engines

from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from ignite.engine import Engine
from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError


def create_supervised_trainer(
    model: nn.Module, optimizer: optim.Optimizer, criterion, device, dtype):
    """
    Factory function for creating a trainer for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device: torch device fpr the target and the batch tensors.
        dtype: torch dtype for the input and the output tensors.

    Note: `engine.state.output` for this engine is the current loss of the model.

    Returns:
        Engine: a trainer engine with supervised update function.
    """
    model.to(device=device, dtype=dtype)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        # operations of tensors
        inputs, targets = batch
        inputs = inputs.to(dtype=dtype, device=device)
        targets = targets.to(dtype=dtype, device=device)
        # optimisation step
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        return loss.item()

    return Engine(_update)


def create_supervised_evaluator(
    model: nn.Module, metrics: Dict, device, dtype,
    output_transform=lambda x, y, y_pred: (y_pred, y,)):
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.
        device: torch device fpr the target and the batch tensors.
        dtype: torch dtype for the input and the output tensors.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred'
            and returns value to be assigned to engine's state.output after each iteration.
            Default is returning `(y_pred, y,)` which fits output expected by metrics.
            If you change it you should use `output_transform` in metrics.

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """
    model.to(device=device, dtype=dtype)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch
            x = x.to(dtype=dtype, device=device)
            y = y.to(dtype=dtype, device=device)
            y_pred = model(x)
            return output_transform(x, y, y_pred)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


class MeanSquaredError(Metric):
    """
    Calculates the mean squared error.

    - `update` must receive output of the form `(y_pred, y)`.
    """
    def reset(self):
        self._sum_of_squared_errors = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        squared_errors = torch.pow(y_pred - y.view_as(y_pred), 2)
        self._sum_of_squared_errors += torch.sum(squared_errors, 0)
        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'MeanSquaredError must have at least one example before it can be computed.'
            )
        return torch.mean(torch.div(self._sum_of_squared_errors, self._num_examples))
