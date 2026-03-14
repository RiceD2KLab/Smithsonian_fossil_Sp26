from typing import Dict, List

import torch.optim
from functorch.dim import Tensor
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def train_one_epoch(
        model: nn.Module,
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device
) -> float:
    """
    Trains the model a single time across the entire dataset.

    :param model: A Faster-RCNN model.
    :param train_dataloader: Both the input tensors and targets (list of dictionaries) are required for training.
    "The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each image, and
    should be in ``0-1`` range. Different images can have different sizes." The targets should contain the following,
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box.
    :param optimizer: A gradient-based optimizer for training the model.
    :param device: The target device for training.
    :return: The average loss for the entire epoch.
    """
    model.train()
    epoch_loss: float = 0.0
    running_loss: float = 0.0
    i = 0
    for images, targets in train_dataloader:
        images: List[Tensor] = list(img.to(device) for img in images)
        targets: List[Dict[str, Tensor]] = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict: Dict[str, Tensor] = model(images, targets)
        losses: Tensor = torch.stack(list(loss_dict.values())).sum()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        epoch_loss += losses.item()
        if not (i + 1) % 100:
            print(f"Batch {i + 1}/{len(train_dataloader)}, Loss: {running_loss / 100:.4f}")
            running_loss = 0.0
        i += 1
    return epoch_loss / len(train_dataloader)


def eval_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device
) -> dict:
    """
    Computes the mAP metric across the entire dataset.

    :param model: A Faster-RCNN model.
    :param dataloader: Just the targets (list of dictionaries) are required for inference. The targets should contain
    the following,
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box.
    :param device: The target device for inference.
    :return: A dictionary of metrics. See the torchmetrics.detection.mean_ap.MeanAveragePrecision documentation.
    """
    model.eval()
    metric = MeanAveragePrecision()
    with torch.no_grad:
        for images, targets in dataloader:
            images: List[torch.Tensor] = list(img.to(device) for img in images)

            predictions: List[Dict[str, Tensor]] = model(images)

            predictions: List[Dict[str, Tensor]] = [{k: v.to('cpu') for k, v in p.items()} for p in predictions]
            targets: List[Dict[str, Tensor]] = [{k: v.to('cpu') for k, v in t.items()} for t in targets]

            metric.update(predictions, targets)

    results = metric.compute()
    return results
