import os.path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model_factory import fasterrcnn_resnet101_fpn_v2
from engine import (
    train_one_epoch,
    eval_one_epoch
)


def _collate_fn(batch):
    """
    A custom collate function to account for images having varying number of target bounding boxes

    :param batch: The batch of images and targets
    :return:
    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    return images, targets


def train(
        model_output_dir,
        epochs: int,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        device: torch.device,
        batch_size=4,
        num_workers=2,
        prefetch_factor=1,
        patience=5,
) -> nn.Module:
    """
    Run the training for a Faster-RCNN model.

    :param model_output_dir: The directory that the best model should be saved to
    :param epochs: The maximum number of epochs to train
    :param train_dataset: The training dataset
    :param validation_dataset: The validation dataset. This is used for early stopping based on mAP computations.
    :param device: The target device.
    :param batch_size: How many image, target pairs per batch.
    :param num_workers: The number of subprocesses that should fetch data during data loading.
    :param prefetch_factor: How many batches should be prefetched per worker during data loading.
    :param patience: How many epochs of no improvement before early training termination.
    :return: The model after training completes.
    """
    model_output_dir = os.path.abspath(model_output_dir)

    model = fasterrcnn_resnet101_fpn_v2()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0005)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True if device == 'cuda' else False,
        prefetch_factor=prefetch_factor,
    )
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        prefetch_factor=prefetch_factor,
    )

    best_map: float = 0.0
    epochs_without_improvement: int = 0
    for i in range(epochs):
        print(f"\n--- EPOCH {i} ---")
        avg_loss = train_one_epoch(model, train_dataloader, optimizer, device)
        print(f"Average training loss: {avg_loss}:.4f")

        map_results = eval_one_epoch(model, validation_dataloader, device)
        map_50 = map_results["map_50"].item()
        print(f"Validation mAP: {map_50:.4f}")

        if map_50 > best_map:
            best_map = map_50
            epochs_without_improvement = 0

            best_path = os.path.join(model_output_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved with mAP: {best_map:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"Failed to improve the mAP this epoch")

        if epochs_without_improvement >= patience:
            print(
                f"No mAP improvements after {epochs_without_improvement} epoch(s). Triggering early stop. Training complete.")
            break

    return model
