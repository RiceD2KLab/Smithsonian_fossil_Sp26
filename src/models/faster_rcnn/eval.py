import torch
from torch.utils.data import Dataset, DataLoader
from src.modeling.faster_rcnn.model_factory import fasterrcnn_resnet101_fpn_v2
from train import _collate_fn
from engine import eval_one_epoch


def evaluate(
        model_file,
        test_dataset: Dataset,
        device: torch.device,
        batch_size=4,
        num_workers=2,
        prefetch_factor=1,
) -> dict:
    """
    Compute the mAP results for a stored model over a given test dataset.

    :param model_file: The file path to a stored pytorch model
    :param test_dataset: The test dataset
    :param device: The target device
    :param batch_size: How many image, target pairs per batch.
    :param num_workers: The number of subprocesses that should fetch data during data loading.
    :param prefetch_factor: How many batches should be prefetched per worker during data loading.
    :return:
    """

    model = fasterrcnn_resnet101_fpn_v2()
    model.load_state_dict(torch.load(model_file))

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        prefetch_factor=prefetch_factor,
    )

    map_results = eval_one_epoch(model, test_dataloader, device)
    return map_results