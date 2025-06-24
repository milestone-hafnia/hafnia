from typing import Any, Dict

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2

from cli.config import Config
from hafnia import torch_helpers
from hafnia.data import load_dataset
from hafnia.dataset.dataset_names import ColumnName
from hafnia.dataset.hafnia_dataset import HafniaDataset, Sample, check_hafnia_dataset
from hafnia.dataset.primitives.bbox import Bbox
from hafnia.dataset.primitives.bitmask import Bitmask
from hafnia.dataset.primitives.classification import Classification
from hafnia.dataset.primitives.polygon import Polygon
from hafnia.dataset.primitives.segmentation import Segmentation

FORCE_REDOWNLOAD = False

DATASETS_EXPECTED = [
    ("midwest-vehicle-detection", {"train": 172, "validation": 21, "test": 21}),
    ("tiny-dataset", {"train": 3, "validation": 2, "test": 3}),
    ("mnist", {"train": 176, "test": 18, "validation": 6}),
    ("caltech-101", {"train": 156, "validation": 24, "test": 20}),
    ("caltech-256", {"train": 163, "validation": 17, "test": 20}),
    ("cifar10", {"train": 171, "validation": 4, "test": 25}),
    ("cifar100", {"train": 428, "validation": 13, "test": 59}),
    # ("easyportrait", {"train": 32, "test": 20, "validation": 10}),
    ("coco-2017", {"train": 192, "validation": 4, "test": 4}),
    # ("sama-coco", {"train": 99, "validation": 1, "test": 1}),
    # ("open-images-v7", {"train": 91, "validation": 3, "test": 9}),
]
DATASET_IDS = [dataset[0] for dataset in DATASETS_EXPECTED]


@pytest.fixture(params=DATASETS_EXPECTED, ids=DATASET_IDS, scope="session")
def loaded_dataset(request) -> Dict[str, Any]:
    """Fixture that loads a dataset and returns it along with metadata."""
    if not Config().is_configured():
        pytest.skip("Not logged in to Hafnia")

    dataset_name, expected_lengths = request.param
    dataset = load_dataset(dataset_name, force_redownload=FORCE_REDOWNLOAD)

    return {
        "dataset": dataset,
        "dataset_name": dataset_name,
        "expected_lengths": expected_lengths,
    }


def hafnia_2_torch_dataset(dataset: HafniaDataset) -> torch.utils.data.Dataset:
    # Define transforms
    transforms = v2.Compose(
        [
            v2.Resize(size=(224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    # Create Torchvision dataset
    dataset_torch = torch_helpers.TorchvisionDataset(
        dataset,
        transforms=transforms,
        keep_metadata=True,
    )

    return dataset_torch


@pytest.mark.slow
def test_dataset_lengths(loaded_dataset):
    """Test that the dataset has the expected number of samples."""
    dataset = loaded_dataset["dataset"]
    expected_split_counts = loaded_dataset["expected_lengths"]

    actual_split_counts = dict(dataset.samples[ColumnName.SPLIT].value_counts().iter_rows())
    assert actual_split_counts == expected_split_counts


@pytest.mark.slow
def test_check_dataset(loaded_dataset, compare_to_expected_image):
    """Test the features of the dataset based on task type."""
    dataset = loaded_dataset["dataset"]
    check_hafnia_dataset(dataset)

    sample_dict = dataset[0]
    sample = Sample(**sample_dict)

    image = sample.draw_annotations()

    compare_to_expected_image(image)


@pytest.mark.slow
def test_dataset_draw_image_and_target(loaded_dataset, compare_to_expected_image):
    """Test data transformations and visualization."""
    dataset = loaded_dataset["dataset"]
    torch_dataset = hafnia_2_torch_dataset(dataset.create_split_dataset("train"))

    # Test single item transformation
    image, targets = torch_dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape[0] in (3, 1)  # RGB or grayscale
    assert image.shape[1:] == (224, 224)  # Resized dimensions

    # Test visualization
    visualized = torch_helpers.draw_image_and_targets(image=image, targets=targets)
    assert isinstance(visualized, torch.Tensor)

    pil_image = v2.functional.to_pil_image(visualized)
    compare_to_expected_image(np.array(pil_image))


@pytest.mark.slow
def test_dataset_dataloader(loaded_dataset):
    """Test dataloader functionality."""
    dataset = loaded_dataset["dataset"]
    torch_dataset = hafnia_2_torch_dataset(dataset.create_split_dataset("train"))

    # Test dataloader with custom collate function
    batch_size = 2
    collate_fn = torch_helpers.TorchVisionCollateFn()
    dataloader_train = DataLoader(batch_size=batch_size, dataset=torch_dataset, collate_fn=collate_fn)

    # Test iteration
    for images, targets in dataloader_train:
        break  # Break immediately to get the first batch

    assert isinstance(images, torch.Tensor)
    assert images.shape[0] == batch_size
    assert images.shape[2:] == (224, 224)

    for task in dataset.info.tasks:
        task_name = f"{task.primitive.column_name()}.{task.name}"
        class_idx_name = f"{task_name}.class_idx"
        assert class_idx_name in targets
        class_idx = targets[class_idx_name]

        class_names_name = f"{task_name}.class_name"
        assert class_names_name in targets
        class_names = targets[class_names_name]
        assert isinstance(class_names, list)

        if task.primitive == Classification:
            assert isinstance(class_idx, torch.Tensor)
        elif task.primitive == Bbox:
            assert isinstance(class_idx, list)
            bboxes_name = f"{task_name}.bbox"
            assert class_names_name in targets, f"Expected {class_names_name} in targets"
            bboxes = targets[bboxes_name]
            assert isinstance(bboxes, list)
            if len(bboxes) > 0:
                assert isinstance(bboxes[0], tv_tensors.BoundingBoxes)
        elif task.primitive == Bitmask:
            assert isinstance(class_idx, list)
            bitmasks_name = f"{task_name}.mask"
            assert bitmasks_name in targets, f"Expected {bitmasks_name} in targets"
            bitmasks = targets[bitmasks_name]
            assert isinstance(bitmasks, list)
            if len(bitmasks) > 0:
                assert isinstance(bitmasks[0], tv_tensors.Mask)
        elif task.primitive == Polygon:
            raise NotImplementedError("Polygon handling not implemented in this test")
        elif task.primitive == Segmentation:
            raise NotImplementedError("Segmentation handling not implemented in this test")
        else:
            raise ValueError(f"Unsupported task primitive: {task.primitive}")
