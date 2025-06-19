import datasets
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from cli.config import Config
from hafnia import torch_helpers
from hafnia.data import load_dataset
from hafnia.dataset.dataset_names import ColumnName
from hafnia.dataset.hafnia_dataset import HafniaDataset, Sample, check_hafnia_dataset

FORCE_REDOWNLOAD = False

DATASETS_EXPECTED = [
    (
        "midwest-vehicle-detection",
        {"train": 172, "validation": 21, "test": 21},
        "ObjectDetection",
    ),
    ("mnist", {"train": 176, "test": 18, "validation": 6}, "ImageClassification"),
    ("caltech-101", {"train": 156, "validation": 24, "test": 20}, "ImageClassification"),
    ("caltech-256", {"train": 163, "validation": 17, "test": 20}, "ImageClassification"),
    ("cifar10", {"train": 171, "validation": 4, "test": 25}, "ImageClassification"),
    ("cifar100", {"train": 428, "validation": 13, "test": 59}, "ImageClassification"),
    # ("easyportrait", {"train": 32, "test": 20, "validation": 10}, "Segmentation"),
    ("coco-2017", {"train": 192, "validation": 8, "test": 8}, "ObjectDetection"),
    # ("sama-coco", {"train": 99, "validation": 1, "test": 1}, "ObjectDetection"),
    # ("open-images-v7", {"train": 91, "validation": 3, "test": 9}, "ObjectDetection"),
]
DATASET_IDS = [dataset[0] for dataset in DATASETS_EXPECTED]


@pytest.fixture(params=DATASETS_EXPECTED, ids=DATASET_IDS, scope="session")
def loaded_dataset(request):
    """Fixture that loads a dataset and returns it along with metadata."""
    if not Config().is_configured():
        pytest.skip("Not logged in to Hafnia")

    dataset_name, expected_lengths, task_type = request.param
    dataset = load_dataset(dataset_name, force_redownload=FORCE_REDOWNLOAD)

    return {
        "dataset": dataset,
        "dataset_name": dataset_name,
        "expected_lengths": expected_lengths,
        "task_type": task_type,
    }


def hafnia_2_torch_dataset(dataset: datasets.Dataset) -> torch.utils.data.Dataset:
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
    dataset: HafniaDataset = loaded_dataset["dataset"]
    expected_split_counts = loaded_dataset["expected_lengths"]

    actual_split_counts = dict(dataset.table[ColumnName.SPLIT].value_counts().iter_rows())
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
    dataset_name = loaded_dataset["dataset_name"]
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
        assert isinstance(images, torch.Tensor)
        assert images.shape[0] == batch_size
        assert images.shape[2:] == (224, 224)
        break
