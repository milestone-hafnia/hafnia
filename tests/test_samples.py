from pathlib import Path

import datasets
import PIL
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from cli.config import Config
from hafnia import torch_helpers
from hafnia.data import load_dataset

FORCE_REDOWNLOAD = False

DATASETS_EXPECTED = [
    (
        "midwest-vehicle-detection",
        {"train": 172, "validation": 21, "test": 21},
        "ObjectDetection",
    ),
    ("mnist", {"train": 60_000, "test": 10_000}, "ImageClassification"),
    ("caltech-101", {"train": 160, "validation": 20, "test": 20}, "ImageClassification"),
    ("caltech-256", {"train": 160, "validation": 20, "test": 20}, "ImageClassification"),
    ("cifar10", {"train": 45000, "validation": 5000, "test": 10000}, "ImageClassification"),
    ("cifar100", {"train": 45000, "validation": 5000, "test": 10000}, "ImageClassification"),
    ("easyportrait", {"train": 32, "test": 20, "validation": 10}, "Segmentation"),
    ("coco-2017", {"train": 182, "validation": 18, "test": 18}, "ObjectDetection"),
    ("sama-coco", {"train": 99, "validation": 1, "test": 1}, "ObjectDetection"),
    ("open-images-v7", {"train": 91, "validation": 3, "test": 9}, "ObjectDetection"),
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


def hf_2_torch_dataset(dataset: datasets.Dataset) -> torch.utils.data.Dataset:
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


def test_dataset_lengths(loaded_dataset):
    """Test that the dataset has the expected number of samples."""
    dataset = loaded_dataset["dataset"]
    expected_lengths = loaded_dataset["expected_lengths"]

    actual_lengths = {split_name: len(split) for split_name, split in dataset.items()}
    assert actual_lengths == expected_lengths


def test_dataset_features(loaded_dataset):
    """Test the features of the dataset based on task type."""
    dataset = loaded_dataset["dataset"]
    dataset_name = loaded_dataset["dataset_name"]
    task_type = loaded_dataset["task_type"]

    for dataset_split in dataset.values():
        assert dataset_split.info.dataset_name == dataset_name
        sample = dataset_split[0]

        if task_type == "ImageClassification":
            assert "classification" in dataset_split.features
            assert "class_idx" in dataset_split.features["classification"]
            assert isinstance(dataset_split.features["classification"]["class_idx"], datasets.ClassLabel)

            assert "classification" in sample
            assert "class_idx" in sample["classification"]
            assert isinstance(sample["classification"]["class_idx"], int)
            assert "class_name" in sample["classification"]
            assert isinstance(sample["classification"]["class_name"], str)
        elif task_type == "ObjectDetection":
            assert "objects" in dataset_split.features
            assert "bbox" in dataset_split.features["objects"].feature
            assert isinstance(dataset_split.features["objects"].feature["bbox"], datasets.Sequence)
            assert isinstance(dataset_split.features["objects"].feature["class_idx"], datasets.ClassLabel)

            assert "objects" in sample
            assert "bbox" in sample["objects"]
            assert isinstance(sample["objects"]["bbox"], list)
            for bbox in sample["objects"]["bbox"]:
                assert isinstance(bbox, list)
                assert len(bbox) == 4

            assert isinstance(sample["objects"]["class_idx"], list)
            assert isinstance(sample["objects"]["class_idx"][0], int)

            assert isinstance(sample["objects"]["class_name"], list)
            assert isinstance(sample["objects"]["class_name"][0], str)
        elif task_type == "Segmentation":
            assert "segmentation" in dataset_split.features
            assert "mask" in dataset_split.features["segmentation"]
            assert isinstance(dataset_split.features["segmentation"]["mask"], datasets.Image)

            assert "segmentation" in sample
            assert "mask" in sample["segmentation"]
            assert isinstance(sample["segmentation"]["mask"], PIL.Image.Image)
        else:
            raise ValueError(f"Unknown task type: {task_type}")


def test_dataset_draw_image_and_target(loaded_dataset):
    """Test data transformations and visualization."""
    dataset = loaded_dataset["dataset"]
    dataset_name = loaded_dataset["dataset_name"]
    HAS_BEEN_ANONYMIZED = False
    torch_dataset = hf_2_torch_dataset(dataset["train"])

    # Test single item transformation
    image, targets = torch_dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape[0] in (3, 1)  # RGB or grayscale
    assert image.shape[1:] == (224, 224)  # Resized dimensions

    # Test visualization
    visualized = torch_helpers.draw_image_and_targets(image=image, targets=targets)
    assert isinstance(visualized, torch.Tensor)

    # Save visualization if directory exists
    if HAS_BEEN_ANONYMIZED:
        output_dir = Path("tests") / "data"
        pil_image = v2.functional.to_pil_image(visualized)
        pil_image.save(output_dir / f"visualized_{dataset_name}.png")


def test_dataset_dataloader(loaded_dataset):
    """Test dataloader functionality."""
    dataset = loaded_dataset["dataset"]
    torch_dataset = hf_2_torch_dataset(dataset["train"])

    # Test dataloader with custom collate function
    skip_stacking = ["objects.bbox", "objects.class_idx"]
    batch_size = 2
    collate_fn = torch_helpers.TorchVisionCollateFn(skip_stacking=skip_stacking)
    dataloader_train = DataLoader(batch_size=batch_size, dataset=torch_dataset, collate_fn=collate_fn)

    # Test iteration
    for images, targets in dataloader_train:
        assert isinstance(images, torch.Tensor)
        assert images.shape[0] == batch_size
        assert images.shape[2:] == (224, 224)
        break
