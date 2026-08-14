import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
from packaging.version import Version

from hafnia.dataset import dataset_helpers
from hafnia.dataset.dataset_helpers import FileStorageMode
from hafnia.dataset.dataset_names import SplitName
from tests.helper_testing import skip_if_no_symlink_support


@dataclass
class CreateSplitNameListFromRatiosTestCase:
    split_ratio: Dict[str, float]
    n_items: int
    expected_lengths: Dict[str, int]


@pytest.mark.parametrize(
    "test_case",
    [
        CreateSplitNameListFromRatiosTestCase(
            split_ratio={SplitName.TRAIN: 0.7, SplitName.TEST: 0.2, SplitName.VAL: 0.1},
            n_items=100,
            expected_lengths={SplitName.TRAIN: 70, SplitName.TEST: 20, SplitName.VAL: 10},
        ),
        CreateSplitNameListFromRatiosTestCase(
            split_ratio={SplitName.TRAIN: 0.70, SplitName.TEST: 0.2, SplitName.VAL: 0.1},
            n_items=1001,
            expected_lengths={SplitName.TRAIN: 701, SplitName.TEST: 200, SplitName.VAL: 100},
        ),
        CreateSplitNameListFromRatiosTestCase(
            split_ratio={SplitName.TRAIN: 0.333, SplitName.TEST: 0.333, SplitName.VAL: 0.333},
            n_items=1002,
            expected_lengths={SplitName.TRAIN: 334, SplitName.TEST: 334, SplitName.VAL: 334},
        ),
        CreateSplitNameListFromRatiosTestCase(
            split_ratio={SplitName.TRAIN: 0.333, SplitName.TEST: 0.333, SplitName.VAL: 0.333},
            n_items=103,
            expected_lengths={SplitName.TRAIN: 35, SplitName.TEST: 34, SplitName.VAL: 34},
        ),
        CreateSplitNameListFromRatiosTestCase(
            split_ratio={SplitName.TRAIN: 0.5, SplitName.TEST: 0.3, SplitName.VAL: 0.2},
            n_items=200,
            expected_lengths={SplitName.TRAIN: 100, SplitName.TEST: 60, SplitName.VAL: 40},
        ),
        CreateSplitNameListFromRatiosTestCase(
            split_ratio={SplitName.TEST: 0.5, SplitName.VAL: 0.5},
            n_items=101,
            expected_lengths={SplitName.TEST: 51, SplitName.VAL: 50},
        ),
        CreateSplitNameListFromRatiosTestCase(
            split_ratio={SplitName.TEST: 0.5, SplitName.VAL: 0.5},
            n_items=100,
            expected_lengths={SplitName.TEST: 50, SplitName.VAL: 50},
        ),
    ],
)
def test_create_split_name_list_from_ratios(test_case: CreateSplitNameListFromRatiosTestCase):
    split_names = dataset_helpers.create_split_name_list_from_ratios(
        split_ratios=test_case.split_ratio,
        n_items=test_case.n_items,
    )

    n_changes_in_split_name = sum([c_name != n_name for c_name, n_name in zip(split_names[:-1], split_names[1:])])
    split_names_have_been_shuffled = n_changes_in_split_name > 3
    assert split_names_have_been_shuffled
    assert len(split_names) == test_case.n_items
    assert dict(collections.Counter(split_names)) == test_case.expected_lengths


def test_save_image_with_hash_name(tmp_path: Path):
    dummy_image = (255 * np.random.rand(100, 100, 3)).astype(np.uint8)  # Create a dummy image
    tmp_path0 = tmp_path / "folder0"
    path_image0 = dataset_helpers.save_image_with_hash_name(dummy_image, tmp_path0)

    tmp_path1 = tmp_path / "folder1"
    path_image1 = dataset_helpers.copy_and_rename_file_to_hash_value(path_image0, tmp_path1)
    assert path_image1.relative_to(tmp_path1) == path_image0.relative_to(tmp_path0)
    assert path_image0.exists()
    assert path_image1.exists()
    assert path_image0.suffix in [".png"]
    assert path_image1.suffix in [".png"]


def test_store_file_as_copy(tmp_path: Path):
    path_source = tmp_path / "source.png"
    path_source.write_bytes(b"image-data")
    path_destination = tmp_path / "destination" / "image.png"

    dataset_helpers.store_file(path_source, path_destination, storage_mode=FileStorageMode.COPY)

    assert path_destination.is_symlink() is False
    assert path_destination.read_bytes() == b"image-data"


def test_store_file_as_symlink(tmp_path: Path):
    skip_if_no_symlink_support(tmp_path)
    path_source = tmp_path / "source.png"
    path_source.write_bytes(b"image-data")
    path_destination = tmp_path / "destination" / "image.png"

    dataset_helpers.store_file(path_source, path_destination, storage_mode=FileStorageMode.SYMLINK)

    assert path_destination.is_symlink()
    assert path_destination.read_bytes() == b"image-data"
    # An absolute link target ensures that the link keeps working if the destination folder is moved
    assert Path(path_destination.readlink()).is_absolute()


def test_store_file_replaces_existing_file_when_skip_is_disabled(tmp_path: Path):
    skip_if_no_symlink_support(tmp_path)
    path_source = tmp_path / "source.png"
    path_source.write_bytes(b"image-data")
    path_destination = tmp_path / "image.png"
    path_destination.write_bytes(b"old-data")

    dataset_helpers.store_file(path_source, path_destination, storage_mode="symlink", allow_skip=True)
    assert path_destination.is_symlink() is False, "An existing file should be kept with 'allow_skip=True'"

    dataset_helpers.store_file(path_source, path_destination, storage_mode="symlink", allow_skip=False)
    assert path_destination.is_symlink(), "An existing file should be replaced with 'allow_skip=False'"


def test_store_file_replaces_broken_symlink(tmp_path: Path):
    skip_if_no_symlink_support(tmp_path)
    path_source = tmp_path / "source.png"
    path_source.write_bytes(b"image-data")
    path_destination = tmp_path / "image.png"
    path_destination.symlink_to(tmp_path / "does_not_exist.png")

    dataset_helpers.store_file(path_source, path_destination, storage_mode="copy", allow_skip=False)

    assert path_destination.is_symlink() is False
    assert path_destination.read_bytes() == b"image-data"


@pytest.mark.parametrize("storage_mode", [FileStorageMode.COPY, FileStorageMode.SYMLINK])
def test_copy_and_rename_file_to_hash_value_storage_modes(tmp_path: Path, storage_mode: FileStorageMode):
    skip_if_no_symlink_support(tmp_path)
    dummy_image = (255 * np.random.rand(100, 100, 3)).astype(np.uint8)
    path_image_source = dataset_helpers.save_image_with_hash_name(dummy_image, tmp_path / "folder0")

    path_dataset_root = tmp_path / "folder1"
    path_image_stored = dataset_helpers.copy_and_rename_file_to_hash_value(
        path_image_source,
        path_dataset_root,
        storage_mode=storage_mode,
    )

    assert path_image_stored.name == path_image_source.name, "Files should be named by content hash for both modes"
    assert path_image_stored.read_bytes() == path_image_source.read_bytes()
    assert path_image_stored.is_symlink() == (storage_mode is FileStorageMode.SYMLINK)


def test_store_file_with_invalid_storage_mode(tmp_path: Path):
    path_source = tmp_path / "source.png"
    path_source.write_bytes(b"image-data")

    with pytest.raises(ValueError):
        dataset_helpers.store_file(path_source, tmp_path / "image.png", storage_mode="hardlink")


@pytest.mark.parametrize("version_str", ["1.0.0", "0.0.1"])
def test_version_from_string(version_str: str):
    version_casted: Version = dataset_helpers.version_from_string(version_str)
    assert isinstance(version_casted, Version)
    assert str(version_casted) == version_str
    assert dataset_helpers.is_valid_version_string(version_str) is True


@pytest.mark.parametrize("invalid_version_str", ["invalid_version", "latest", None, "1.0", "1", 1])
def test_invalid_version_from_string(invalid_version_str: str):
    with pytest.raises((ValueError, TypeError)):
        dataset_helpers.version_from_string(invalid_version_str, raise_error=True)
    version_none = dataset_helpers.version_from_string(invalid_version_str, raise_error=False)
    assert version_none is None
    assert dataset_helpers.is_valid_version_string(invalid_version_str) is False


def test_dataset_name_and_version_from_string():
    name_only = "dataset_name"
    name, version = dataset_helpers.dataset_name_and_version_from_string(name_only, resolve_missing_version=True)
    assert name == "dataset_name"
    assert version == "latest"

    name_version = "dataset_name:1.0.0"
    name, version = dataset_helpers.dataset_name_and_version_from_string(name_version)
    assert name == "dataset_name"
    assert version == "1.0.0"

    name_latest = "dataset_name:latest"
    name, version = dataset_helpers.dataset_name_and_version_from_string(name_latest)
    assert name == "dataset_name"
    assert version == "latest"


@pytest.mark.parametrize(
    "invalid_string",
    ["dataset_name", "dataset:name:extra", "dataset_name:asdf", "dataset_name:0.1", 123, None, "", "dataset_name:"],
)
def test_invalid_dataset_name_and_version_from_string(invalid_string: str):
    with pytest.raises((ValueError, TypeError)):
        dataset_helpers.dataset_name_and_version_from_string(invalid_string, resolve_missing_version=False)
