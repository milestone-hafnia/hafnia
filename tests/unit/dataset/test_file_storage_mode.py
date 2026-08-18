"""Tests for storing dataset images as real copies or as symbolic links (`FileStorageMode`)."""

from pathlib import Path
from typing import List

import polars as pl
import pytest

from hafnia.dataset.dataset_helpers import FileStorageMode
from hafnia.dataset.dataset_names import SampleField
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.utils import is_image_file
from tests.helper_testing import get_micro_hafnia_dataset, skip_if_no_symlink_support

STORAGE_MODES = [FileStorageMode.COPY, FileStorageMode.SYMLINK]


def get_stored_image_paths(path_folder: Path) -> List[Path]:
    return sorted(path for path in path_folder.rglob("*") if path.is_file() and is_image_file(path))


def assert_images_are_stored_as(path_output: Path, storage_mode: FileStorageMode) -> None:
    path_images = get_stored_image_paths(path_output)
    assert len(path_images) > 0, f"Expected exported images in '{path_output}'"
    for path_image in path_images:
        is_symlink = path_image.is_symlink()
        assert is_symlink == (storage_mode is FileStorageMode.SYMLINK), (
            f"Unexpected storage type for '{path_image}' with '{storage_mode}'"
        )
        if is_symlink:
            path_source = Path(path_image.readlink())
            assert path_source.is_absolute(), "Symlinks should point at an absolute path"
            assert path_source.is_file(), f"Symlink '{path_image}' does not point at an existing file"


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_write_storage_mode(tmp_path: Path, storage_mode: FileStorageMode) -> None:
    skip_if_no_symlink_support(tmp_path)
    dataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")
    path_output = tmp_path / "written_dataset"

    dataset.write(path_output, storage_mode=storage_mode)

    assert_images_are_stored_as(path_output / "data", storage_mode)

    # The written dataset can be reloaded and images can be read through the symlinks
    dataset_reloaded = HafniaDataset.from_path(path_output)
    assert len(dataset_reloaded.samples) == len(dataset.samples)
    for path_image in dataset_reloaded.samples[SampleField.FILE_PATH].to_list():
        assert Path(path_image).read_bytes(), f"Unable to read image data from '{path_image}'"


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_to_yolo_format_storage_mode(tmp_path: Path, storage_mode: FileStorageMode) -> None:
    skip_if_no_symlink_support(tmp_path)
    dataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")
    path_output = tmp_path / "yolo_dataset"

    dataset.to_yolo_format(path_output=path_output, storage_mode=storage_mode)

    assert_images_are_stored_as(path_output, storage_mode)
    HafniaDataset.from_yolo_format(path_dataset=path_output, dataset_name="micro-tiny-dataset")


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_to_coco_format_storage_mode(tmp_path: Path, storage_mode: FileStorageMode) -> None:
    skip_if_no_symlink_support(tmp_path)
    dataset = get_micro_hafnia_dataset(dataset_name="micro-coco-2017")
    path_output = tmp_path / "coco_dataset"

    dataset.to_coco_format(path_output=path_output, storage_mode=storage_mode)

    assert_images_are_stored_as(path_output, storage_mode)
    HafniaDataset.from_coco_format(path_dataset=path_output)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_to_image_classification_folder_storage_mode(tmp_path: Path, storage_mode: FileStorageMode) -> None:
    skip_if_no_symlink_support(tmp_path)
    dataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")
    path_output = tmp_path / "image_classification_dataset"

    dataset.to_image_classification_folder(
        path_output=path_output,
        task_name="Time of Day",
        storage_mode=storage_mode,
    )

    assert_images_are_stored_as(path_output, storage_mode)
    HafniaDataset.from_image_classification_folder(path_folder=path_output)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_storage_mode_accepts_string_value(tmp_path: Path, storage_mode: FileStorageMode) -> None:
    skip_if_no_symlink_support(tmp_path)
    dataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")
    path_output = tmp_path / "written_dataset"

    dataset.write(path_output, storage_mode=storage_mode.value)

    assert_images_are_stored_as(path_output / "data", storage_mode)


def test_re_export_repairs_broken_symlinks(tmp_path: Path) -> None:
    """Writing into an existing folder skips files that are already there, but must repair broken links."""
    skip_if_no_symlink_support(tmp_path)
    dataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")
    path_output = tmp_path / "written_dataset"
    dataset.write(path_output, storage_mode=FileStorageMode.SYMLINK)

    # Break the stored links, as if the original files had been moved away and later restored
    for path_image in get_stored_image_paths(path_output / "data"):
        path_image.unlink()
        path_image.symlink_to(tmp_path / "does_not_exist.png")

    dataset.write(path_output, storage_mode=FileStorageMode.SYMLINK)

    assert_images_are_stored_as(path_output / "data", FileStorageMode.SYMLINK)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_export_with_missing_image_file_raises(tmp_path: Path, storage_mode: FileStorageMode) -> None:
    """
    A sample pointing at a missing image should fail the export for both modes. Without a check,
    symlink mode would create a broken symlink and the failure would first appear at training time.
    """
    skip_if_no_symlink_support(tmp_path)
    dataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")
    path_missing = tmp_path / "missing_image.jpg"
    dataset.samples = dataset.samples.with_columns(
        pl.lit(str(path_missing)).alias(SampleField.FILE_PATH),
    )

    with pytest.raises(FileNotFoundError):
        dataset.to_yolo_format(path_output=tmp_path / "yolo_dataset", storage_mode=storage_mode)

    with pytest.raises(FileNotFoundError):
        dataset.write(tmp_path / "written_dataset", storage_mode=storage_mode)


def test_write_with_unknown_storage_mode(tmp_path: Path) -> None:
    dataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")

    with pytest.raises(ValueError):
        dataset.write(tmp_path / "written_dataset", storage_mode="hardlink")
