import json
from pathlib import Path

import pytest

from hafnia.dataset.dataset_names import SampleField
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import DatasetMetadataFilePaths, Sample
from tests import helper_testing
from tests.helper_testing_datasets import from_paths_test_setup, write_sample_files


def test_sample_fields():
    column_variable_names = list(SampleField.__annotations__)
    sample_fields = Sample.model_fields.keys()
    for column_variable_name in column_variable_names:
        column_name = getattr(SampleField, column_variable_name)
        assert column_name in sample_fields, (
            f"Column name '{column_name}' defined in 'FieldName.{column_variable_name}' "
            f"not found in '{Sample.__name__}' fields. Available fields are: {list(sample_fields)}"
        )


def test_metadata_files_from_path_failure_modes(tmp_path: Path):
    path_dataset = helper_testing.get_path_micro_hafnia_dataset(dataset_name="micro-tiny-dataset", force_update=False)

    dataset = HafniaDataset.from_path(path_dataset)
    dataset.write(path_folder=tmp_path)
    assert tmp_path.exists()

    with pytest.raises(FileNotFoundError, match="Dataset info file missing"):
        metadata = DatasetMetadataFilePaths.from_path(tmp_path)
        metadata.dataset_info = "non_existing_file.json"
        metadata.exists(raise_error=True)
    assert metadata.exists(raise_error=False) is False

    with pytest.raises(FileNotFoundError, match="Missing annotation file"):
        metadata = DatasetMetadataFilePaths.from_path(tmp_path)
        metadata.annotations_jsonl = None
        metadata.annotations_parquet = None
        metadata.exists(raise_error=True)
    assert metadata.exists(raise_error=False) is False

    with pytest.raises(FileNotFoundError, match="Missing annotation file"):
        metadata = DatasetMetadataFilePaths.from_path(tmp_path)
        metadata.annotations_jsonl = "non_existing_file.json"
        metadata.annotations_parquet = "non_existing_file.json"
        metadata.exists(raise_error=True)
    assert metadata.exists(raise_error=False) is False


def test_metadata_files_from_path():
    path_dataset = helper_testing.get_path_micro_hafnia_dataset(dataset_name="micro-tiny-dataset", force_update=False)

    metadata = DatasetMetadataFilePaths.from_path(path_dataset)
    metadata.annotations_jsonl = None
    metadata.exists(raise_error=True)  # Works because parquet exists
    assert metadata.exists() is True

    metadata = DatasetMetadataFilePaths.from_path(path_dataset)
    metadata.annotations_parquet = None
    metadata.exists(raise_error=True)  # Works because jsonl exists
    assert metadata.exists() is True


def test_samples_from_files(tmp_path: Path):
    _, samples = from_paths_test_setup(tmp_path)
    sample_paths = write_sample_files(tmp_path, samples)

    assert len(list(Sample.samples_from_file(sample_paths[0]))) == 1

    samples_iterator = Sample.samples_from_files(sample_paths, media_paths=[tmp_path / "override.jpg", None])
    file_paths = [sample.file_path for sample in samples_iterator]
    assert file_paths == [(tmp_path / "override.jpg").as_posix(), (tmp_path / "image1.jpg").as_posix()]

    with pytest.raises(FileNotFoundError):
        Sample.samples_from_files(sample_paths + [tmp_path / "missing.jsonl"])


def test_sample_to_file_and_samples_to_file(tmp_path: Path):
    _, samples = from_paths_test_setup(tmp_path)

    # No suffix defaults to '.jsonl'
    path_jsonl = samples[0].to_file(tmp_path / "out" / "single")
    assert path_jsonl.suffix == ".jsonl"
    assert list(Sample.samples_from_file(path_jsonl)) == [samples[0]]

    path_json = samples[0].to_file(tmp_path / "out" / "single.json")
    assert isinstance(json.loads(path_json.read_text()), dict)
    assert list(Sample.samples_from_file(path_json)) == [samples[0]]

    for suffix in [".jsonl", ".json"]:
        path_multi = Sample.samples_to_file(iter(samples), tmp_path / "out" / f"multi{suffix}")
        assert list(Sample.samples_from_file(path_multi)) == samples

    with pytest.raises(ValueError, match="Unsupported file format"):
        samples[0].to_file(tmp_path / "out" / "single.txt")
    with pytest.raises(ValueError, match="No samples to write"):
        Sample.samples_to_file([], tmp_path / "out" / "empty.jsonl")
    assert not (tmp_path / "out" / "empty.jsonl").exists()
