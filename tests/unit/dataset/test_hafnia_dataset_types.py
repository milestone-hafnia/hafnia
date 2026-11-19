import json
from pathlib import Path
from typing import List

import pytest

from hafnia.dataset import dataset_helpers
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


def media_samples_setup(path_source: Path) -> List[Sample]:
    """Samples with absolute 'file_path's to media files with unique content (and thereby unique hash values)."""
    path_source.mkdir(parents=True, exist_ok=True)
    _, samples = from_paths_test_setup(path_source)
    for sample in samples:
        (path_source / str(sample.file_path)).write_text(f"content of {sample.file_path}")
    return [
        sample.model_copy(update={"file_path": (path_source / str(sample.file_path)).as_posix()}) for sample in samples
    ]


def test_sample_to_file_stores_media_by_hash(tmp_path: Path):
    path_source = tmp_path / "source"
    samples = media_samples_setup(path_source)
    path_out = tmp_path / "out"

    path_sample = samples[0].to_file(path_out)

    path_media = path_out / f"{path_sample.stem}.jpg"  # Sample and media file are named by the media hash
    assert path_sample == path_out / f"{path_sample.stem}.jsonl"
    assert path_sample.stem == dataset_helpers.hash_file_xxhash(Path(str(samples[0].file_path)))
    assert path_media.read_text() == "content of image0.jpg"
    assert not path_media.is_symlink()
    assert json.loads(path_sample.read_text())[SampleField.FILE_PATH] == path_media.name  # Relative to sample file

    (sample_read,) = Sample.samples_from_file(path_sample)
    assert sample_read.file_path == path_media.absolute().as_posix()
    assert sample_read.model_copy(update={"file_path": samples[0].file_path}) == samples[0]

    # '.json' (with or without a leading dot) stores a single sample as an object
    path_json = samples[1].to_file(path_out, suffix="json")
    assert path_json.suffix == ".json"
    assert isinstance(json.loads(path_json.read_text()), dict)


def test_samples_to_file_multiple_samples_of_same_media(tmp_path: Path):
    path_source = tmp_path / "source"
    sample = media_samples_setup(path_source)[0]
    frames = [sample.model_copy(update={"collection_index": i}) for i in range(3)]

    for suffix in [".jsonl", ".json"]:
        path_sample = Sample.samples_to_file(iter(frames), tmp_path / "out", suffix=suffix)
        assert [s.collection_index for s in Sample.samples_from_file(path_sample)] == [0, 1, 2]

    # An existing sample file of the same media file is overwritten
    path_sample = Sample.samples_to_file(frames[:1], tmp_path / "out")
    assert len(list(Sample.samples_from_file(path_sample))) == 1
    assert len(list((tmp_path / "out").glob("*.jpg"))) == 1, "The media file is stored once"


def test_samples_to_file_round_trip_with_from_paths(tmp_path: Path):
    path_source = tmp_path / "source"
    samples = media_samples_setup(path_source)
    dataset_info, _ = from_paths_test_setup(tmp_path)  # Only for 'dataset_info' - writes empty images to 'tmp_path'
    sample_paths = [sample.to_file(tmp_path / "out") for sample in samples]
    assert len(set(sample_paths)) == 2, "Samples of different media files are stored in different sample files"

    dataset = HafniaDataset.from_paths(sample_paths, dataset_info=dataset_info)

    assert len(dataset) == 2
    columns = [SampleField.SPLIT, SampleField.CLASSIFICATIONS, SampleField.HEIGHT, SampleField.WIDTH]
    dataset_expected = HafniaDataset.from_samples_list(samples, info=dataset_info)
    assert dataset.samples.select(columns).equals(dataset_expected.samples.select(columns))
    assert all(Path(path).parent == tmp_path / "out" for path in dataset.samples[SampleField.FILE_PATH].to_list())


def test_sample_to_file_relative_file_path_is_resolved_from_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path_source = tmp_path / "source"
    sample = media_samples_setup(path_source)[0]
    monkeypatch.chdir(path_source)

    path_sample = sample.model_copy(update={"file_path": "image0.jpg"}).to_file(tmp_path / "out")

    assert (tmp_path / "out" / f"{path_sample.stem}.jpg").read_text() == "content of image0.jpg"


def test_sample_to_file_as_symlink(tmp_path: Path):
    helper_testing.skip_if_no_symlink_support(tmp_path)
    path_source = tmp_path / "source"
    sample = media_samples_setup(path_source)[0]

    path_sample = sample.to_file(tmp_path / "out", storage_mode="symlink")

    path_media = tmp_path / "out" / f"{path_sample.stem}.jpg"
    assert path_media.is_symlink()
    assert path_media.resolve() == Path(str(sample.file_path)).resolve()


def test_samples_to_file_errors(tmp_path: Path):
    path_source = tmp_path / "source"
    samples = media_samples_setup(path_source)
    path_out = tmp_path / "out"

    with pytest.raises(ValueError, match="No samples to write"):
        Sample.samples_to_file([], path_out)
    with pytest.raises(ValueError, match="must have a 'file_path'"):
        samples[0].model_copy(update={"file_path": None}).to_file(path_out)
    with pytest.raises(ValueError, match="must have the same 'file_path'"):
        Sample.samples_to_file(samples, path_out)
    with pytest.raises(ValueError, match="Unsupported sample file format '.txt'"):
        samples[0].to_file(path_out, suffix=".txt")
    with pytest.raises(FileNotFoundError):
        samples[0].model_copy(update={"file_path": (path_source / "missing.jpg").as_posix()}).to_file(path_out)
    assert not path_out.exists() or not any(path_out.iterdir()), "Nothing is written on errors"
