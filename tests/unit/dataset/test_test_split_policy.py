import shutil
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from hafnia.dataset.dataset_names import SampleField, SplitName
from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe
from hafnia.dataset.hafnia_dataset import (
    ENV_TEST_SPLIT_POLICY,
    META_PROTECTED_TEST_SPLIT,
    HafniaDataset,
    apply_test_split_policy,
    download_or_get_dataset_path,
)
from hafnia.dataset.hafnia_dataset_types import DatasetMetadataFilePaths
from tests.helper_testing import get_micro_hafnia_dataset

SPLITS = [
    SplitName.TRAIN,
    SplitName.TRAIN,
    SplitName.TRAIN,
    SplitName.VAL,
    SplitName.VAL,
    SplitName.TEST,
    SplitName.TEST,
    SplitName.UNDEFINED,
]
DATASET_NAME = "tiny-dataset"
DATASET_VERSION = "1.0.0"


def make_dataset(flagged: bool = True) -> HafniaDataset:
    micro = get_micro_hafnia_dataset("micro-tiny-dataset")
    base = micro.samples.drop(SampleField.FILE_PATH)
    samples = pl.concat([base] * 3).head(len(SPLITS))
    samples = samples.with_columns(
        pl.Series(SampleField.SPLIT, SPLITS),
        pl.Series(SampleField.REMOTE_PATH, [f"s3://bucket/data/sample-{i}.png" for i in range(len(SPLITS))]),
        pl.Series(SampleField.SAMPLE_INDEX, list(range(len(SPLITS))), dtype=pl.UInt64),
    )
    info = micro.info.model_copy(deep=True)
    info.version = DATASET_VERSION
    info.meta = {META_PROTECTED_TEST_SPLIT: True} if flagged else None
    return HafniaDataset(samples=samples, info=info)


def split_counts(dataset: HafniaDataset) -> dict:
    return dict(dataset.samples[SampleField.SPLIT].value_counts().iter_rows())


def original_test_remote_paths() -> set:
    samples = make_dataset().samples
    return set(samples.filter(pl.col(SampleField.SPLIT) == SplitName.TEST)[SampleField.REMOTE_PATH].to_list())


def remote_paths(dataset: HafniaDataset) -> set:
    return set(dataset.samples[SampleField.REMOTE_PATH].to_list())


def set_env(monkeypatch: pytest.MonkeyPatch, cloud: bool, policy: Optional[str] = None) -> None:
    if cloud:
        monkeypatch.setenv("HAFNIA_CLOUD", "true")
    else:
        monkeypatch.delenv("HAFNIA_CLOUD", raising=False)
    if policy is None:
        monkeypatch.delenv(ENV_TEST_SPLIT_POLICY, raising=False)
    else:
        monkeypatch.setenv(ENV_TEST_SPLIT_POLICY, policy)


def warning_messages(mock_warning: MagicMock) -> List[str]:
    return [str(call.args[0]) for call in mock_warning.call_args_list]


### apply_test_split_policy ###


def test_local_job_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, cloud=False)
    dataset = make_dataset()
    with patch("hafnia.log.user_logger.warning") as mock_warning:
        result = apply_test_split_policy(dataset)
    assert result.samples.equals(dataset.samples)
    mock_warning.assert_not_called()


@pytest.mark.parametrize("meta", [None, {}, {"other": 1}, {META_PROTECTED_TEST_SPLIT: False}])
def test_cloud_unflagged_is_unchanged(monkeypatch: pytest.MonkeyPatch, meta: Optional[dict]) -> None:
    set_env(monkeypatch, cloud=True)
    dataset = make_dataset()
    dataset.info.meta = meta
    with patch("hafnia.log.user_logger.warning") as mock_warning:
        result = apply_test_split_policy(dataset)
    assert result.samples.equals(dataset.samples)
    mock_warning.assert_not_called()


@pytest.mark.parametrize("policy", [None, "", "withhold", " WITHHOLD "])
def test_cloud_flagged_withholds_test(monkeypatch: pytest.MonkeyPatch, policy: Optional[str]) -> None:
    set_env(monkeypatch, cloud=True, policy=policy)
    dataset = make_dataset()
    with patch("hafnia.log.user_logger.warning") as mock_warning:
        result = apply_test_split_policy(dataset)

    counts_before = split_counts(dataset)
    counts_after = split_counts(result)
    assert SplitName.TEST not in counts_after
    for split in (SplitName.TRAIN, SplitName.VAL, SplitName.UNDEFINED):
        assert counts_after[split] == counts_before[split]
    assert result.info.meta == {META_PROTECTED_TEST_SPLIT: True}

    messages = warning_messages(mock_warning)
    assert len(messages) == 1
    assert DATASET_NAME in messages[0]
    assert str(counts_before[SplitName.TEST]) in messages[0]


@pytest.mark.parametrize("policy", ["separate", " Separate "])
def test_cloud_flagged_separate_keeps_test(monkeypatch: pytest.MonkeyPatch, policy: str) -> None:
    set_env(monkeypatch, cloud=True, policy=policy)
    dataset = make_dataset()
    with patch("hafnia.log.user_logger.warning") as mock_warning:
        result = apply_test_split_policy(dataset)
    assert result.samples.equals(dataset.samples)
    mock_warning.assert_not_called()


def test_unknown_policy_withholds_and_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, cloud=True, policy="seperate")
    with patch("hafnia.log.user_logger.warning") as mock_warning:
        result = apply_test_split_policy(make_dataset())
    assert SplitName.TEST not in split_counts(result)
    assert any("seperate" in message for message in warning_messages(mock_warning))


def test_already_filtered_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, cloud=True)
    dataset = make_dataset()
    dataset = dataset.update_samples(dataset.samples.filter(pl.col(SampleField.SPLIT) != SplitName.TEST))
    with patch("hafnia.log.user_logger.warning") as mock_warning:
        result = apply_test_split_policy(dataset)
    assert result.samples.equals(dataset.samples)
    mock_warning.assert_not_called()


### Platform load path (network edges patched) ###


@pytest.fixture
def platform(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Serve the flagged fixture as platform metadata files; cwd is tmp_path, so the cache is ./.data."""
    path_platform = tmp_path / "platform"
    HafniaDataset.write_annotations(make_dataset(), path_folder=path_platform)
    monkeypatch.chdir(tmp_path)

    def fake_download_meta(resource_credentials, version, path_dataset: Path) -> list:
        path_dataset.mkdir(parents=True, exist_ok=True)
        files = [p for p in path_platform.iterdir() if p.is_file()]
        for path_file in files:
            shutil.copy(path_file, path_dataset / path_file.name)
        return [str(path_dataset / p.name) for p in files]

    synced: List[HafniaDataset] = []

    def fake_sync(dataset: HafniaDataset, path_output_folder: Path, **kwargs) -> HafniaDataset:
        synced.append(dataset)
        return dataset

    with (
        patch("hafnia.dataset.hafnia_dataset.Config"),
        patch("hafnia.dataset.hafnia_dataset.get_read_credentials_by_name", return_value=MagicMock()),
        patch("hafnia.dataset.hafnia_dataset.download_meta_dataset_files_from_version", side_effect=fake_download_meta),
        patch("hafnia.dataset.operations.dataset_s3_storage.sync_files_from_platform", side_effect=fake_sync),
    ):
        yield synced


def path_cache() -> Path:
    return Path(".data") / "datasets" / DATASET_NAME


def cached_splits() -> dict:
    metadata_files = DatasetMetadataFilePaths.from_path(path_cache())
    parquet = pl.read_parquet(metadata_files.annotations_parquet)
    jsonl = pl.read_ndjson(metadata_files.annotations_jsonl)
    return {
        "parquet": set(parquet[SampleField.SPLIT].to_list()),
        "jsonl": set(jsonl[SampleField.SPLIT].to_list()),
    }


def test_download_path_filters_cache_and_synced_files(platform, monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, cloud=True)
    download_or_get_dataset_path(DATASET_NAME, version=DATASET_VERSION, download_files=True)

    for splits in cached_splits().values():
        assert SplitName.TEST not in splits
    assert len(platform) == 1
    assert SplitName.TEST not in split_counts(platform[0])


def test_download_path_separate_keeps_test(platform, monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, cloud=True, policy="separate")
    download_or_get_dataset_path(DATASET_NAME, version=DATASET_VERSION, download_files=True)

    for splits in cached_splits().values():
        assert SplitName.TEST in splits
    assert SplitName.TEST in split_counts(platform[0])


def test_from_name_filters_cached_unfiltered_copy(platform, monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, cloud=False)
    download_or_get_dataset_path(DATASET_NAME, version=DATASET_VERSION, download_files=False)
    assert SplitName.TEST in cached_splits()["parquet"]

    set_env(monkeypatch, cloud=True)
    dataset = HafniaDataset.from_name(DATASET_NAME, version=DATASET_VERSION, download_files=False)
    assert SplitName.TEST not in split_counts(dataset)


### Recipes ###


def test_recipe_resplit_has_no_original_test_samples(platform, monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, cloud=True)
    recipe = DatasetRecipe.from_name(DATASET_NAME, version=DATASET_VERSION).splits_by_ratios(
        {SplitName.TRAIN: 0.5, SplitName.VAL: 0.25, SplitName.TEST: 0.25}
    )
    dataset = recipe.build(download_files=False)
    assert len(dataset) > 0
    assert remote_paths(dataset).isdisjoint(original_test_remote_paths())


def test_recipe_from_path_on_cache_has_no_original_test_samples(platform, monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, cloud=True)
    recipe = DatasetRecipe.from_merger(
        [
            DatasetRecipe.from_name(DATASET_NAME, version=DATASET_VERSION),
            DatasetRecipe.from_path(path_cache(), check_for_images=False),
        ]
    )
    dataset = recipe.build(download_files=False)
    assert len(dataset) > 0
    assert remote_paths(dataset).isdisjoint(original_test_remote_paths())


def test_from_path_is_not_filtered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, cloud=True)
    HafniaDataset.write_annotations(make_dataset(), path_folder=tmp_path)
    dataset = HafniaDataset.from_path(tmp_path, check_for_images=False)
    assert SplitName.TEST in split_counts(dataset)
