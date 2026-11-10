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
from tests.helper_testing import get_micro_hafnia_dataset

SPLITS = [SplitName.TRAIN, SplitName.TRAIN, SplitName.VAL, SplitName.TEST, SplitName.TEST, SplitName.UNDEFINED]
DATASET_NAME = "tiny-dataset"
DATASET_VERSION = "1.0.0"


def make_dataset(flagged: bool = True) -> HafniaDataset:
    micro = get_micro_hafnia_dataset("micro-tiny-dataset")
    samples = pl.concat([micro.samples.drop(SampleField.FILE_PATH)] * 2).head(len(SPLITS))
    samples = samples.with_columns(
        pl.Series(SampleField.SPLIT, SPLITS),
        pl.Series(SampleField.REMOTE_PATH, [f"s3://bucket/data/sample-{i}.png" for i in range(len(SPLITS))]),
        pl.Series(SampleField.SAMPLE_INDEX, list(range(len(SPLITS))), dtype=pl.UInt64),
    )
    info = micro.info.model_copy(deep=True)
    info.version = DATASET_VERSION
    info.meta = {META_PROTECTED_TEST_SPLIT: True} if flagged else None
    return HafniaDataset(samples=samples, info=info)


def original_test_remote_paths() -> set:
    return set(make_dataset().create_split_dataset(SplitName.TEST).samples[SampleField.REMOTE_PATH].to_list())


def set_env(monkeypatch: pytest.MonkeyPatch, cloud: bool, policy: Optional[str] = None) -> None:
    if cloud:
        monkeypatch.setenv("HAFNIA_CLOUD", "true")
    else:
        monkeypatch.delenv("HAFNIA_CLOUD", raising=False)
    if policy is None:
        monkeypatch.delenv(ENV_TEST_SPLIT_POLICY, raising=False)
    else:
        monkeypatch.setenv(ENV_TEST_SPLIT_POLICY, policy)


def test_local_job_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, cloud=False)
    dataset = make_dataset()
    assert apply_test_split_policy(dataset).samples.equals(dataset.samples)


def test_cloud_unflagged_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, cloud=True)
    dataset = make_dataset(flagged=False)
    assert apply_test_split_policy(dataset).samples.equals(dataset.samples)


def test_cloud_flagged_withholds_test(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, cloud=True)
    dataset = make_dataset()
    with patch("hafnia.log.user_logger.warning") as mock_warning:
        result = apply_test_split_policy(dataset)

    expected = dataset.calculate_split_counts()
    expected.pop(SplitName.TEST)
    assert result.calculate_split_counts() == expected
    mock_warning.assert_called_once()


def test_cloud_flagged_separate_keeps_test(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, cloud=True, policy="separate")
    dataset = make_dataset()
    assert apply_test_split_policy(dataset).samples.equals(dataset.samples)


def test_unknown_policy_withholds(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, cloud=True, policy="seperate")
    assert SplitName.TEST not in apply_test_split_policy(make_dataset()).calculate_split_counts()


@pytest.fixture
def platform(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Serve the flagged dataset as platform metadata files. The cache lands in tmp_path/.data."""
    path_platform = tmp_path / "platform"
    make_dataset().write_annotations(path_folder=path_platform)
    monkeypatch.chdir(tmp_path)

    def fake_download_meta(resource_credentials, version, path_dataset: Path) -> list:
        shutil.copytree(path_platform, path_dataset, dirs_exist_ok=True)
        return []

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


def test_download_syncs_no_test_files(platform, monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, cloud=True)
    download_or_get_dataset_path(DATASET_NAME, version=DATASET_VERSION, download_files=True)
    assert SplitName.TEST not in platform[0].calculate_split_counts()


def test_recipe_resplit_has_no_original_test_samples(platform, monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, cloud=True)
    recipe = DatasetRecipe.from_name(DATASET_NAME, version=DATASET_VERSION).splits_by_ratios(
        {SplitName.TRAIN: 0.5, SplitName.TEST: 0.5}
    )
    remote_paths = set(recipe.build(download_files=False).samples[SampleField.REMOTE_PATH].to_list())
    assert remote_paths.isdisjoint(original_test_remote_paths())


def test_recipe_from_path_on_cache_has_no_original_test_samples(platform, monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, cloud=True)
    path_cache = Path(".data") / "datasets" / DATASET_NAME
    recipe = DatasetRecipe.from_merger(
        [
            DatasetRecipe.from_name(DATASET_NAME, version=DATASET_VERSION),
            DatasetRecipe.from_path(path_cache, check_for_images=False),
        ]
    )
    remote_paths = set(recipe.build(download_files=False).samples[SampleField.REMOTE_PATH].to_list())
    assert remote_paths.isdisjoint(original_test_remote_paths())
