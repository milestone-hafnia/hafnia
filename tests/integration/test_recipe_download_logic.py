from pathlib import Path

import pytest

from hafnia.dataset.dataset_names import SampleField
from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.utils import is_hafnia_configured
from tests.helper_testing_datasets import DATASET_SPEC_MNIST


def test_recipe_download_files_with_write(tmp_path):
    """
    Verifies the logic used in the "fetching data" step. The main idea is to save resources by
    only downloading files that are used in the final recipe (after merging and filtering) and
    also avoiding unnecessary copy and hashing of files.


    We do this by combining the following steps:
    1) Skipping initial download of dataset images/videos with `download_files=False`. Causing only
       annotations / metadata to be downloaded.
    2) Downloading images/videos to a specific path with `download_files(path)`
    3) Writing the dataset to a specific path with `allow_skip=True` and `rename_by_hash=False` to avoid
    unnecessary copying and hashing of files that are already in the right place with the right name.

    # Code snippet to be used in "fetching data" step in the recipe build logic:
    dataset: HafniaDataset = recipe.build(download_files=False)
    dataset = dataset.download_files(path_write)
    dataset.write(path_write, rename_by_hash=False, allow_skip=True)
    """
    if not is_hafnia_configured():
        pytest.skip("Hafnia platform not configured. Skipping recipe download logic test.")

    recipe = DatasetRecipe.from_merge(
        recipe0=DatasetRecipe.from_name(DATASET_SPEC_MNIST.name, version=DATASET_SPEC_MNIST.version),
        recipe1=DatasetRecipe.from_name(DATASET_SPEC_MNIST.name, version=DATASET_SPEC_MNIST.version),
    )

    path_write = tmp_path / "write"

    dataset: HafniaDataset = recipe.build(download_files=False)  # Images/videos are not downloaded yet
    dataset = dataset.download_files(
        path_write
    )  # Now images are download to a single folder (avoid both a copy and downloading files that we don't need)

    # All file paths should now exist locally
    file_paths = dataset.samples[SampleField.FILE_PATH].to_list()
    assert len(file_paths) > 0
    for path in file_paths:
        assert Path(path).exists(), f"Expected downloaded file to exist: {path}"

    assert len(list(path_write.iterdir())) == 1, "Expect only 'data' folder in the download path"
    dataset.write(path_write, rename_by_hash=False, allow_skip=True)
    assert len(list(path_write.iterdir())) == 4, "Expect remaining files"

    # Written files should preserve original filenames
    written_dataset = HafniaDataset.from_path(path_write)
    written_paths = written_dataset.samples[SampleField.FILE_PATH].to_list()
    assert len(written_paths) == len(file_paths)
    for written_path in written_paths:
        assert Path(written_path).exists(), f"Expected written file to exist: {written_path}"
