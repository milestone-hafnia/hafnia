from contextlib import contextmanager

import mdi
import pytest

SKIP = True


def is_mdi_profile_missing(profile: str) -> bool:
    return profile not in mdi.data.config.list_profiles()


DATASETS = ["mnist", "midwest-extension", "easyportrait", "tiny-dataset"]


@contextmanager
def set_mdi_profile(profile: str) -> None:
    """
    Ensures that the current profile is set to the desired profile
    and then resets it back to the original profile after the test.
    """
    current_profile = mdi.data.config.get_current_profile_name()
    if current_profile is None:
        raise ValueError(
            "We can only return profile to the original profile if it exists. "
            "Set profile `mdi login` to some desired profile before running 'set_mdi_profile'. "
        )
    mdi.data.config.set_current_profile(profile)
    yield

    mdi.data.config.set_current_profile(current_profile)


@pytest.mark.parametrize("dataset_name", DATASETS)
def test_load_dataset(dataset_name: str) -> None:
    mdi_profile = "staging"
    if SKIP:
        pytest.skip(
            "Skipping test deliberately with `SKIP=True`. "
            "Loading all datasets is generally too slow for a test. "
            "Enable dataset loading by setting `SKIP=False`."
        )
    if is_mdi_profile_missing(mdi_profile):
        pytest.skip(
            f"Skipping test because '{mdi_profile}' profile is "
            "not available. To run this test, please create the profile by "
            f"running 'mdi login' to create a `{mdi_profile}`."
        )

    with set_mdi_profile(mdi_profile):
        dataset = mdi.load_dataset(dataset_name, force_redownload=True)
    assert dataset is not None
