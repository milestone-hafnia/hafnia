import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from hafnia.dataset.dataset_names import ColumnName
from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe, FromName
from hafnia.dataset.dataset_recipe.recipe_transforms import (
    DefineSampleSetBySize,
    SelectSamples,
    Shuffle,
    SplitIntoMultipleSplits,
    SplitsByRatios,
)
from hafnia.dataset.dataset_recipe.recipe_types import RecipeTransform
from hafnia.dataset.hafnia_dataset import HafniaDataset
from tests.helper_testing import get_micro_hafnia_dataset


@dataclass
class TestCaseRecipeTransform:
    recipe_transform: RecipeTransform
    as_python_code: str
    short_name: str

    def as_dataset_recipe(self) -> DatasetRecipe:
        return DatasetRecipe(creation=FromName(name="test"), operations=[self.recipe_transform])


def get_test_cases() -> list[TestCaseRecipeTransform]:
    return [
        TestCaseRecipeTransform(
            recipe_transform=SelectSamples(n_samples=10, shuffle=True, seed=42),
            as_python_code="select_samples(n_samples=10, shuffle=True, seed=42, with_replacement=False)",
            short_name="SelectSamples",
        ),
        TestCaseRecipeTransform(
            recipe_transform=Shuffle(seed=123),
            as_python_code="shuffle(seed=123)",
            short_name="Shuffle",
        ),
        TestCaseRecipeTransform(
            recipe_transform=SplitsByRatios(split_ratios={"train": 0.5, "val": 0.25, "test": 0.25}, seed=42),
            as_python_code="splits_by_ratios(split_ratios={'train': 0.5, 'val': 0.25, 'test': 0.25}, seed=42)",
            short_name="SplitsByRatios",
        ),
        TestCaseRecipeTransform(
            recipe_transform=DefineSampleSetBySize(n_samples=100),
            as_python_code="define_sample_set_by_size(n_samples=100, seed=42)",
            short_name="DefineSampleSetBySize",
        ),
        TestCaseRecipeTransform(
            recipe_transform=SplitIntoMultipleSplits(split_name="test", split_ratios={"test": 0.5, "val": 0.5}),
            as_python_code="split_into_multiple_splits(split_name='test', split_ratios={'test': 0.5, 'val': 0.5})",
            short_name="SplitIntoMultipleSplits",
        ),
    ]


def test_check_that_all_recipe_transforms_has_a_test_case():
    """
    Ensure that all recipe transformations are tested.
    This is useful to ensure that new transformations are added to the test suite.
    """
    in_test_recipe_transforms = {tc.recipe_transform.__class__ for tc in get_test_cases()}
    recipe_transforms = set(RecipeTransform.get_nested_subclasses())

    transforms_missing_tests = recipe_transforms.difference(in_test_recipe_transforms)
    missing_transforms = {tr.__name__ for tr in transforms_missing_tests}
    error_msg = (
        f"We expect all recipe transformations to have a test case, but have found '{RecipeTransform.__name__}' "
        f"classes/subclasses that are not tested. \nPlease add a '{TestCaseRecipeTransform.__name__}' "
        f"for the {missing_transforms=} in the list of test cases found in '{get_test_cases.__name__}()' "
        "to ensure they are tested."
    )

    assert len(transforms_missing_tests) == 0, error_msg


@pytest.mark.parametrize("test_case", get_test_cases(), ids=lambda tc: tc.as_python_code)
def test_cases_serialization_deserialization_of_recipe_transform(test_case: TestCaseRecipeTransform):
    dataset_recipe = test_case.as_dataset_recipe()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
        path_json = Path(tmp_file.name)
        dataset_recipe.as_json_file(path_json)
        dataset_recipe_again = DatasetRecipe.from_json_file(path_json)

    assert dataset_recipe_again == dataset_recipe


@pytest.mark.parametrize("test_case", get_test_cases(), ids=lambda tc: tc.as_python_code)
def test_cases_as_python_code(test_case: TestCaseRecipeTransform):
    """
    Test that the `as_python_code` method of the recipe transformation returns the expected string representation.
    """
    code_str = test_case.recipe_transform.as_python_code(keep_default_fields=True, as_kwargs=True)

    assert code_str == test_case.as_python_code


@pytest.mark.parametrize("test_case", get_test_cases(), ids=lambda tc: tc.as_python_code)
def test_cases_as_short_name(test_case: TestCaseRecipeTransform):
    """
    Test that the `as_short_name` method of the recipe transformation returns the expected string representation.
    """
    short_name = test_case.recipe_transform.as_short_name()

    assert short_name == test_case.short_name


def test_sample_transformation():
    dataset: HafniaDataset = get_micro_hafnia_dataset(dataset_name="tiny-dataset")  # type: ignore[annotation-unchecked]
    n_samples = 2
    sample_transformation = SelectSamples(n_samples=n_samples, shuffle=True, seed=42)

    new_dataset = sample_transformation.build(dataset)
    assert isinstance(new_dataset, HafniaDataset), "Sampled dataset is not a HafniaDataset instance"
    assert len(new_dataset) == n_samples, (
        f"Sampled dataset length {len(new_dataset)} does not match expected {n_samples}"
    )


def test_sample_transformation_without_replacement():
    """Without replacement, the number of samples should not exceed the actual dataset size."""
    dataset: HafniaDataset = get_micro_hafnia_dataset(dataset_name="tiny-dataset")  # type: ignore[annotation-unchecked]

    max_actual_number_of_samples = len(dataset)
    n_samples = 100  # A micro dataset is only 3 samples, so this should be capped to 3
    sample_transformation = SelectSamples(n_samples=n_samples, shuffle=True, seed=42, with_replacement=False)

    new_dataset = sample_transformation.build(dataset)
    assert isinstance(new_dataset, HafniaDataset), "Sampled dataset is not a HafniaDataset instance"
    assert len(new_dataset) == max_actual_number_of_samples, (
        f"Sampled dataset length {len(new_dataset)} does not match expected {max_actual_number_of_samples}"
    )


def test_sample_transformation_with_replacement():
    """With replacement, the number of samples can exceed the actual dataset size."""
    dataset: HafniaDataset = get_micro_hafnia_dataset(dataset_name="tiny-dataset")  # type: ignore[annotation-unchecked]

    assert len(dataset) < 100, "The micro dataset should have less than 100 samples for this test to be valid"
    n_samples = 100  # The micro dataset is only 3 samples. With_replacement=True it will duplicate samples
    sample_transformation = SelectSamples(n_samples=n_samples, shuffle=True, seed=42, with_replacement=True)

    new_dataset = sample_transformation.build(dataset)
    assert isinstance(new_dataset, HafniaDataset), "Sampled dataset is not a HafniaDataset instance"
    assert len(new_dataset) == n_samples, (
        f"Sampled dataset length {len(new_dataset)} does not match expected {n_samples}"
    )


def test_shuffle_transformation():
    dataset: HafniaDataset = get_micro_hafnia_dataset(dataset_name="tiny-dataset")  # type: ignore[annotation-unchecked]

    shuffle_transformation = Shuffle(seed=123)
    new_dataset = shuffle_transformation.build(dataset)

    shuffle_transformation = Shuffle(seed=123)
    new_dataset2 = shuffle_transformation.build(dataset)
    is_same = all(new_dataset.samples[ColumnName.SAMPLE_INDEX] == new_dataset2.samples[ColumnName.SAMPLE_INDEX])
    assert is_same, "Shuffled datasets should be equal with the same seed"

    is_same = all(new_dataset.samples[ColumnName.SAMPLE_INDEX] == dataset.samples[ColumnName.SAMPLE_INDEX])
    assert not is_same, "Shuffled dataset should not match original dataset"
    assert isinstance(new_dataset, HafniaDataset), "Shuffled dataset is not a HafniaDataset instance"
    assert len(new_dataset) == len(dataset), (
        f"Shuffled dataset length {len(new_dataset)} does not match original {len(dataset)}"
    )


def test_splits_by_ratios_transformation():
    dataset: HafniaDataset = get_micro_hafnia_dataset(dataset_name="tiny-dataset")  # type: ignore[annotation-unchecked]
    # The micro dataset is small, so we duplicate samples up to 100 samples for testing
    dataset = dataset.select_samples(n_samples=100, seed=42, with_replacement=True)

    split_ratios = {"train": 0.5, "val": 0.25, "test": 0.25}
    splits_transformation = SplitsByRatios(split_ratios=split_ratios, seed=42)
    new_dataset = splits_transformation.build(dataset)
    assert isinstance(new_dataset, HafniaDataset), "Splits dataset is not a HafniaDataset instance"

    actual_split_counts = new_dataset.split_counts()
    expected_split_count = {name: int(ratio * len(dataset)) for name, ratio in split_ratios.items()}
    assert actual_split_counts == expected_split_count


def test_define_sample_by_size_transformation():
    n_samples = 100
    dataset: HafniaDataset = get_micro_hafnia_dataset(dataset_name="tiny-dataset")  # type: ignore[annotation-unchecked]
    # The micro dataset is small, so we duplicate samples up to 100 samples for testing
    dataset = dataset.select_samples(n_samples=n_samples, seed=42, with_replacement=True)

    define_sample_transformation = DefineSampleSetBySize(n_samples=n_samples)
    new_dataset = define_sample_transformation.build(dataset)
    assert isinstance(new_dataset, HafniaDataset), "Sampled dataset is not a HafniaDataset instance"

    assert new_dataset.samples[ColumnName.IS_SAMPLE].sum() == n_samples, (
        f"Sampled dataset should have {n_samples} samples, but has {new_dataset.samples[ColumnName.IS_SAMPLE].sum()}"
    )


def test_split_into_multiple_splits():
    n_samples = 100
    dataset: HafniaDataset = get_micro_hafnia_dataset(dataset_name="tiny-dataset")  # type: ignore[annotation-unchecked]
    # The micro dataset is small, so we duplicate samples up to 100 samples for testing
    dataset = dataset.select_samples(n_samples=n_samples, seed=42, with_replacement=True)
    dataset = dataset.splits_by_ratios(split_ratios={"train": 0.5, "test": 0.5}, seed=42)

    divide_split_name = "test"
    split_ratios = {"test": 0.5, "val": 0.5}

    # Create a test split in the dataset

    split_transformation = SplitIntoMultipleSplits(split_name=divide_split_name, split_ratios=split_ratios)
    new_dataset = split_transformation.build(dataset)

    expected_split_counts = {"train": int(0.5 * n_samples), "test": int(0.25 * n_samples), "val": int(0.25 * n_samples)}
    actual_split_counts = new_dataset.split_counts()
    assert isinstance(new_dataset, HafniaDataset), "New dataset is not a HafniaDataset instance"
    assert actual_split_counts == expected_split_counts, (
        f"Expected split counts {expected_split_counts}, but got {actual_split_counts}"
    )
