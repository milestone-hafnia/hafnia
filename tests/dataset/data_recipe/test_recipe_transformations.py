from hafnia.dataset.data_recipe.recipe_transformations import (
    DefineSampleSetBySize,
    SelectSamples,
    Shuffle,
    SplitIntoMultipleSplits,
    SplitsByRatios,
)
from hafnia.dataset.dataset_names import ColumnName
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.helper_testing import get_micro_hafnia_dataset


def test_sample_transformation():
    dataset: HafniaDataset = get_micro_hafnia_dataset(dataset_name="tiny-dataset")  # type: ignore[annotation-unchecked]
    n_samples = 2
    sample_transformation = SelectSamples(n_samples=n_samples, shuffle=True, seed=42)

    new_dataset = sample_transformation(dataset)
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

    new_dataset = sample_transformation(dataset)
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

    new_dataset = sample_transformation(dataset)
    assert isinstance(new_dataset, HafniaDataset), "Sampled dataset is not a HafniaDataset instance"
    assert len(new_dataset) == n_samples, (
        f"Sampled dataset length {len(new_dataset)} does not match expected {n_samples}"
    )


def test_shuffle_transformation():
    dataset: HafniaDataset = get_micro_hafnia_dataset(dataset_name="tiny-dataset")  # type: ignore[annotation-unchecked]

    shuffle_transformation = Shuffle(seed=123)
    new_dataset = shuffle_transformation(dataset)

    shuffle_transformation = Shuffle(seed=123)
    new_dataset2 = shuffle_transformation(dataset)
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
    new_dataset = splits_transformation(dataset)
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
    new_dataset = define_sample_transformation(dataset)
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

    split_transformation = SplitIntoMultipleSplits(divide_split_name=divide_split_name, split_ratios=split_ratios)
    new_dataset = split_transformation(dataset)

    expected_split_counts = {"train": int(0.5 * n_samples), "test": int(0.25 * n_samples), "val": int(0.25 * n_samples)}
    actual_split_counts = new_dataset.split_counts()
    assert isinstance(new_dataset, HafniaDataset), "New dataset is not a HafniaDataset instance"
    assert actual_split_counts == expected_split_counts, (
        f"Expected split counts {expected_split_counts}, but got {actual_split_counts}"
    )
