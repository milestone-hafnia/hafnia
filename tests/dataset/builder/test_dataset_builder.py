from pathlib import Path

import pytest

from hafnia.dataset.builder.builders import (
    DatasetBuilder,
    DatasetFromName,
    DatasetFromPath,
    DatasetMerger,
    Serializable,
    SerializableFunction,
    Transforms,
)
from hafnia.dataset.builder.dataset_transformations import Sample, Shuffle, SplitsByRatios
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.operations import dataset_transformations
from hafnia.helper_testing import get_hafnia_functions_from_module
from hafnia.utils import snake_to_pascal_case


def get_dataset_specification() -> DatasetBuilder:
    dataset_specification = DatasetMerger(
        builders=[
            Transforms(
                loader=DatasetFromName(name="mnist", force_redownload=False),
                transforms=[
                    Sample(n_samples=20, shuffle=True, seed=42),
                    Shuffle(seed=123),
                ],
            ),
            Transforms(
                loader=DatasetFromName(name="mnist", force_redownload=False),
                transforms=[
                    Sample(n_samples=30, shuffle=True, seed=42),
                    SplitsByRatios(split_ratios={"train": 0.8, "val": 0.1, "test": 0.1}, seed=42),
                ],
            ),
            DatasetFromName(name="mnist", force_redownload=False),
        ]
    )

    return dataset_specification


@pytest.mark.parametrize("serializable_function", SerializableFunction.get_nested_subclasses())
def test_serializable_functions_check_signature(serializable_function: SerializableFunction):
    """
    SerializableFunction converts a function into a serializable class.
    It ensures that the function signature is the same as the expected model fields.
    """
    serializable_function.check_signature()  # Ensures that function signatures match the expected model fields


@pytest.mark.parametrize("transformation_function_name", get_hafnia_functions_from_module(dataset_transformations))
def test_check_dataset_transformations_have_builders(transformation_function_name: str):
    """
    Ensure that all dataset transformations have a corresponding SerializableFunction.
    """
    skip_list = ["transform_images", "merge"]
    if transformation_function_name in skip_list:
        pytest.skip(f"Skipping {transformation_function_name} as it is not a SerializableFunction.")

    function_name_pascal_case = snake_to_pascal_case(transformation_function_name)
    serializable_functions = list(SerializableFunction.name_to_type_mapping())

    expected_class_name = f"class {function_name_pascal_case}({SerializableFunction.__name__}):"
    assert function_name_pascal_case in serializable_functions, (
        f"Transformation function '{transformation_function_name}' in 'operations/dataset_transformations.py' does not have "
        f"a corresponding '{expected_class_name}' in 'builder/dataset_transformations.py'. \n"
        " We expect all functions in 'operations/dataset_transformations.py' to have a 'corresponding' class in "
        f"'builder/dataset_transformations.py'.\n"
        f"Please add '{expected_class_name}' class for it in 'builder/dataset_transformations.py'."
    )


def test_dataset_specification_serialization_deserialization_dict():
    """
    Test that Serializable can be serialized and deserialized correctly.
    """
    dataset_specification = get_dataset_specification()

    # Smoke test - it can be serialized
    serialized_data: dict = dataset_specification.model_dump()  # type: ignore[annotation-unchecked]

    # Smoke test - it can be deserialized
    deserialized_specification = Serializable.from_dict(serialized_data)

    assert isinstance(deserialized_specification, DatasetBuilder)  # type: ignore[misc]
    assert deserialized_specification == dataset_specification, "Deserialized specification does not match original"


def test_dataset_specification_serialization_deserialization_json():
    """
    Test that Serializable can be serialized and deserialized correctly.
    """
    dataset_specification = get_dataset_specification()

    # Smoke test - it can be serialized
    serialized_data: str = dataset_specification.model_dump_json()

    # Smoke test - it can be deserialized
    deserialized_specification = Serializable.from_json_str(serialized_data)

    assert isinstance(deserialized_specification, DatasetBuilder)  # type: ignore[misc]
    assert deserialized_specification == dataset_specification, "Deserialized specification does not match original"


@pytest.mark.parametrize(
    "dataset_builder",
    [
        DatasetFromName(name="mnist", force_redownload=False),
        DatasetFromPath(path_folder=Path(".data/datasets/mnist"), check_for_images=False),
        DatasetMerger(
            builders=[
                DatasetFromName(name="mnist", force_redownload=False),
                DatasetFromPath(path_folder=Path(".data/datasets/mnist"), check_for_images=False),
            ]
        ),
        Transforms(
            loader=DatasetFromName(name="mnist", force_redownload=False),
            transforms=[
                Sample(n_samples=20, shuffle=True, seed=42),
                Shuffle(seed=123),
            ],
        ),
    ],
)
def test_run_dataset_builders(dataset_builder: DatasetBuilder):
    """
    Test that LoadDataset builder can be created and serialized.
    """

    # Ensure that the dataset builder can be called and returns a HafniaDataset
    dataset: HafniaDataset = dataset_builder()
    assert isinstance(dataset, HafniaDataset), "Dataset is not an instance of HafniaDataset"
