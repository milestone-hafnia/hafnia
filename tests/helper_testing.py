from inspect import getmembers, isfunction, signature
from pathlib import Path
from types import FunctionType
from typing import Any, Callable, Dict, List, Tuple, Union, get_origin

import hafnia
from hafnia.dataset import primitives
from hafnia.dataset.dataset_names import FILENAME_ANNOTATIONS_JSONL
from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe
from hafnia.dataset.hafnia_dataset import HafniaDataset, Sample

MICRO_DATASETS = {
    "micro-tiny-dataset": "tiny-dataset",
    "micro-coco-2017": "coco-2017",
}


def get_path_workspace() -> Path:
    return Path(__file__).parents[1]


def get_path_expected_images() -> Path:
    return get_path_workspace() / "tests" / "data" / "expected_images"


def get_path_test_data() -> Path:
    return get_path_workspace() / "tests" / "data"


def get_path_micro_hafnia_dataset_no_check() -> Path:
    return get_path_test_data() / "micro_test_datasets"


def get_path_micro_hafnia_dataset(dataset_name: str, force_update=False) -> Path:
    import pytest

    if dataset_name not in MICRO_DATASETS:
        raise ValueError(f"Dataset name '{dataset_name}' is not recognized. Available options: {list(MICRO_DATASETS)}")

    path_test_dataset = get_path_micro_hafnia_dataset_no_check() / dataset_name
    path_test_dataset_annotations = path_test_dataset / FILENAME_ANNOTATIONS_JSONL
    if path_test_dataset_annotations.exists() and not force_update:
        return path_test_dataset

    hafnia_dataset_name = MICRO_DATASETS[dataset_name]
    hafnia_dataset = HafniaDataset.from_name(hafnia_dataset_name, force_redownload=True)
    hafnia_dataset = hafnia_dataset.select_samples(n_samples=3, seed=42)
    hafnia_dataset.write(path_test_dataset)

    format_version_mismatch = hafnia_dataset.info.format_version != hafnia.__dataset_format_version__
    if format_version_mismatch:
        raise ValueError(
            f"You are trying to update the micro test dataset '{dataset_name}' (located in '{path_test_dataset}'), "
            f"with 'force_update=True'. This will re-download '{hafnia_dataset_name}'. "
            f"However, the format version for the re-downloaded dataset ('{hafnia_dataset.info.format_version}'), "
            f"is still not matching the current format version ('{hafnia.__dataset_format_version__}'). "
            f"You will need to recreate '{hafnia_dataset_name}' using the 'data-management' repo to update the "
            f"dataset format version."
        )

    if force_update:
        pytest.fail(
            "Sample image and metadata have been updated using 'force_update=True'. Set 'force_update=False' and rerun the test."
        )
    pytest.fail("Missing test sample image. Please rerun the test.")
    return path_test_dataset


def get_sample_micro_hafnia_dataset(dataset_name: str, force_update=False) -> Sample:
    micro_dataset = get_micro_hafnia_dataset(dataset_name=dataset_name, force_update=force_update)
    sample_dict = micro_dataset[0]
    sample = Sample(**sample_dict)
    return sample


def get_micro_hafnia_dataset(dataset_name: str, force_update: bool = False) -> HafniaDataset:
    path_dataset = get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=force_update)
    hafnia_dataset = HafniaDataset.from_path(path_dataset)
    return hafnia_dataset


def is_typing_type(annotation: Any) -> bool:
    return get_origin(annotation) is not None


def annotation_as_string(annotation: Union[type, str]) -> str:
    """Convert type annotation to string."""
    if isinstance(annotation, str):
        return annotation.replace("'", "")
    if is_typing_type(annotation):  # Is using typing types like List, Dict, etc.
        # This is a simple approach to remove typing annotations as demonstrated below:
        # "typing.List[str]" --> "List[str]"
        # "typing.Optional[typing.Dict[str, int]]" --> "Dict[str, int]"
        # "typing.Optional[typing.Type[hafnia.dataset.primitives.primitive.Primitive]]" --> "Optional[Type[Primitive]]"
        # Add more rules to 'replace_dict' as needed
        # We are using a simple string replacement approach to avoid complex logic or regex converter functions
        # that are hard to debug - when issues appear. Instead we can just add more rules to 'replace_dict'.
        annotation_str = str(annotation)
        replace_dict = {
            "typing.": "",
            "hafnia.dataset.primitives.primitive.": "",
        }

        for key, value in replace_dict.items():
            annotation_str = annotation_str.replace(key, value)
        if "." in annotation_str:
            raise ValueError(
                f"Could not convert annotation '{annotation}' to string. "
                f"Found '.' in '{annotation_str}'. Add replace rules to 'replace_dict'."
            )
        return annotation_str
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def get_hafnia_functions_from_module(python_module) -> Dict[str, FunctionType]:
    def dataset_is_first_arg(func: Callable) -> bool:
        """
        Check if the function has 'HafniaDataset' as the first parameter.
        """
        func_signature = signature(func)
        params = func_signature.parameters
        if len(params) == 0:
            return False
        first_argument_type = list(params.values())[0]

        annotation_as_str = annotation_as_string(first_argument_type.annotation)
        return annotation_as_str == "HafniaDataset"

    functions = {func[0]: func[1] for func in getmembers(python_module, isfunction) if dataset_is_first_arg(func[1])}
    return functions


def get_dummy_recipe() -> DatasetRecipe:
    dataset_recipe = (
        DatasetRecipe.from_merger(
            recipes=[
                DatasetRecipe.from_name(name="mnist", force_redownload=False)
                .select_samples(n_samples=20, shuffle=True, seed=42)
                .shuffle(seed=123),
                DatasetRecipe.from_name(name="mnist", force_redownload=False)
                .select_samples(n_samples=30, shuffle=True, seed=42)
                .splits_by_ratios(split_ratios={"train": 0.8, "val": 0.1, "test": 0.1}, seed=42),
                DatasetRecipe.from_name(name="mnist", force_redownload=False),
            ]
        )
        .class_mapper(get_strict_class_mapping_mnist())
        .rename_task(old_task_name=primitives.Classification.default_task_name(), new_task_name="digits")
        .select_samples_by_class_name(name=["odd"])
    )

    return dataset_recipe


def get_strict_class_mapping_midwest() -> Dict[str, str]:
    strict_class_mapping = {
        "Person": "person",  # Index 0
        "Vehicle.Trailer": "__REMOVE__",  # Removed not provided an index
        "Vehicle.Bicycle": "__REMOVE__",
        "Vehicle.Motorcycle": "vehicle",  # Index 1
        "Vehicle.Car": "vehicle",
        "Vehicle.Van": "vehicle",
        "Vehicle.RV": "__REMOVE__",
        "Vehicle.Single_Truck": "truck",  # Index 2
        "Vehicle.Combo_Truck": "__REMOVE__",
        "Vehicle.Pickup_Truck": "truck",
        "Vehicle.Emergency_Vehicle": "vehicle",
        "Vehicle.Bus": "vehicle",
        "Vehicle.Heavy_Duty_Vehicle": "vehicle",
    }
    return strict_class_mapping


def get_strict_class_mapping_mnist() -> Dict[str, str]:
    strict_class_mapping = {
        "0 - zero": "even",  # "0 - zero" will be renamed to "even". "even" appear first and get class index 0
        "1 - one": "odd",  # "1 - one" will be renamed to "odd". "odd" appear second and will get class index 1
        "2 - two": "even",
        "3 - three": "odd",
        "4 - four": "even",
        "5 - five": "odd",
        "6 - six": "even",
        "7 - seven": "odd",
        "8 - eight": "even",
        "9 - nine": "__REMOVE__",  # Remove all samples with class "9 - nine"
    }
    return strict_class_mapping


def dict_as_list_of_tuples(mapping: Dict[str, str]) -> List[Tuple[str, str]]:
    return [(key, value) for key, value in mapping.items()]
