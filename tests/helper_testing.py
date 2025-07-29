from inspect import getmembers, isfunction, signature
from pathlib import Path
from types import FunctionType
from typing import Any, Callable, Dict, Union, get_origin

from hafnia import utils
from hafnia.dataset.dataset_names import FILENAME_ANNOTATIONS_JSONL, DatasetVariant
from hafnia.dataset.hafnia_dataset import HafniaDataset, Sample

MICRO_DATASETS = {
    "tiny-dataset": utils.PATH_DATASETS / "tiny-dataset",
    "coco-2017": utils.PATH_DATASETS / "coco-2017",
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
    path_dataset = MICRO_DATASETS[dataset_name]

    path_test_dataset = get_path_micro_hafnia_dataset_no_check() / dataset_name
    path_test_dataset_annotations = path_test_dataset / FILENAME_ANNOTATIONS_JSONL
    if path_test_dataset_annotations.exists() and not force_update:
        return path_test_dataset

    hafnia_dataset = HafniaDataset.from_path(path_dataset / DatasetVariant.SAMPLE.value)
    hafnia_dataset = hafnia_dataset.select_samples(n_samples=3, seed=42)
    hafnia_dataset.write(path_test_dataset)

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


def is_hafnia_configured() -> bool:
    """
    Check if Hafnia is configured by verifying if the API key is set.
    """
    from cli.config import Config

    return Config().is_configured()


def is_typing_type(annotation: Any) -> bool:
    return get_origin(annotation) is not None


def annotation_as_string(annotation: Union[type, str]) -> str:
    """Convert type annotation to string."""
    if isinstance(annotation, str):
        return annotation.replace("'", "")
    if is_typing_type(annotation):  # Is using typing types like List, Dict, etc.
        return str(annotation).replace("typing.", "")
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
