from pathlib import Path
from typing import Any, Optional, Union

from hafnia import utils
from hafnia.dataset.builder.builders import DatasetBuilder, DatasetFromName, DatasetFromPath, DatasetMerger, Transforms


def convert_to_explicit_specification(spec: Any) -> DatasetBuilder:
    """
    Recursively convert implicit dataset specification to explicit form.
    Handles mixed implicit/explicit specifications.

    Conversion rules:
    - str -> DatasetFromName
    - Path -> DatasetFromPath
    - tuple -> DatasetMerger
    - list -> Transforms (first element is the loader, rest are transformations)
    - DatasetBuilder instances (explicit objects) -> returned as-is

    Example: Dataset builder from dataset name:
    ```python
    dataset_specification = "mnist"
    explicit_specification = convert_implicit_dataset_specification_to_explicit(dataset_specification)
    >>> explicit_specification
    DatasetFromName(dataset_name='mnist', force_redownload=False)
    ```

    Example: Dataset builder from tuple (merging multiple datasets):
    ```python
    dataset_specification = ("dataset1", "dataset2")
    explicit_specification = convert_implicit_dataset_specification_to_explicit(dataset_specification)
    >>> explicit_specification
    DatasetMerger(
        builders=[
            DatasetFromName(dataset_name='dataset1', force_redownload=False),
            DatasetFromName(dataset_name='dataset2', force_redownload=False)
        ]
    )

    Example: Dataset builder from list (loader and transformations):
    ```python
    dataset_specification = ["mnist", Sample(n_samples=20), Shuffle(seed=123)]
    explicit_specification = convert_implicit_dataset_specification_to_explicit(dataset_specification)
    >>> explicit_specification
    Transforms(
        loader=DatasetFromName(dataset_name='mnist', force_redownload=False),
        transforms=[Sample(n_samples=20), Shuffle(seed=123)]
    )
    ```

    """
    if isinstance(spec, DatasetBuilder):  # type: ignore
        # It is possible to do an early return if spec is a 'DatasetBuilder'-type even for nested and
        # potentially mixed specifications. If you (really) think about it, this might surprise you,
        # as this will bypass the conversion logic for nested specifications.
        # However, this is not a problem because 'DatasetBuilder' are also pydantic models,
        # so if a user introduces a 'DatasetBuilder'-type in the dataset specification (in potentially
        # some nested and mixed implicit/explicit form) it will (due to pydantic validation) force
        # the user to specify all nested specifications to be converted to explicit form.
        return spec

    if isinstance(spec, str):  # str-type is convert to DatasetFromName
        return DatasetFromName(name=spec)

    if isinstance(spec, Path):  # Path-type is convert to DatasetFromPath
        return DatasetFromPath(path_folder=spec)

    if isinstance(spec, tuple):  # tuple-type is convert to DatasetMerger
        builders = [convert_to_explicit_specification(item) for item in spec]
        return DatasetMerger(builders=builders)

    if isinstance(spec, list):  # list-type is convert to Transforms
        if len(spec) == 0:
            raise ValueError("List specification cannot be empty")

        loader_spec = spec[0]  # First element is the loader specification
        loader = convert_to_explicit_specification(loader_spec)

        transforms = spec[1:]  # Remaining items are transformations
        return Transforms(loader=loader, transforms=transforms)

    raise ValueError(f"Unsupported specification type: {type(spec)}")


def unique_specification_name(specification: DatasetBuilder) -> str:
    if isinstance(specification, DatasetFromName):
        # If the dataset specification is simply a DatasetFromName, we bypass the hashing logic
        # and return the name directly. The dataset is already uniquely identified by its name.
        # Add  version if need... Optionally, you may also completely delete this exception
        # and always return the unique name based on the hash.
        return specification.name
    specification_json_str = specification.model_dump_json()
    hash_specification = utils.hash_from_string(specification_json_str)
    short_specification = specification.short_name()
    unique_name = f"{short_specification}_{hash_specification}"
    return unique_name


def get_dataset_path_from_specification(
    specification: DatasetBuilder,
    path_datasets: Optional[Union[Path, str]] = None,
) -> Path:
    path_datasets = path_datasets or utils.PATH_DATASETS
    path_datasets = Path(path_datasets)
    unique_dataset_name = unique_specification_name(specification)
    return path_datasets / unique_dataset_name
