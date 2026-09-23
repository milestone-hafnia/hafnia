"""Registry of public datasets that can be downloaded and converted without platform credentials."""

from typing import Callable, Dict, List

from hafnia.dataset.format_conversions.format_coco_2017 import (
    DATASET_NAME as DATASET_NAME_COCO_2017,
)
from hafnia.dataset.format_conversions.format_coco_2017 import (
    coco_2017_as_hafnia_dataset,
)

# Names of the public datasets that are converted with torchvision. They are listed here - instead of
# read from 'torchvision_to_hafnia_converters()' - to keep the registry usable without torchvision,
# which is an optional dependency. 'test_torchvision_dataset_names_are_up_to_date' guards the list.
TORCHVISION_DATASET_NAMES: List[str] = ["mnist", "cifar10", "cifar100", "caltech-101", "caltech-256"]


def public_dataset_to_hafnia_converters() -> Dict[str, Callable]:
    """Map the name of each supported public dataset to its 'HafniaDataset' converter function.

    Each converter downloads the dataset from its public source, converts it into the Hafnia format
    and accepts the 'force_redownload' and 'n_samples' arguments of
    'HafniaDataset.from_name_public_dataset'.

    The torchvision converters are resolved when they are called, so the datasets that do not need
    torchvision - such as 'coco-2017' - also work without it installed.
    """
    converters: Dict[str, Callable] = {name: torchvision_converter(name) for name in TORCHVISION_DATASET_NAMES}
    converters[DATASET_NAME_COCO_2017] = coco_2017_as_hafnia_dataset
    return converters


def torchvision_converter(name: str) -> Callable:
    """Wrap a torchvision-based converter, so torchvision is first imported when it is used."""

    def converter(*args, **kwargs):
        try:
            from hafnia.dataset.format_conversions.torchvision_datasets import (
                torchvision_to_hafnia_converters,
            )
        except ImportError as error:
            raise ImportError(
                f"The public dataset '{name}' is converted with torchvision, which is not installed. "
                f"Install it with 'uv sync --group dev' or 'pip install torchvision'."
            ) from error
        return torchvision_to_hafnia_converters()[name](*args, **kwargs)

    converter.__name__ = f"{name.replace('-', '_')}_as_hafnia_dataset"
    return converter
