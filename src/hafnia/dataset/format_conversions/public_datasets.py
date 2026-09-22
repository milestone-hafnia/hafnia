"""Registry of public datasets that can be downloaded and converted without platform credentials."""

from typing import Callable, Dict


def public_dataset_to_hafnia_converters() -> Dict[str, Callable]:
    """Map the name of each supported public dataset to its 'HafniaDataset' converter function.

    Each converter downloads the dataset from its public source, converts it into the Hafnia format
    and accepts the 'force_redownload' and 'n_samples' arguments of
    'HafniaDataset.from_name_public_dataset'.
    """
    from hafnia.dataset.format_conversions.format_coco_2017 import (
        DATASET_NAME as DATASET_NAME_COCO_2017,
    )
    from hafnia.dataset.format_conversions.format_coco_2017 import (
        coco_2017_as_hafnia_dataset,
    )
    from hafnia.dataset.format_conversions.torchvision_datasets import (
        torchvision_to_hafnia_converters,
    )

    return {
        **torchvision_to_hafnia_converters(),
        DATASET_NAME_COCO_2017: coco_2017_as_hafnia_dataset,
    }
