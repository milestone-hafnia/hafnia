from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from hafnia.dataset.dataset_names import ColumnName

if TYPE_CHECKING:
    from hafnia.dataset.hafnia_dataset import HafniaDataset


def split_counts(dataset: HafniaDataset) -> Dict[str, int]:
    """
    Returns a dictionary with the counts of samples in each split of the dataset.
    """
    return dict(dataset.samples[ColumnName.SPLIT].value_counts().iter_rows())
