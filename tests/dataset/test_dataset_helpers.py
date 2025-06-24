import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from hafnia.dataset import dataset_helpers
from hafnia.dataset.dataset_names import SplitName


@dataclass
class CreateSplitNameListFromRatiosTestCase:
    split_ratio: Dict[str, float]
    n_items: int
    expected_lengths: Dict[str, int]


@pytest.mark.parametrize(
    "test_case",
    [
        CreateSplitNameListFromRatiosTestCase(
            split_ratio={SplitName.TRAIN: 0.7, SplitName.TEST: 0.2, SplitName.VAL: 0.1},
            n_items=100,
            expected_lengths={SplitName.TRAIN: 70, SplitName.TEST: 20, SplitName.VAL: 10},
        ),
        CreateSplitNameListFromRatiosTestCase(
            split_ratio={SplitName.TRAIN: 0.70, SplitName.TEST: 0.2, SplitName.VAL: 0.1},
            n_items=1001,
            expected_lengths={SplitName.TRAIN: 701, SplitName.TEST: 200, SplitName.VAL: 100},
        ),
        CreateSplitNameListFromRatiosTestCase(
            split_ratio={SplitName.TRAIN: 0.333, SplitName.TEST: 0.333, SplitName.VAL: 0.333},
            n_items=1002,
            expected_lengths={SplitName.TRAIN: 334, SplitName.TEST: 334, SplitName.VAL: 334},
        ),
        CreateSplitNameListFromRatiosTestCase(
            split_ratio={SplitName.TRAIN: 0.333, SplitName.TEST: 0.333, SplitName.VAL: 0.333},
            n_items=103,
            expected_lengths={SplitName.TRAIN: 35, SplitName.TEST: 34, SplitName.VAL: 34},
        ),
        CreateSplitNameListFromRatiosTestCase(
            split_ratio={SplitName.TRAIN: 0.5, SplitName.TEST: 0.3, SplitName.VAL: 0.2},
            n_items=200,
            expected_lengths={SplitName.TRAIN: 100, SplitName.TEST: 60, SplitName.VAL: 40},
        ),
        CreateSplitNameListFromRatiosTestCase(
            split_ratio={SplitName.TEST: 0.5, SplitName.VAL: 0.5},
            n_items=101,
            expected_lengths={SplitName.TEST: 51, SplitName.VAL: 50},
        ),
        CreateSplitNameListFromRatiosTestCase(
            split_ratio={SplitName.TEST: 0.5, SplitName.VAL: 0.5},
            n_items=100,
            expected_lengths={SplitName.TEST: 50, SplitName.VAL: 50},
        ),
    ],
)
def test_create_split_name_list_from_ratios(test_case: CreateSplitNameListFromRatiosTestCase):
    split_names = dataset_helpers.create_split_name_list_from_ratios(
        split_ratios=test_case.split_ratio,
        n_items=test_case.n_items,
    )

    n_changes_in_split_name = sum([c_name != n_name for c_name, n_name in zip(split_names[:-1], split_names[1:])])
    split_names_have_been_shuffled = n_changes_in_split_name > 3
    assert split_names_have_been_shuffled
    assert len(split_names) == test_case.n_items
    assert dict(collections.Counter(split_names)) == test_case.expected_lengths


def test_save_image_with_hash_name(tmp_path: Path):
    dummy_image = (255 * np.random.rand(100, 100, 3)).astype(np.uint8)  # Create a dummy image
    path_image = dataset_helpers.save_image_with_hash_name(dummy_image, tmp_path)
    filename_from_path = dataset_helpers.filename_as_hash_from_path(path_image)
    assert filename_from_path == path_image.name
    assert path_image.exists()
    assert path_image.suffix in [".png"]
