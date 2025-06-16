import random
from typing import Dict, List

from hafnia.dataset.dataset_names import SplitName


def split_names_from_ratios(split_ratios: Dict[str, float], n_items: int, seed: int = 42) -> List[str]:
    samples_per_split = split_sizes_from_ratios(split_ratios=split_ratios, n_items=n_items)

    split_name_column = []
    for split_name, n_split_samples in samples_per_split.items():
        split_name_column.extend([split_name] * n_split_samples)
    random.Random(seed).shuffle(split_name_column)  # Shuffle the split names to avoid bias in the splits

    return split_name_column


def split_sizes_from_ratios(n_items: int, split_ratios: Dict[str, float]) -> Dict[str, int]:
    split_sizes = {split_name: int(n_items * split_ratio) for split_name, split_ratio in split_ratios.items()}
    if sum(split_ratios.values()) == 1.0:
        split_sizes[SplitName.TRAIN] = split_sizes[SplitName.TRAIN] + n_items - sum(split_sizes.values())
    return split_sizes


def select_evenly_across_list(lst: list, num_samples: int):
    if num_samples >= len(lst):
        return lst  # No need to sample
    step = (len(lst) - 1) / (num_samples - 1)
    indices = [int(round(step * i)) for i in range(num_samples)]  # noqa: RUF046
    return [lst[index] for index in indices]


def prefix_dict(d: dict, prefix: str) -> dict:
    return {f"{prefix}.{k}": v for k, v in d.items()}
