import math
import random
from typing import Dict, List


def create_split_name_list_from_ratios(split_ratios: Dict[str, float], n_items: int, seed: int = 42) -> List[str]:
    samples_per_split = split_sizes_from_ratios(split_ratios=split_ratios, n_items=n_items)

    split_name_column = []
    for split_name, n_split_samples in samples_per_split.items():
        split_name_column.extend([split_name] * n_split_samples)
    random.Random(seed).shuffle(split_name_column)  # Shuffle the split names

    return split_name_column


def split_sizes_from_ratios(n_items: int, split_ratios: Dict[str, float]) -> Dict[str, int]:
    summed_ratios = sum(split_ratios.values())
    abs_tols = 0.0011  # Allow some tolerance for floating point errors {"test": 0.333, "val": 0.333, "train": 0.333}
    if not math.isclose(summed_ratios, 1.0, abs_tol=abs_tols):  # Allow tolerance to allow e.g. (0.333, 0.333, 0.333)
        raise ValueError(f"Split ratios must sum to 1.0. The summed values of {split_ratios} is {summed_ratios}")

    # recaculate split sizes
    split_ratios = {split_name: split_ratio / summed_ratios for split_name, split_ratio in split_ratios.items()}
    split_sizes = {split_name: int(n_items * split_ratio) for split_name, split_ratio in split_ratios.items()}

    remaining_items = n_items - sum(split_sizes.values())
    if remaining_items > 0:  # Distribute remaining items evenly across splits
        for _ in range(remaining_items):
            # Select name by the largest error from the expected distribution
            total_size = sum(split_sizes.values())
            distribution_error = {
                split_name: abs(split_ratios[split_name] - (size / total_size))
                for split_name, size in split_sizes.items()
            }

            split_with_largest_error = sorted(distribution_error.items(), key=lambda x: x[1], reverse=True)[0][0]
            split_sizes[split_with_largest_error] += 1

    if sum(split_sizes.values()) != n_items:
        raise ValueError("Something is wrong. The split sizes do not match the number of items.")

    return split_sizes


def select_evenly_across_list(lst: list, num_samples: int):
    if num_samples >= len(lst):
        return lst  # No need to sample
    step = (len(lst) - 1) / (num_samples - 1)
    indices = [int(round(step * i)) for i in range(num_samples)]  # noqa: RUF046
    return [lst[index] for index in indices]


def prefix_dict(d: dict, prefix: str) -> dict:
    return {f"{prefix}.{k}": v for k, v in d.items()}
