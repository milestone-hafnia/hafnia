import io
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import xxhash
from PIL import Image


def create_split_name_list_from_ratios(split_ratios: Dict[str, float], n_items: int, seed: int = 42) -> List[str]:
    samples_per_split = split_sizes_from_ratios(split_ratios=split_ratios, n_items=n_items)

    split_name_column = []
    for split_name, n_split_samples in samples_per_split.items():
        split_name_column.extend([split_name] * n_split_samples)
    random.Random(seed).shuffle(split_name_column)  # Shuffle the split names

    return split_name_column


def hash_file_xxhash(path: Path, chunk_size: int = 262144) -> str:
    hasher = xxhash.xxh3_128()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):  # 8192, 16384, 32768, 65536
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_from_bytes(data: bytes) -> str:
    hasher = xxhash.xxh3_128()
    hasher.update(data)
    return hasher.hexdigest()


def save_image_with_hash_name(image: np.ndarray, path_folder: Path) -> Path:
    pil_image = Image.fromarray(image)
    path_image = save_pil_image_with_hash_name(pil_image, path_folder)
    return path_image


def save_pil_image_with_hash_name(image: Image.Image, path_folder: Path, allow_skip: bool = True) -> Path:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    hash_value = hash_from_bytes(buffer.getvalue())
    path_image = Path(path_folder) / relative_path_from_hash(hash=hash_value, suffix=".png")
    if allow_skip and path_image.exists():
        return path_image
    path_image.parent.mkdir(parents=True, exist_ok=True)
    image.save(path_image)
    return path_image


def copy_and_rename_file_to_hash_value(path_source: Path, path_dataset_root: Path) -> Path:
    """
    Copies a file to a dataset root directory with a hash-based name and sub-directory structure.
    """

    if not path_source.exists():
        raise FileNotFoundError(f"Source file {path_source} does not exist.")

    hash_value = hash_file_xxhash(path_source)
    path_file = path_dataset_root / relative_path_from_hash(hash=hash_value, suffix=path_source.suffix)
    path_file.parent.mkdir(parents=True, exist_ok=True)
    if not path_file.exists():
        shutil.copy2(path_source, path_file)

    return path_file


def relative_path_from_hash(hash: str, suffix: str) -> Path:
    path_file = Path("data") / f"{hash}{suffix}"
    return path_file


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
