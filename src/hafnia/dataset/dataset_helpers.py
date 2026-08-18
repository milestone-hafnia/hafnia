import io
import math
import random
import shutil
import tempfile
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xxhash
from packaging.version import InvalidVersion, Version
from PIL import Image

from hafnia.log import user_logger


class FileStorageMode(str, Enum):
    """How image/video files are placed at a new location when a dataset is written or exported.

    - `COPY` (default): Store real copies of the files. The written dataset is self-contained, but
        a dataset is duplicated on disk - for large datasets this can be 100s of GBs.
    - `SYMLINK`: Store symbolic links pointing at the original files. Disk usage is negligible, but
        the written dataset is only valid as long as the original files stay in place. Note that
        symlinks come with caveats: they break if the source files are moved/deleted, they are not
        preserved by all archiving/upload tools, they may resolve to the wrong location if either
        source or destination is moved to another machine or mounted at another path, and on
        Windows they require Developer Mode or administrator privileges.
    """

    COPY = "copy"
    SYMLINK = "symlink"


def resolve_storage_mode(
    storage_mode: Union["FileStorageMode", str],
    path_output: Optional[Path] = None,
) -> "FileStorageMode":
    """Normalize a user-provided storage mode and verify up-front that it is usable.

    For `FileStorageMode.SYMLINK` a warning is issued and - when `path_output` is provided - symlink
    support is checked on the target filesystem, so that an export fails immediately instead of
    halfway through writing files.

    Args:
        storage_mode: Storage mode as a `FileStorageMode` or its string value ("copy"/"symlink").
        path_output: Optional output folder used to check symlink support. The folder is created if
            it does not exist.

    Returns:
        The normalized `FileStorageMode`.
    """
    storage_mode = FileStorageMode(storage_mode)
    if storage_mode is FileStorageMode.SYMLINK:
        user_logger.warning(
            "Storing files as symbolic links. No file data is duplicated, but the written dataset "
            "will break if the original files are moved or deleted."
        )
        if path_output is not None:
            check_symlink_support(path_output)
    return storage_mode


def check_symlink_support(path_folder: Path) -> None:
    """Check that symbolic links can be created in `path_folder` and raise a helpful error if not."""
    path_folder = Path(path_folder)
    path_folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path_folder) as path_tmp_dir:
        path_source = Path(path_tmp_dir) / "symlink_check_source"
        path_source.touch()
        path_link = Path(path_tmp_dir) / "symlink_check_link"
        try:
            path_link.symlink_to(path_source)
        except OSError as e:
            raise OSError(
                f"Unable to create symbolic links in '{path_folder}': {e}. "
                "On Windows, symbolic links require Developer Mode to be enabled "
                "('Settings > System > For developers') or an elevated (administrator) process. "
                f"Use storage_mode='{FileStorageMode.COPY.value}' to store real copies of the files instead."
            ) from e


def store_file(
    path_source: Path,
    path_destination: Path,
    storage_mode: Union[FileStorageMode, str] = FileStorageMode.COPY,
    allow_skip: bool = True,
) -> Path:
    """Store a file (image/video) at a new location as a real copy or as a symbolic link.

    Args:
        path_source: Existing source file.
        path_destination: Destination path. Parent folders are created if missing.
        storage_mode: `FileStorageMode.COPY` (default) to copy the file data or
            `FileStorageMode.SYMLINK` to create a symbolic link pointing at `path_source`. The
            string values `"copy"` and `"symlink"` are also accepted.
        allow_skip: If True (default), keep an already existing destination file as is.

    Returns:
        The destination path.

    Raises:
        FileNotFoundError: If `path_source` does not exist. Without this check, symlink mode would
            silently create a broken symlink, as 'Path.resolve()' also resolves missing paths.
    """
    storage_mode = FileStorageMode(storage_mode)
    path_source = Path(path_source)
    path_destination = Path(path_destination)

    if not path_source.exists():  # 'exists' is False for both missing files and broken symlinks
        raise FileNotFoundError(f"Source file {path_source} does not exist.")

    # 'exists' follows symlinks, so a broken symlink is not a usable destination and is never skipped.
    # Re-running an export into an existing folder will therefore repair broken links.
    if allow_skip and path_destination.exists():
        return path_destination

    # 'is_symlink' is checked explicitly, because 'exists' is False for a broken symlink
    if path_destination.exists() or path_destination.is_symlink():
        path_destination.unlink()

    path_destination.parent.mkdir(parents=True, exist_ok=True)
    if storage_mode is FileStorageMode.COPY:
        shutil.copy2(path_source, path_destination)
        return path_destination

    # Symlinks are created with an absolute source path. Relative links would break as soon as the
    # written dataset (or the source dataset) is moved to another folder.
    path_source_absolute = path_source.resolve()
    try:
        path_destination.symlink_to(path_source_absolute)
    except OSError as e:
        raise OSError(
            f"Unable to create a symbolic link '{path_destination}' -> '{path_source_absolute}': {e}. "
            "On Windows, symbolic links require Developer Mode to be enabled "
            "('Settings > System > For developers') or an elevated (administrator) process. "
            f"Use storage_mode='{FileStorageMode.COPY.value}' to store real copies of the files instead."
        ) from e
    return path_destination


def is_valid_version_string(version: Optional[str], allow_none: bool = False, allow_latest: bool = False) -> bool:
    if allow_none and version is None:
        return True
    if allow_latest and version == "latest":
        return True
    return version_from_string(version, raise_error=False) is not None


def version_from_string(version: Optional[str], raise_error: bool = True) -> Optional[Version]:
    if version is None:
        if raise_error:
            raise ValueError("Version is 'None'. A valid version string is required e.g '1.0.0'")
        return None

    try:
        version_casted = Version(version)
    except (InvalidVersion, TypeError) as e:
        if raise_error:
            raise ValueError(f"Invalid version string/type: {version}") from e
        return None

    # Check if version is semantic versioning (MAJOR.MINOR.PATCH)
    if len(version_casted.release) < 3:
        if raise_error:
            raise ValueError(f"Version string '{version}' is not semantic versioning (MAJOR.MINOR.PATCH)")
        return None
    return version_casted


def dataset_name_and_version_from_string(
    string: str,
    resolve_missing_version: bool = True,
) -> Tuple[str, Optional[str]]:
    if not isinstance(string, str):
        raise TypeError(f"'{type(string)}' for '{string}' is an unsupported type. Expected 'str' e.g 'mnist:1.0.0'")

    parts = string.split(":")
    if len(parts) == 1:
        dataset_name = parts[0]
        if resolve_missing_version:
            version = "latest"  # Default to 'latest' if version is missing. This will be resolved to a specific version later.
            user_logger.info(f"Version is missing in dataset name: {string}. Defaulting to version='latest'.")
        else:
            raise ValueError(f"Version is missing in dataset name: {string}. Use 'name:version'")
    elif len(parts) == 2:
        dataset_name, version = parts
    else:
        raise ValueError(f"Invalid dataset name format: {string}. Use 'name' or 'name:version' ")

    if not is_valid_version_string(version, allow_none=True, allow_latest=True):
        raise ValueError(f"Invalid version string: {version}. Use semantic versioning e.g. '1.0.0' or 'latest'")

    return dataset_name, version


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


def save_image_with_hash_name(image: np.ndarray, path_folder: Path, allow_skip: bool = True) -> Path:
    pil_image = Image.fromarray(image)
    path_image = save_pil_image_with_hash_name(pil_image, path_folder, allow_skip=allow_skip)
    return path_image


def save_pil_image_with_hash_name(
    image: Image.Image,
    path_folder: Path,
    allow_skip: bool = True,
    compress_level: int = 1,
) -> Path:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=compress_level)
    png_bytes = buffer.getvalue()
    hash_value = hash_from_bytes(png_bytes)
    path_image = Path(path_folder) / relative_path_from_hash(hash=hash_value, suffix=".png")
    if allow_skip and path_image.exists():
        return path_image
    path_image.parent.mkdir(parents=True, exist_ok=True)
    path_image.write_bytes(png_bytes)
    return path_image


def copy_and_rename_file_to_hash_value(
    path_source: Path,
    path_dataset_root: Path,
    allow_skip: bool = True,
    storage_mode: Union[FileStorageMode, str] = FileStorageMode.COPY,
) -> Path:
    """
    Stores a file in a dataset root directory with a hash-based name and sub-directory structure.

    Depending on `storage_mode` the file is stored as a real copy (default) or as a symbolic link
    pointing at `path_source`. Accepts a `FileStorageMode` or the strings `"copy"`/`"symlink"`.
    """

    if not path_source.exists():
        raise FileNotFoundError(f"Source file {path_source} does not exist.")

    hash_value = hash_file_xxhash(path_source)
    path_file = path_dataset_root / relative_path_from_hash(hash=hash_value, suffix=path_source.suffix)

    return store_file(
        path_source=path_source,
        path_destination=path_file,
        storage_mode=storage_mode,
        allow_skip=allow_skip,
    )


def relative_path_from_hash(hash: str, suffix: str) -> str:
    return f"{hash}{suffix}"


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
