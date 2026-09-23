"""Download, extract and convert the public COCO 2017 dataset into a 'HafniaDataset'.

COCO 2017 is not available through torchvision, so the images and annotations are downloaded
directly from 'https://cocodataset.org' and converted with the COCO format converter in
'hafnia.dataset.format_conversions.format_coco'.

The dataset is downloaded as zip archives into a cache folder (see 'get_path_coco_2017') and
extracted into the same folder. Downloads and extractions are skipped when the extracted files are
already present, so repeated calls only pay the conversion cost.

The human pose annotations of COCO 2017 are converted into 'Skeleton' primitives. Note that COCO
annotates a varying number of the 17 body keypoints per person, while a 'Skeleton' is required to
contain all keypoints of its skeleton template. All keypoints are therefore kept with the COCO
visibility ('0=not labeled', '1=labeled but not visible', '2=labeled and visible') in
'KeyPoint.meta', and keypoints that are not labeled keep the (0, 0) coordinate used by COCO.
"""

import hashlib
import shutil
import textwrap
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from hafnia import utils
from hafnia.dataset.dataset_names import SplitName
from hafnia.dataset.format_conversions.format_coco import (
    CocoSplitPaths,
    from_coco_dataset_by_split_definitions,
)
from hafnia.log import user_logger

if TYPE_CHECKING:  # Using 'TYPE_CHECKING' to avoid circular imports during type checking
    from hafnia.dataset.hafnia_dataset import HafniaDataset

DATASET_NAME = "coco-2017"
URL_IMAGES = "http://images.cocodataset.org/zips"
URL_ANNOTATIONS = "http://images.cocodataset.org/annotations"
FOLDER_NAME_ARCHIVES = ".archives"
# Timeout for the socket operations of a download. The COCO archives are large, so the timeout is
# applied per read and not to the download as a whole.
DOWNLOAD_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class CocoArchive:
    """A zip archive of the COCO 2017 dataset and the files it provides after extraction."""

    url: str
    # Files and folders - relative to the dataset root - that exist after extraction of the archive.
    # Used to skip the download and extraction of archives that are already available on disk.
    extracted_paths: Tuple[str, ...]
    # MD5 checksum of the archive. It is 'None' for the large image archives, as COCO provides no
    # checksum for these. Their content is still verified by the CRC checks of zip during extraction.
    md5: Optional[str] = None

    @property
    def file_name(self) -> str:
        return self.url.rsplit("/", maxsplit=1)[-1]


ARCHIVE_IMAGES_TRAIN = CocoArchive(url=f"{URL_IMAGES}/train2017.zip", extracted_paths=("train2017",))
ARCHIVE_IMAGES_VAL = CocoArchive(
    url=f"{URL_IMAGES}/val2017.zip",
    extracted_paths=("val2017",),
    md5="442b8da7639aecaf257c1dceb8ba8c80",
)
ARCHIVE_IMAGES_TEST = CocoArchive(url=f"{URL_IMAGES}/test2017.zip", extracted_paths=("test2017",))
ARCHIVE_ANNOTATIONS_TRAINVAL = CocoArchive(
    url=f"{URL_ANNOTATIONS}/annotations_trainval2017.zip",
    extracted_paths=(
        "annotations/instances_train2017.json",
        "annotations/instances_val2017.json",
        "annotations/person_keypoints_train2017.json",
        "annotations/person_keypoints_val2017.json",
    ),
    md5="f4bbac642086de4f52a3fdda2de5fa2c",
)
ARCHIVE_IMAGE_INFO_TEST = CocoArchive(
    url=f"{URL_ANNOTATIONS}/image_info_test2017.zip",
    extracted_paths=("annotations/image_info_test2017.json",),
    md5="85da7065e5e600ebfee8af1edb634eb5",
)


@dataclass(frozen=True)
class Coco2017Split:
    """The images, annotations and archives of one split of the COCO 2017 dataset."""

    split: str  # Hafnia split name, e.g. 'validation'
    folder_images: str  # Image folder relative to the dataset root, e.g. 'val2017'
    path_instances_json: str  # Object annotations relative to the dataset root
    archives: Tuple[CocoArchive, ...]
    path_keypoints_json: Optional[str] = None  # Human pose annotations relative to the dataset root


COCO_2017_SPLITS: Dict[str, Coco2017Split] = {
    SplitName.TRAIN: Coco2017Split(
        split=SplitName.TRAIN,
        folder_images="train2017",
        path_instances_json="annotations/instances_train2017.json",
        path_keypoints_json="annotations/person_keypoints_train2017.json",
        archives=(ARCHIVE_IMAGES_TRAIN, ARCHIVE_ANNOTATIONS_TRAINVAL),
    ),
    SplitName.VAL: Coco2017Split(
        split=SplitName.VAL,
        folder_images="val2017",
        path_instances_json="annotations/instances_val2017.json",
        path_keypoints_json="annotations/person_keypoints_val2017.json",
        archives=(ARCHIVE_IMAGES_VAL, ARCHIVE_ANNOTATIONS_TRAINVAL),
    ),
    # The test split is not annotated publicly. 'image_info_test2017.json' contains the images and
    # the categories of the dataset, but no object annotations.
    SplitName.TEST: Coco2017Split(
        split=SplitName.TEST,
        folder_images="test2017",
        path_instances_json="annotations/image_info_test2017.json",
        path_keypoints_json=None,
        archives=(ARCHIVE_IMAGES_TEST, ARCHIVE_IMAGE_INFO_TEST),
    ),
}

# Splits that are converted by 'coco_2017_as_hafnia_dataset'. All splits are defined above and can be
# downloaded, but only the validation split is converted for now. The training split (19 GB of images)
# and the unannotated test split are enabled as a follow-up.
SUPPORTED_SPLITS: List[str] = [SplitName.VAL]

COCO_2017_BIBTEX = textwrap.dedent("""\
    @inproceedings{lin2014microsoft,
        title={Microsoft COCO: Common Objects in Context},
        author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro
                and Ramanan, Deva and Doll{\\'a}r, Piotr and Zitnick, C Lawrence},
        booktitle={European Conference on Computer Vision (ECCV)},
        pages={740--755},
        year={2014},
        organization={Springer}
    }""")


def get_path_coco_2017() -> Path:
    """Cache folder with the downloaded and extracted COCO 2017 dataset."""
    return utils.get_path_public_dataset_downloads() / DATASET_NAME


def coco_2017_as_hafnia_dataset(
    force_redownload: bool = False,
    n_samples: Optional[int] = None,
    splits: Optional[List[str]] = None,
    path_root: Optional[Path] = None,
) -> "HafniaDataset":
    """Download the public COCO 2017 dataset and convert it into a 'HafniaDataset'.

    Images and annotations are downloaded from 'https://cocodataset.org' and cached on disk, so the
    download and extraction is skipped if the dataset has already been downloaded and extracted.
    The dataset contains object detection ('Bbox'), instance segmentation ('Bitmask') and human pose
    ('Skeleton') annotations.

    Args:
        force_redownload: If True, delete the cached files of the requested splits and download again.
        n_samples: Optional cap on the number of samples (split evenly across the requested splits).
        splits: Splits to convert. Defaults to 'SUPPORTED_SPLITS' (the validation split).
        path_root: Optional override of the cache folder. Defaults to 'get_path_coco_2017()'.
    """
    splits = splits or SUPPORTED_SPLITS
    unsupported_splits = [split for split in splits if split not in SUPPORTED_SPLITS]
    if unsupported_splits:
        raise ValueError(
            f"The splits {unsupported_splits} of '{DATASET_NAME}' are not supported yet. "
            f"Supported splits: {SUPPORTED_SPLITS}."
        )

    path_root = download_and_extract_coco_2017(
        splits=splits,
        force_redownload=force_redownload,
        path_root=path_root,
    )
    split_definitions = get_coco_2017_split_paths(path_root=path_root, splits=splits)

    dataset = from_coco_dataset_by_split_definitions(
        split_definitions=split_definitions,
        max_samples=n_samples,
        dataset_name=DATASET_NAME,
    )
    dataset.info.version = "1.0.0"
    dataset.info.reference_bibtex = COCO_2017_BIBTEX
    dataset.info.reference_paper_url = "https://arxiv.org/abs/1405.0312"
    dataset.info.reference_dataset_page = "https://cocodataset.org"
    return dataset


def get_coco_2017_split_paths(path_root: Path, splits: Optional[List[str]] = None) -> List[CocoSplitPaths]:
    """Resolve the image folders and label files of the extracted COCO 2017 splits."""
    splits = splits or SUPPORTED_SPLITS
    split_paths = []
    for split in splits:
        split_definition = COCO_2017_SPLITS[split]
        path_keypoints_json = None
        if split_definition.path_keypoints_json is not None:
            path_keypoints_json = path_root / split_definition.path_keypoints_json
        split_paths.append(
            CocoSplitPaths(
                split=split_definition.split,
                path_images=path_root / split_definition.folder_images,
                path_instances_json=path_root / split_definition.path_instances_json,
                path_keypoints_json=path_keypoints_json,
            )
        )
    return split_paths


def download_and_extract_coco_2017(
    splits: Optional[List[str]] = None,
    force_redownload: bool = False,
    path_root: Optional[Path] = None,
    remove_archive_after_extraction: bool = True,
) -> Path:
    """Download and extract the archives required by the given COCO 2017 splits.

    Archives that have already been extracted are skipped, so the download is only done once.

    Args:
        splits: Splits to download. Defaults to 'SUPPORTED_SPLITS' (the validation split).
        force_redownload: If True, delete the extracted files of the requested splits and download again.
        path_root: Optional override of the cache folder. Defaults to 'get_path_coco_2017()'.
        remove_archive_after_extraction: If True, delete the zip archive once it has been extracted.
            The COCO archives are large (19 GB for the training images), so they are removed by default.

    Returns:
        The dataset root folder with the extracted images and annotations.
    """
    splits = splits or SUPPORTED_SPLITS
    path_root = path_root or get_path_coco_2017()

    archives: List[CocoArchive] = []
    for split in splits:
        if split not in COCO_2017_SPLITS:
            raise ValueError(f"Unknown split '{split}' for '{DATASET_NAME}'. Splits: {list(COCO_2017_SPLITS)}")
        for archive in COCO_2017_SPLITS[split].archives:
            # The same archive is shared by multiple splits - e.g. the train and validation annotations
            if archive not in archives:
                archives.append(archive)

    for archive in archives:
        download_and_extract_coco_archive(
            archive=archive,
            path_root=path_root,
            force_redownload=force_redownload,
            remove_archive_after_extraction=remove_archive_after_extraction,
        )
    return path_root


def download_and_extract_coco_archive(
    archive: CocoArchive,
    path_root: Path,
    force_redownload: bool = False,
    remove_archive_after_extraction: bool = True,
) -> None:
    """Download and extract a single COCO archive into 'path_root' unless it is already extracted."""
    path_archive = path_root / FOLDER_NAME_ARCHIVES / archive.file_name

    if force_redownload:
        for extracted_path in archive.extracted_paths:
            remove_path(path_root / extracted_path)
        get_path_extraction_marker(archive=archive, path_root=path_root).unlink(missing_ok=True)
        path_archive.unlink(missing_ok=True)

    if is_archive_extracted(archive=archive, path_root=path_root):
        user_logger.info(f"Skipping download of '{archive.file_name}'. Already extracted in '{path_root}'")
        return

    if path_archive.exists() and not is_matching_md5(path_archive, md5=archive.md5):
        user_logger.warning(f"Removing '{path_archive}' as the checksum does not match the expected checksum.")
        path_archive.unlink()

    if path_archive.exists():
        user_logger.info(f"Using already downloaded archive: '{path_archive}'")
    else:
        download_file(url=archive.url, path_file=path_archive)
        if not is_matching_md5(path_archive, md5=archive.md5):
            path_archive.unlink()
            raise ValueError(
                f"The checksum of the downloaded archive '{archive.file_name}' does not match the expected "
                f"checksum '{archive.md5}'. The download may be corrupted - please try again."
            )

    extract_archive(path_archive=path_archive, path_output=path_root)
    get_path_extraction_marker(archive=archive, path_root=path_root).write_text(archive.url)

    if remove_archive_after_extraction:
        path_archive.unlink(missing_ok=True)


def get_path_extraction_marker(archive: CocoArchive, path_root: Path) -> Path:
    """Marker file written after an archive has been fully extracted.

    The marker is used - and not just the extracted files - to also detect an extraction that was
    interrupted. An interrupted extraction leaves e.g. the 'val2017' image folder with only a part
    of the images, which would otherwise look like a fully extracted archive.
    """
    return path_root / FOLDER_NAME_ARCHIVES / f"{archive.file_name}.extracted"


def is_archive_extracted(archive: CocoArchive, path_root: Path) -> bool:
    """Check if the archive has been fully extracted and the extracted files are still present."""
    if not get_path_extraction_marker(archive=archive, path_root=path_root).exists():
        return False
    return all((path_root / extracted_path).exists() for extracted_path in archive.extracted_paths)


def is_matching_md5(path_file: Path, md5: Optional[str], chunk_size: int = 1024 * 1024) -> bool:
    """Check the MD5 checksum of a file. Files are accepted ('True') when no checksum is provided."""
    if md5 is None:
        return True
    file_hash = hashlib.md5()
    with path_file.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            file_hash.update(chunk)
    return file_hash.hexdigest() == md5


def download_file(url: str, path_file: Path, chunk_size: int = 1024 * 1024) -> Path:
    """Download a file with a progress bar. The file is only moved in place after a full download."""
    path_file.parent.mkdir(parents=True, exist_ok=True)
    path_partial_file = path_file.with_suffix(path_file.suffix + ".part")
    path_partial_file.unlink(missing_ok=True)

    user_logger.info(f"Downloading '{url}' to '{path_file}'")
    progress_bar = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TextColumn("ETA:"),
        TimeRemainingColumn(),
        TextColumn("| Elapsed:"),
        TimeElapsedColumn(),
    )
    # 'urlopen' is used with the public COCO dataset urls and the 'file://' urls of the unit tests
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:  # noqa: S310
        content_length = response.headers.get("Content-Length")
        total_bytes = int(content_length) if content_length is not None else None
        with progress_bar as progress, path_partial_file.open("wb") as file:
            task = progress.add_task(f"Downloading '{path_file.name}'", total=total_bytes)
            for chunk in iter(lambda: response.read(chunk_size), b""):
                file.write(chunk)
                progress.update(task, advance=len(chunk))

    path_file.unlink(missing_ok=True)
    path_partial_file.rename(path_file)
    return path_file


def extract_archive(path_archive: Path, path_output: Path) -> Path:
    """Extract a zip archive with a progress bar showing the number of extracted files."""
    user_logger.info(f"Extracting '{path_archive}' to '{path_output}'")
    path_output.mkdir(parents=True, exist_ok=True)
    path_archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path_archive) as zip_file:
        members = zip_file.namelist()
        description = f"Extracting '{path_archive.name}'"
        for member in utils.progress_bar(members, description=description):
            zip_file.extract(member, path=path_output)
    return path_output


def remove_path(path: Path) -> None:
    """Remove a file or a folder if it exists."""
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        return
    path.unlink(missing_ok=True)
