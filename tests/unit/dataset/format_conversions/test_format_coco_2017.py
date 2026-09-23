import hashlib
import json
import zipfile
from pathlib import Path
from typing import Callable

import polars as pl
import pytest

from hafnia.dataset.dataset_names import SampleField, SplitName
from hafnia.dataset.format_conversions import format_coco, format_coco_2017
from hafnia.dataset.format_conversions.format_coco_2017 import CocoArchive
from hafnia.dataset.hafnia_dataset_types import Sample
from hafnia.dataset.primitives import Skeleton
from tests import helper_testing

N_COCO_KEYPOINTS = 17
N_COCO_SKELETON_EDGES = 19
IMAGE_NAME_WITH_KEYPOINTS = "000000000139.jpg"


def get_path_coco_2017_test_dataset() -> Path:
    """A tiny COCO 2017 dataset with the original folder structure and two validation images."""
    return helper_testing.get_path_test_dataset_formats() / "format_coco_2017"


def get_coco_2017_test_dataset():
    path_dataset = get_path_coco_2017_test_dataset()
    split_definitions = format_coco_2017.get_coco_2017_split_paths(path_root=path_dataset, splits=[SplitName.VAL])
    return format_coco.from_coco_dataset_by_split_definitions(
        split_definitions=split_definitions,
        max_samples=None,
        dataset_name=format_coco_2017.DATASET_NAME,
    )


def test_coco_2017_split_paths_include_keypoints() -> None:
    path_dataset = get_path_coco_2017_test_dataset()
    split_definitions = format_coco_2017.get_coco_2017_split_paths(path_root=path_dataset, splits=[SplitName.VAL])

    assert len(split_definitions) == 1
    split_definition = split_definitions[0]
    assert split_definition.split == SplitName.VAL
    assert split_definition.path_images == path_dataset / "val2017"
    assert split_definition.path_instances_json == path_dataset / "annotations" / "instances_val2017.json"
    assert split_definition.path_keypoints_json == path_dataset / "annotations" / "person_keypoints_val2017.json"


def test_from_coco_2017_with_keypoints() -> None:
    dataset = get_coco_2017_test_dataset()

    # The keypoint annotations are converted into a 'Skeleton' task with a skeleton template per class
    task = dataset.info.get_task_by_primitive(Skeleton)
    assert task.name == Skeleton.default_task_name()
    assert task.get_class_names() == ["person"]
    template = task.classes[0].skeleton  # type: ignore[index]
    assert template is not None
    assert len(template.keypoint_names) == N_COCO_KEYPOINTS
    assert template.keypoint_names[0] == "nose"
    assert len(template.edges) == N_COCO_SKELETON_EDGES
    # COCO defines edges as 1-indexed keypoint pairs. The first edge is [16, 14] and becomes (15, 13)
    assert (template.edges[0].index_start, template.edges[0].index_end) == (15, 13)

    # Skeletons are stored as annotations of the sample and keep the keypoint order of the template
    # Samples without any labeled keypoints have no skeletons ('None' in the samples table)
    skeletons_per_sample = dataset.samples["skeletons"].to_list()
    skeletons = [Skeleton(**skeleton) for skeletons in skeletons_per_sample for skeleton in (skeletons or [])]
    assert len(skeletons) == 1, "Only one of the two person annotations in the test dataset has labeled keypoints"
    skeleton = skeletons[0]
    assert skeleton.class_name == "person"
    assert skeleton.class_idx == 0
    assert [keypoint.class_name for keypoint in skeleton.keypoints] == template.keypoint_names
    assert [keypoint.class_idx for keypoint in skeleton.keypoints] == list(range(N_COCO_KEYPOINTS))
    assert skeleton.meta == {"iscrowd": 0, "num_keypoints": 15}

    # Keypoints are normalized and store the COCO visibility (0=not labeled, 1=labeled, 2=visible)
    for keypoint in skeleton.keypoints:
        assert 0.0 <= keypoint.point.x <= 1.0
        assert 0.0 <= keypoint.point.y <= 1.0
        assert keypoint.meta is not None and keypoint.meta["visibility"] in [0, 1, 2]
        # Keypoints that COCO has not labeled are marked as unlabeled and are skipped when drawing
        assert keypoint.labeled == (keypoint.meta["visibility"] > 0)
    n_labeled_keypoints = sum(keypoint.labeled for keypoint in skeleton.keypoints)
    assert n_labeled_keypoints == 15

    # Bboxes and bitmasks of the instance annotations are still converted
    assert dataset.samples["bboxes"].list.len().sum() > 0
    assert dataset.samples["bitmasks"].list.len().sum() > 0

    # 'check_splits=False' as only the validation split of COCO 2017 is converted for now
    dataset.check_dataset(check_splits=False)


def test_from_coco_2017_keypoints_visualized(compare_to_expected_image: Callable) -> None:
    dataset = get_coco_2017_test_dataset()
    samples = dataset.samples.filter(pl.col(SampleField.FILE_PATH).str.contains(IMAGE_NAME_WITH_KEYPOINTS))
    assert len(samples) == 1
    sample = Sample(**samples.row(0, named=True))

    # 'tasks' is required to also draw the edges between the keypoints of a skeleton. Keypoints
    # that COCO has not labeled keep the (0, 0) coordinate, but are marked as unlabeled and are
    # therefore skipped - together with their edges - when the skeleton is drawn.
    sample_visualized = sample.draw_annotations(tasks=dataset.info.tasks)
    compare_to_expected_image(sample_visualized)


def test_skeleton_task_from_coco_keypoint_categories_without_keypoints() -> None:
    categories = [{"id": 1, "name": "person", "supercategory": "person"}]
    with pytest.raises(ValueError, match="No COCO categories with keypoints found"):
        format_coco.skeleton_task_from_coco_keypoint_categories(categories)


def test_coco_keypoint_annotation_with_unexpected_number_of_keypoints() -> None:
    path_keypoints = get_path_coco_2017_test_dataset() / "annotations" / "person_keypoints_val2017.json"
    keypoint_labels = format_coco.read_coco_keypoint_label_file(path_keypoints)

    annotation = {"id": 1, "category_id": 1, "keypoints": [0, 0, 1], "num_keypoints": 1}
    with pytest.raises(ValueError, match="keypoint values"):
        keypoint_labels.to_skeleton(annotation, image_height=100, image_width=100)


def test_download_is_skipped_when_dataset_is_extracted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Downloads are skipped when the files of an archive are already extracted."""

    def fail_on_download(*args, **kwargs):
        raise AssertionError("The dataset is already extracted, so no download is expected")

    monkeypatch.setattr(format_coco_2017, "download_file", fail_on_download)

    for split in [SplitName.VAL]:
        for archive in format_coco_2017.COCO_2017_SPLITS[split].archives:
            for extracted_path in archive.extracted_paths:
                path_extracted = tmp_path / extracted_path
                path_extracted.parent.mkdir(parents=True, exist_ok=True)
                path_extracted.touch()
            path_marker = format_coco_2017.get_path_extraction_marker(archive=archive, path_root=tmp_path)
            path_marker.parent.mkdir(parents=True, exist_ok=True)
            path_marker.touch()

    path_root = format_coco_2017.download_and_extract_coco_2017(splits=[SplitName.VAL], path_root=tmp_path)
    assert path_root == tmp_path


def test_download_and_extract_coco_archive(tmp_path: Path) -> None:
    """Download, checksum verification and extraction of an archive using a local 'file://' archive."""
    path_archive_source = tmp_path / "source" / "annotations.zip"
    path_archive_source.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path_archive_source, "w") as zip_file:
        zip_file.writestr("annotations/instances_val2017.json", json.dumps({"images": []}))

    md5 = hashlib.md5(path_archive_source.read_bytes()).hexdigest()
    archive = CocoArchive(
        url=path_archive_source.as_uri(),
        extracted_paths=("annotations/instances_val2017.json",),
        md5=md5,
    )

    path_root = tmp_path / "coco-2017"
    format_coco_2017.download_and_extract_coco_archive(archive=archive, path_root=path_root)

    assert (path_root / "annotations" / "instances_val2017.json").exists()
    # The archive is large for the real dataset and is removed after a successful extraction
    assert not (path_root / format_coco_2017.FOLDER_NAME_ARCHIVES / "annotations.zip").exists()

    # A second call is a no-op, as the files of the archive have already been extracted
    assert format_coco_2017.is_archive_extracted(archive=archive, path_root=path_root)
    format_coco_2017.download_and_extract_coco_archive(archive=archive, path_root=path_root)
    assert (path_root / "annotations" / "instances_val2017.json").exists()

    # An interrupted extraction is detected and extracted again, as the marker file is missing
    format_coco_2017.get_path_extraction_marker(archive=archive, path_root=path_root).unlink()
    assert not format_coco_2017.is_archive_extracted(archive=archive, path_root=path_root)


def test_download_and_extract_coco_archive_with_wrong_checksum(tmp_path: Path) -> None:
    path_archive_source = tmp_path / "source" / "annotations.zip"
    path_archive_source.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path_archive_source, "w") as zip_file:
        zip_file.writestr("annotations/instances_val2017.json", json.dumps({"images": []}))

    archive = CocoArchive(
        url=path_archive_source.as_uri(),
        extracted_paths=("annotations/instances_val2017.json",),
        md5="0" * 32,
    )
    path_root = tmp_path / "coco-2017"
    with pytest.raises(ValueError, match="checksum"):
        format_coco_2017.download_and_extract_coco_archive(archive=archive, path_root=path_root)
