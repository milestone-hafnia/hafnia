import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import polars as pl
from pycocotools import mask as coco_utils

from hafnia.dataset import license_types
from hafnia.dataset.dataset_helpers import FileStorageMode, resolve_storage_mode, store_file
from hafnia.dataset.dataset_names import SampleField, SplitName
from hafnia.dataset.format_conversions import format_coco, format_helpers
from hafnia.utils import progress_bar

if TYPE_CHECKING:  # Using 'TYPE_CHECKING' to avoid circular imports during type checking
    from hafnia.dataset.hafnia_dataset import HafniaDataset

from hafnia.dataset.hafnia_dataset_types import Attribution, ClassInfo, DatasetInfo, License, Sample, TaskInfo
from hafnia.dataset.primitives import Bbox, Bitmask, KeyPoint, Point, Skeleton, SkeletonEdge, SkeletonTemplate
from hafnia.log import user_logger

COCO_KEY_FILE_NAME = "file_name"

HAFNIA_TO_ROBOFLOW_SPLIT_NAME = {
    SplitName.TRAIN: "train",
    SplitName.VAL: "valid",
    SplitName.TEST: "test",
}
ROBOFLOW_ANNOTATION_FILE_NAME = "_annotations.coco.json"


@dataclass
class CocoSplitPaths:
    split: str
    path_images: Path
    path_instances_json: Path
    # Keypoint annotations are stored in a separate label file in the original COCO format,
    # e.g. 'person_keypoints_val2017.json' for the human pose keypoints of COCO 2017.
    path_keypoints_json: Optional[Path] = None


def from_coco_format(
    path_dataset: Path,
    coco_format_type: str = "roboflow",
    max_samples: Optional[int] = None,
    dataset_name: str = "coco-2017",
):
    """Import a COCO-formatted dataset as a `HafniaDataset`.

    Resolves split-level image folders and `instances.json` files according to the chosen layout
    and merges them into one dataset. Currently supports the Roboflow on-disk layout (one
    subfolder per split, each containing a `_annotations.coco.json` instances file).

    Args:
        path_dataset: Root folder containing the COCO splits.
        coco_format_type: Layout type; only ``"roboflow"`` is currently supported.
        max_samples: Optional cap on the total number of samples (split evenly across splits).
        dataset_name: Name to assign to the resulting dataset.
    """
    split_definitions = get_split_paths_for_coco_dataset_formats(
        path_dataset=path_dataset, coco_format_type=coco_format_type
    )

    hafnia_dataset = from_coco_dataset_by_split_definitions(
        split_definitions=split_definitions,
        max_samples=max_samples,
        dataset_name=dataset_name,
    )

    return hafnia_dataset


def get_split_paths_for_coco_dataset_formats(
    path_dataset: Path,
    coco_format_type: str,
) -> List[CocoSplitPaths]:
    splits = []
    if coco_format_type == "roboflow":
        for split_def in format_helpers.get_splits_from_folder(path_dataset):
            splits.append(
                CocoSplitPaths(
                    split=split_def.name,
                    path_images=split_def.path,
                    path_instances_json=split_def.path / ROBOFLOW_ANNOTATION_FILE_NAME,
                )
            )
        return splits

    raise ValueError(f"The specified '{coco_format_type=}' is not supported.")


def from_coco_dataset_by_split_definitions(
    split_definitions: List[CocoSplitPaths],
    max_samples: Optional[int],
    dataset_name: str,
) -> "HafniaDataset":
    from hafnia.dataset.hafnia_dataset import HafniaDataset

    if max_samples is None:
        max_samples_per_split = None
    else:
        max_samples_per_split = max_samples // len(split_definitions)
    samples = []
    tasks: List[TaskInfo] = []
    for split_definition in split_definitions:
        if split_definition.path_instances_json is None or not split_definition.path_instances_json.exists():
            raise FileNotFoundError(
                f"Expected COCO dataset files not found for split '{split_definition.split}'. "
                f"Label file doesn't exist: {split_definition.path_instances_json}"
            )
        if not split_definition.path_images.exists():
            raise FileNotFoundError(
                f"Expected COCO dataset files not found for split '{split_definition.split}'. "
                f"Images folder doesn't exist: {split_definition.path_images}"
            )

        samples_in_split, tasks_in_split = coco_format_folder_with_split_to_hafnia_samples(
            path_label_file=split_definition.path_instances_json,
            max_samples_per_split=max_samples_per_split,
            path_images=split_definition.path_images,
            split_name=split_definition.split,
            path_keypoints_label_file=split_definition.path_keypoints_json,
        )

        for task_in_split in tasks_in_split:
            matching_tasks = [task for task in tasks if task.name == task_in_split.name]

            add_missing_task = len(matching_tasks) == 0
            if add_missing_task:
                tasks.append(task_in_split)
                continue

            if len(matching_tasks) != 1:
                raise ValueError("Duplicate task names found across splits in the COCO dataset.")
            match_task = matching_tasks[0]
            if task_in_split != match_task:
                raise ValueError(
                    f"Inconsistent task found across splits in the COCO dataset for task name '{task_in_split.name}'. "
                )

        samples.extend(samples_in_split)

    dataset_info = DatasetInfo(
        dataset_name=dataset_name,
        tasks=tasks,
    )

    hafnia_dataset = HafniaDataset.from_samples_list(samples, info=dataset_info)
    return hafnia_dataset


def coco_format_folder_with_split_to_hafnia_samples(
    path_label_file: Path,
    path_images: Path,
    split_name: str,
    max_samples_per_split: Optional[int],
    path_keypoints_label_file: Optional[Path] = None,
) -> Tuple[List[Sample], List[TaskInfo]]:
    if not path_label_file.exists():
        raise FileNotFoundError(f"Expected label file not found: {path_label_file}")
    user_logger.info("Loading coco label file as json")
    image_and_annotation_dict = json.loads(path_label_file.read_text())
    user_logger.info("Converting coco dataset to HafniaDataset samples")

    id_to_category, class_names = get_coco_id_category_mapping(image_and_annotation_dict.get("categories", []))
    tasks = [
        TaskInfo.from_class_names(primitive=Bbox, class_names=class_names),
        TaskInfo.from_class_names(primitive=Bitmask, class_names=class_names),
    ]

    coco_licenses = image_and_annotation_dict.get("licenses", [])
    id_to_license_mapping = {lic["id"]: license_types.get_license_by_url(lic["url"]) for lic in coco_licenses}

    coco_images = image_and_annotation_dict.get("images", [])
    if max_samples_per_split is not None:
        coco_images = coco_images[:max_samples_per_split]
    id_to_image = {img["id"]: img for img in coco_images}

    img_id_to_annotations = group_coco_annotations_by_image_id(image_and_annotation_dict.get("annotations", []))

    keypoint_labels = None
    if path_keypoints_label_file is not None:
        keypoint_labels = read_coco_keypoint_label_file(path_keypoints_label_file)
        tasks.append(keypoint_labels.task)

    samples = []
    for img_id, image_dict in progress_bar(
        id_to_image.items(), description=f"Convert coco to hafnia sample '{split_name}'"
    ):
        image_annotations = img_id_to_annotations.get(img_id, [])

        if keypoint_labels is None:
            skeletons = []
        else:
            skeletons = keypoint_labels.skeletons_for_image(
                image_id=img_id,
                image_height=image_dict["height"],
                image_width=image_dict["width"],
            )

        sample = fiftyone_coco_to_hafnia_sample(
            path_images=path_images,
            image_dict=image_dict,
            image_annotations=image_annotations,
            id_to_category=id_to_category,
            class_names=class_names,
            id_to_license_mapping=id_to_license_mapping,
            split_name=split_name,
            skeletons=skeletons,
        )
        samples.append(sample)

    return samples, tasks


def group_coco_annotations_by_image_id(coco_annotations: List[Dict]) -> Dict[int, List[Dict]]:
    """Group a flat list of COCO annotations by the 'image_id' of each annotation."""
    img_id_to_annotations: Dict[int, List[Dict]] = {}
    for annotation in coco_annotations:
        img_id_to_annotations.setdefault(annotation["image_id"], []).append(annotation)
    return img_id_to_annotations


@dataclass
class CocoKeypointLabels:
    """Keypoint annotations of a COCO keypoint label file, e.g. 'person_keypoints_val2017.json'.

    In the COCO format, keypoints are stored in a separate label file with one 'keypoints' list per
    object annotation. The keypoint names and the edges ('skeleton') between keypoints are defined
    per category and are stored as a 'SkeletonTemplate' on the 'ClassInfo' of the 'Skeleton' task.
    """

    task: TaskInfo
    category_id_to_class_name: Dict[int, str]
    annotations_by_image_id: Dict[int, List[Dict]]

    def skeletons_for_image(self, image_id: int, image_height: int, image_width: int) -> List[Skeleton]:
        """Convert the keypoint annotations of one image into 'Skeleton' primitives."""
        skeletons = []
        for annotation in self.annotations_by_image_id.get(image_id, []):
            skeleton = self.to_skeleton(annotation, image_height=image_height, image_width=image_width)
            if skeleton is None:
                continue
            skeletons.append(skeleton)
        return skeletons

    def to_skeleton(self, annotation: Dict, image_height: int, image_width: int) -> Optional[Skeleton]:
        """Convert one COCO keypoint annotation into a 'Skeleton'.

        Returns 'None' for annotations without keypoints - either because the category defines no
        keypoints or because no keypoints of the object have been labeled ('num_keypoints=0').
        """
        class_name = self.category_id_to_class_name.get(annotation["category_id"])
        if class_name is None:
            return None

        class_info = self.task.get_class_by_name(class_name)
        template = class_info.skeleton if class_info else None
        if template is None:
            raise ValueError(f"Missing skeleton template for class '{class_name}' in task '{self.task.name}'.")
        keypoint_names = template.keypoint_names

        # COCO stores keypoints flattened as [x0, y0, visibility0, x1, y1, visibility1, ...]
        flat_keypoints = annotation.get("keypoints") or []
        n_expected_values = 3 * len(keypoint_names)
        if len(flat_keypoints) != n_expected_values:
            raise ValueError(
                f"The keypoint annotation '{annotation.get('id')}' of class '{class_name}' has "
                f"{len(flat_keypoints)} keypoint values, but {n_expected_values} values are expected for the "
                f"{len(keypoint_names)} keypoints of the skeleton template: {keypoint_names}."
            )

        visibilities = flat_keypoints[2::3]
        n_labeled_keypoints = sum(1 for visibility in visibilities if visibility > 0)
        if n_labeled_keypoints == 0:
            # Objects with no labeled keypoints (e.g. a person in a crowd) are still annotated in the
            # instances label file, so no information is lost by skipping the empty skeleton.
            return None

        object_id = str(annotation["id"])
        keypoints = [
            KeyPoint(
                point=Point(
                    x=flat_keypoints[3 * keypoint_index] / image_width,
                    y=flat_keypoints[3 * keypoint_index + 1] / image_height,
                ),
                class_name=keypoint_name,
                class_idx=keypoint_index,
                object_id=object_id,
                # Visibility as defined by COCO: 0=not labeled, 1=labeled but not visible, 2=labeled and
                # visible. Keypoints that are not labeled are stored as the (0, 0) coordinate by COCO.
                meta={"visibility": int(visibilities[keypoint_index])},
            )
            for keypoint_index, keypoint_name in enumerate(keypoint_names)
        ]

        return Skeleton(
            keypoints=keypoints,
            class_name=class_name,
            class_idx=self.task.get_class_index(class_name),
            object_id=object_id,
            task_name=self.task.name or Skeleton.default_task_name(),
            meta={
                "iscrowd": annotation.get("iscrowd"),
                "num_keypoints": annotation.get("num_keypoints", n_labeled_keypoints),
            },
        )


def read_coco_keypoint_label_file(
    path_keypoints_label_file: Path, task_name: Optional[str] = None
) -> CocoKeypointLabels:
    """Read a COCO keypoint label file, e.g. 'person_keypoints_val2017.json'.

    Args:
        path_keypoints_label_file: Path to a COCO label file with a 'keypoints' list per annotation.
        task_name: Optional name for the resulting 'Skeleton' task. Defaults to the task name of the
            'Skeleton' primitive ('pose_estimation').
    """
    if not path_keypoints_label_file.exists():
        raise FileNotFoundError(f"Expected keypoint label file not found: {path_keypoints_label_file}")
    user_logger.info(f"Loading coco keypoint label file as json: '{path_keypoints_label_file.name}'")
    keypoints_dict = json.loads(path_keypoints_label_file.read_text())

    coco_categories = keypoints_dict.get("categories", [])
    task = skeleton_task_from_coco_keypoint_categories(coco_categories, task_name=task_name)
    class_names = task.get_class_names() or []
    category_id_to_class_name = {
        category["id"]: category["name"] for category in coco_categories if category["name"] in class_names
    }
    return CocoKeypointLabels(
        task=task,
        category_id_to_class_name=category_id_to_class_name,
        annotations_by_image_id=group_coco_annotations_by_image_id(keypoints_dict.get("annotations", [])),
    )


def skeleton_task_from_coco_keypoint_categories(
    coco_categories: List[Dict], task_name: Optional[str] = None
) -> TaskInfo:
    """Create a 'Skeleton' task with a skeleton template per class from COCO keypoint categories.

    Categories without keypoints are skipped, as e.g. the 'person_keypoints' label files of COCO 2017
    may contain categories with no keypoint definition.
    """
    classes = []
    for category in coco_categories:
        keypoint_names = category.get("keypoints")
        if not keypoint_names:
            continue
        # COCO edges are pairs of 1-indexed keypoint indices, e.g. [[16, 14], [14, 12], ...]
        edges = [
            SkeletonEdge(index_start=index_start - 1, index_end=index_end - 1)
            for index_start, index_end in category.get("skeleton", [])
        ]
        classes.append(
            ClassInfo(
                name=category["name"],
                skeleton=SkeletonTemplate(keypoint_names=list(keypoint_names), edges=edges),
            )
        )

    if len(classes) == 0:
        raise ValueError("No COCO categories with keypoints found. Expected at least one category with keypoints.")

    return TaskInfo(primitive=Skeleton, classes=classes, name=task_name)


def get_coco_id_category_mapping(
    coco_categories: List[dict],
) -> Tuple[Dict[int, dict], List[str]]:
    category_mapping = {}
    for i_cat, category in enumerate(coco_categories):
        category = category.copy()  # Create a copy to avoid modifying the original dictionary.
        category["class_idx"] = i_cat  # Add an index to the category for easier access.
        category_mapping[category["id"]] = category  # Map the category ID to the category dictionary.
    sorted_category_mapping = dict(sorted(category_mapping.items(), key=lambda item: item[1]["class_idx"]))
    class_names = [cat_data["name"] for cat_data in sorted_category_mapping.values()]
    return sorted_category_mapping, class_names


def convert_segmentation_to_rle_list(segmentation: Union[Dict, List], height: int, width: int) -> List[Dict]:
    is_polygon_format = isinstance(segmentation, list)
    if is_polygon_format:  # Multiple polygons format
        rles = coco_utils.frPyObjects(segmentation, height, width)
        return rles

    is_rle_format = isinstance(segmentation, dict) and "counts" in segmentation
    if is_rle_format:  # RLE format
        counts = segmentation["counts"]  # type: ignore
        uncompressed_list_of_ints = isinstance(counts, list)
        if uncompressed_list_of_ints:  # Uncompressed RLE. Counts is List[int]
            rles = coco_utils.frPyObjects([segmentation], height, width)
            return rles

        is_compressed_str_or_bytes = isinstance(counts, str | bytes)
        if is_compressed_str_or_bytes:  # Compressed RLE. Counts is str
            rles = [segmentation]
            return rles

    raise ValueError("Segmentation format not recognized for conversion to RLE.")


def fiftyone_coco_to_hafnia_sample(
    path_images: Path,
    image_dict: Dict,
    image_annotations: List[Dict],
    id_to_category: Dict,
    class_names: List[str],
    id_to_license_mapping: Dict[int, License],
    split_name: str,
    skeletons: Optional[List[Skeleton]] = None,
) -> Sample:
    image_dict = image_dict.copy()  # Create a copy to avoid modifying the original dictionary.
    file_name_relative = image_dict.pop(COCO_KEY_FILE_NAME)
    file_name = path_images / file_name_relative
    if not file_name.exists():
        raise FileNotFoundError(f"Expected image file not found: {file_name}. Please check the dataset structure.")

    img_width = image_dict.pop("width")
    img_height = image_dict.pop("height")
    bitmasks: List[Bitmask] = []
    bboxes: List[Bbox] = []
    for obj_instance in image_annotations:
        category_data = id_to_category[obj_instance["category_id"]]
        class_name = category_data["name"]  # Get the name of the category.
        class_idx = class_names.index(class_name)
        bbox_list = obj_instance["bbox"]
        if isinstance(bbox_list[0], float):  # Polygon coordinates are often floats.
            bbox_ints = [int(coord) for coord in bbox_list]
        else:
            bbox_ints = bbox_list
        rle_list = convert_segmentation_to_rle_list(obj_instance["segmentation"], height=img_height, width=img_width)
        rle = coco_utils.merge(rle_list)
        rle_string = rle["counts"]
        if isinstance(rle_string, bytes):
            rle_string = rle_string.decode("utf-8")

        if "area" in obj_instance and obj_instance["area"] is not None:
            area_px = obj_instance["area"]
        else:
            area_px = coco_utils.area(rle).item()
        area = float(area_px) / (img_height * img_width)
        bitmask = Bitmask(
            top=bbox_ints[1],
            left=bbox_ints[0],
            height=bbox_ints[3],
            width=bbox_ints[2],
            area=area,
            rle_string=rle_string,
            class_name=class_name,
            class_idx=class_idx,
            object_id=str(obj_instance["id"]),
            meta={"iscrowd": obj_instance["iscrowd"]},
        )
        bitmasks.append(bitmask)

        bbox = Bbox.from_coco(bbox=bbox_list, height=img_height, width=img_width)
        bbox.class_name = class_name
        bbox.class_idx = class_idx
        bbox.object_id = str(obj_instance["id"])  # Use the ID from the instance if available.
        bbox.meta = {"iscrowd": obj_instance["iscrowd"]}
        bbox.area = bbox.calculate_area(image_height=img_height, image_width=img_width)
        bboxes.append(bbox)

    if "license" in image_dict:
        license_data: License = id_to_license_mapping[image_dict["license"]]

        capture_date = datetime.fromisoformat(image_dict["date_captured"])
        source_url = image_dict["flickr_url"] if "flickr_url" in image_dict else image_dict.get("coco_url")
        attribution = Attribution(
            date_captured=capture_date,
            licenses=[license_data],
            source_url=source_url,
        )
    else:
        attribution = None

    return Sample(
        file_path=str(file_name),
        width=img_width,
        height=img_height,
        split=split_name,
        bboxes=bboxes,  # Bboxes will be added later if needed.
        bitmasks=bitmasks,  # Add the bitmask to the sample.
        skeletons=skeletons or None,  # Skeletons are read from a separate COCO keypoint label file.
        attribution=attribution,
        meta=image_dict,
    )


def to_coco_format(
    dataset: "HafniaDataset",
    path_output: Path,
    task_name: Optional[str] = None,
    coco_format_type: str = "roboflow",
    storage_mode: Union[FileStorageMode, str] = FileStorageMode.COPY,
) -> List[CocoSplitPaths]:
    """Export a `HafniaDataset` to COCO format on disk and return the paths of each split.

    Writes one `instances.json` per split, plus the corresponding image folder layout. If the
    dataset has both `Bitmask` and `Bbox` tasks and `task_name` is omitted, segmentation masks are
    preferred over bounding boxes.

    Args:
        dataset: Dataset to export.
        path_output: Output root folder; one subfolder per split is created beneath it.
        task_name: Specific task to export. If None, a single eligible task is auto-selected
            (Bitmask first, then Bbox).
        coco_format_type: Output layout. Currently only ``"roboflow"`` is supported.
        storage_mode: How image/video files are stored: `FileStorageMode.COPY` (default) for real
            copies or `FileStorageMode.SYMLINK` for symbolic links to the original files - avoiding
            duplication on disk, but breaking if the original files are moved or deleted. The string
            values `"copy"` and `"symlink"` are also accepted.

    Returns:
        A list of `CocoSplitPaths` describing the files written for each split.
    """
    storage_mode = resolve_storage_mode(storage_mode, path_output=path_output)
    samples_modified_all = dataset.samples.with_row_index("id")

    if SampleField.ATTRIBUTION in samples_modified_all.columns:
        samples_modified_all = samples_modified_all.unnest(SampleField.ATTRIBUTION)
        license_table = (
            samples_modified_all["licenses"]
            .explode()
            .struct.unnest()
            .unique()
            .with_row_index("id")
            .select(["id", "name", "url"])
        )
        license_mapping = {lic["name"]: lic["id"] for lic in license_table.iter_rows(named=True)}
    else:
        license_mapping = None
        license_table = None

    if task_name is not None:
        task_info = dataset.info.get_task_by_name(task_name)
    else:
        # Auto derive the task to be used for COCO conversion as only one Bitmask/Bbox task can be present
        # in the coco format. Will first search for Bitmask (because COCO supports segmentation), then Bbox afterwards.
        tasks_info = dataset.info.get_tasks_by_primitive(Bitmask)
        if len(tasks_info) == 0:
            tasks_info = dataset.info.get_tasks_by_primitive(Bbox)
        if len(tasks_info) == 0:
            raise ValueError("No 'Bitmask' or 'Bbox' primitive found in dataset tasks for COCO conversion")
        if len(tasks_info) > 1:
            task_names = [task.name for task in tasks_info]
            raise ValueError(
                f"Found multiple tasks {task_names} for 'Bitmask'/'Bbox' primitive in dataset."
                " Please specify 'task_name'."
            )
        task_info = tasks_info[0]

    categories_list_dict = [
        {"id": i, "name": c, "supercategory": "NotDefined"} for i, c in enumerate(task_info.get_class_names() or [])
    ]
    category_mapping = {cat["name"]: cat["id"] for cat in categories_list_dict}

    split_names = samples_modified_all[SampleField.SPLIT].unique().to_list()

    list_split_paths = []
    for split_name in split_names:
        if coco_format_type == "roboflow":
            path_split = path_output / HAFNIA_TO_ROBOFLOW_SPLIT_NAME[split_name]
            split_paths = format_coco.CocoSplitPaths(
                split=split_name,
                path_images=path_split,
                path_instances_json=path_split / ROBOFLOW_ANNOTATION_FILE_NAME,
            )
        else:
            raise ValueError(f"The specified '{coco_format_type=}' is not supported.")
        samples_in_split = samples_modified_all.filter(pl.col(SampleField.SPLIT) == split_name)
        images_table, annotation_table = _convert_bbox_bitmask_to_coco_format(
            samples_modified=samples_in_split,
            license_mapping=license_mapping,
            task_info=task_info,
            category_mapping=category_mapping,  # type: ignore[arg-type]
        )

        split_paths.path_images.mkdir(parents=True, exist_ok=True)
        src_paths = images_table[COCO_KEY_FILE_NAME].to_list()
        new_relative_image_path = []
        for src_path in src_paths:
            dst_path = split_paths.path_images / Path(src_path).name
            new_relative_image_path.append(dst_path.relative_to(split_paths.path_images).as_posix())
            store_file(
                path_source=Path(src_path),
                path_destination=dst_path,
                storage_mode=storage_mode,
                allow_skip=True,
            )

        images_table_files_moved = images_table.with_columns(
            pl.Series(new_relative_image_path).alias(COCO_KEY_FILE_NAME)
        )
        split_labels = {
            "info": dataset.info.model_dump(mode="json"),
            "images": list(images_table_files_moved.iter_rows(named=True)),
            "categories": categories_list_dict,
            "annotations": list(annotation_table.iter_rows(named=True)),
        }
        if license_table is not None:
            split_labels["licenses"] = list(license_table.iter_rows(named=True))
        split_paths.path_instances_json.parent.mkdir(parents=True, exist_ok=True)
        split_paths.path_instances_json.write_text(json.dumps(split_labels))

        list_split_paths.append(split_paths)

    return list_split_paths


def _convert_bbox_bitmask_to_coco_format(
    samples_modified: pl.DataFrame,
    license_mapping: Optional[Dict[str, int]],
    task_info: TaskInfo,
    category_mapping: Dict[str, int],
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    if task_info.primitive not in [Bbox, Bitmask]:
        raise ValueError(f"Unsupported primitive '{task_info.primitive}' for COCO conversion")

    task_sample_field = task_info.primitive.column_name()
    select_image_table_columns = [
        pl.col("id"),
        pl.col(SampleField.WIDTH).alias("width"),
        pl.col(SampleField.HEIGHT).alias("height"),
        pl.col(SampleField.FILE_PATH).alias(COCO_KEY_FILE_NAME),
    ]

    if license_mapping is not None:
        samples_modified = samples_modified.with_columns(pl.col("licenses").list.first().struct.unnest())
        select_image_table_columns = select_image_table_columns + [
            pl.col("name").replace_strict(license_mapping, return_dtype=pl.Int64).alias("license"),
            pl.col("source_url").alias("flickr_url"),
            pl.col("source_url").alias("coco_url"),
            pl.col("date_captured"),
        ]

    images_table = samples_modified.select(select_image_table_columns)

    annotation_table_full = (
        samples_modified.select(
            pl.col("id").alias("image_id"),
            pl.col(SampleField.HEIGHT).alias("image_height"),
            pl.col(SampleField.WIDTH).alias("image_width"),
            pl.col(task_sample_field),
        )
        .explode(task_sample_field)
        .with_row_index("id")
        .unnest(task_sample_field)
    )

    if "meta" not in annotation_table_full.columns:
        annotation_table_full = annotation_table_full.with_columns(pl.lit(None).alias("meta"))
    iscrowd_list = [0 if row is None else (row.get("iscrowd") or 0) for row in annotation_table_full["meta"]]
    annotation_table_full = annotation_table_full.with_columns(pl.Series(iscrowd_list).alias("iscrowd"))

    if task_info.primitive == Bitmask:
        annotation_table = annotation_table_full.select(
            pl.col("id"),
            pl.col("image_id"),
            category_id=pl.col("class_name").replace_strict(category_mapping, return_dtype=pl.Int64),
            segmentation=pl.struct(
                counts=pl.col("rle_string"),
                size=pl.concat_arr(
                    pl.col("image_height"),
                    pl.col("image_width"),
                ),
            ),
            area=pl.col("area") * pl.col("image_height") * pl.col("image_width"),
            bbox=pl.concat_arr(
                pl.col("left"),  # bbox x coordinate
                pl.col("top"),  # bbox y coordinate
                pl.col("width"),  # bbox width
                pl.col("height"),  # bbox height
            ),
            iscrowd=pl.col("iscrowd"),
        )

    elif task_info.primitive == Bbox:
        annotation_table = annotation_table_full.select(
            pl.col("id"),
            pl.col("image_id"),
            category_id=pl.col("class_name").replace_strict(category_mapping, return_dtype=pl.Int64),
            segmentation=pl.lit([]),
            area=pl.col("height") * pl.col("width") * pl.col("image_height") * pl.col("image_width"),
            bbox=pl.concat_arr(
                pl.col("top_left_x") * pl.col("image_width"),  # x coordinate
                pl.col("top_left_y") * pl.col("image_height"),  # y coordinate
                pl.col("width") * pl.col("image_width"),  # width
                pl.col("height") * pl.col("image_height"),  # height
            ),
            iscrowd=pl.col("iscrowd"),
        )

    return images_table, annotation_table
