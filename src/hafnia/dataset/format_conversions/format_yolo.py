import shutil
from pathlib import Path
from typing import List, Optional

from PIL import Image
from rich.progress import track

from hafnia.dataset import primitives
from hafnia.dataset.dataset_names import SplitName
from hafnia.dataset.hafnia_dataset import DatasetInfo, HafniaDataset, Sample, TaskInfo

FILENAME_YOLO_CLASS_NAMES = "obj.names"
FILENAME_YOLO_IMAGES_TXT = "images.txt"


def get_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size  # (width, height)


def from_yolo_format(
    path_yolo_dataset: Path,
    split_name: str = SplitName.UNDEFINED,
    dataset_name: str = "yolo-dataset",
    filename_class_names: str = FILENAME_YOLO_CLASS_NAMES,
    filename_images_txt: str = FILENAME_YOLO_IMAGES_TXT,
):
    """
    Imports a YOLO (Darknet) formatted dataset as a HafniaDataset.
    """
    path_class_names = path_yolo_dataset / filename_class_names

    if split_name not in SplitName.all_split_names():
        raise ValueError(f"Invalid split name: {split_name}. Must be one of {SplitName.all_split_names()}")

    if not path_class_names.exists():
        raise FileNotFoundError(f"File with class names not found at '{path_class_names.resolve()}'.")

    class_names_text = path_class_names.read_text()
    if class_names_text.strip() == "":
        raise ValueError(f"File with class names not found at '{path_class_names.resolve()}' is empty")

    class_names = [class_name for class_name in class_names_text.splitlines() if class_name.strip() != ""]

    if len(class_names) == 0:
        raise ValueError(f"File with class names not found at '{path_class_names.resolve()}' has no class names")

    path_images_txt = path_yolo_dataset / filename_images_txt

    if not path_images_txt.exists():
        raise FileNotFoundError(f"File with images not found at '{path_images_txt.resolve()}'")

    images_txt_text = path_images_txt.read_text()
    if len(images_txt_text.strip()) == 0:
        raise ValueError(f"File is empty at '{path_images_txt.resolve()}'")

    image_paths_raw = [line.strip() for line in images_txt_text.splitlines()]

    samples: List[Sample] = []
    for image_path_raw in track(image_paths_raw):
        path_image = path_yolo_dataset / image_path_raw
        if not path_image.exists():
            raise FileNotFoundError(f"File with image not found at '{path_image.resolve()}'")
        width, height = get_image_size(path_image)

        path_label = path_image.with_suffix(".txt")
        if not path_label.exists():
            raise FileNotFoundError(f"File with labels not found at '{path_label.resolve()}'")

        boxes: List[primitives.Bbox] = []
        bbox_strings = path_label.read_text().splitlines()
        for bbox_string in bbox_strings:
            parts = bbox_string.strip().split()
            if len(parts) != 5:
                raise ValueError(f"Invalid bbox format in file {path_label.resolve()}: {bbox_string}")

            class_idx = int(parts[0])
            x_center, y_center, bbox_width, bbox_height = (float(value) for value in parts[1:5])

            top_left_x = x_center - bbox_width / 2
            top_left_y = y_center - bbox_height / 2

            bbox = primitives.Bbox(
                top_left_x=top_left_x,
                top_left_y=top_left_y,
                width=bbox_width,
                height=bbox_height,
                class_idx=class_idx,
                class_name=class_names[class_idx] if 0 <= class_idx < len(class_names) else None,
            )
            boxes.append(bbox)

        sample = Sample(
            file_path=path_image.absolute().as_posix(),
            height=height,
            width=width,
            split=split_name,
            objects=boxes,
        )
        samples.append(sample)

    tasks = [TaskInfo(primitive=primitives.Bbox, class_names=class_names)]
    info = DatasetInfo(dataset_name=dataset_name, tasks=tasks)
    hafnia_dataset = HafniaDataset.from_samples_list(samples, info=info)
    return hafnia_dataset


def as_yolo_format(
    dataset: HafniaDataset,
    path_export_yolo_dataset: Path,
    task_name: Optional[str] = None,
):
    """Exports a HafniaDataset as YOLO (Darknet) format."""
    bbox_task = dataset.info.get_task_by_task_name_and_primitive(task_name=task_name, primitive=primitives.Bbox)

    class_names = bbox_task.class_names or []
    if len(class_names) == 0:
        raise ValueError(
            f"Hafnia dataset task '{bbox_task.name}' has no class names defined. This is required for YOLO export."
        )
    path_export_yolo_dataset.mkdir(parents=True, exist_ok=True)
    path_class_names = path_export_yolo_dataset / FILENAME_YOLO_CLASS_NAMES
    path_class_names.write_text("\n".join(class_names))

    path_data_folder = path_export_yolo_dataset / "data"
    path_data_folder.mkdir(parents=True, exist_ok=True)
    image_paths: List[str] = []
    for sample_dict in dataset:
        sample = Sample(**sample_dict)
        path_image_src = Path(sample.file_path)
        path_image_dst = path_data_folder / path_image_src.name
        shutil.copy2(path_image_src, path_image_dst)
        image_paths.append(path_image_dst.relative_to(path_export_yolo_dataset).as_posix())
        path_label = path_image_dst.with_suffix(".txt")
        bboxes = sample.objects or []
        bbox_strings = [bbox_to_yolo_format(bbox) for bbox in bboxes]
        path_label.write_text("\n".join(bbox_strings))

    path_images_txt = path_export_yolo_dataset / FILENAME_YOLO_IMAGES_TXT
    path_images_txt.write_text("\n".join(image_paths))


def bbox_to_yolo_format(bbox: primitives.Bbox) -> str:
    """
    From hafnia bbox to yolo bbox string conversion
    Both yolo and hafnia use normalized coordinates [0, 1]
        Hafnia: top_left_x, top_left_y, width, height
        Yolo (darknet): "<object-class> <x_center> <y_center> <width> <height>"
    Example (3 bounding boxes):
        1 0.716797 0.395833 0.216406 0.147222
        0 0.687109 0.379167 0.255469 0.158333
        1 0.420312 0.395833 0.140625 0.166667
    """
    x_center = bbox.top_left_x + bbox.width / 2
    y_center = bbox.top_left_y + bbox.height / 2
    return f"{bbox.class_idx} {x_center} {y_center} {bbox.width} {bbox.height}"
