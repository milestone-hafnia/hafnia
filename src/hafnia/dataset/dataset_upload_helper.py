from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import boto3
import polars as pl
from pydantic import BaseModel, ConfigDict

import hafnia.dataset.primitives.bbox
import hafnia.dataset.primitives.bitmask
import hafnia.dataset.primitives.classification
import hafnia.dataset.primitives.polygon
import hafnia.dataset.primitives.segmentation
from cli.config import Config
from hafnia.dataset import primitives
from hafnia.dataset.dataset_names import (
    ColumnName,
    DatasetVariant,
    DeploymentStage,
    FieldName,
    SplitName,
)
from hafnia.dataset.hafnia_dataset import HafniaDataset, TaskInfo
from hafnia.dataset.primitives.primitive import Primitive
from hafnia.http import post
from hafnia.log import user_logger
from hafnia.platform import get_dataset_id


def generate_bucket_name(dataset_name: str, deployment_stage: DeploymentStage) -> str:
    # TODO: When moving to versioning we do NOT need 'staging' and 'production' specific buckets
    # and the new name convention should be: f"hafnia-dataset-{dataset_name}"
    return f"mdi-{deployment_stage.value}-{dataset_name}"


class DbDataset(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    model_config = ConfigDict(use_enum_values=True)  # To parse Enum values as strings
    name: str
    data_captured_start: Optional[datetime] = None
    data_captured_end: Optional[datetime] = None
    data_received_start: Optional[datetime] = None
    data_received_end: Optional[datetime] = None
    latest_update: Optional[datetime] = None
    license_citation: Optional[str] = None
    version: Optional[str] = None
    s3_bucket_name: Optional[str] = None
    annotation_date: Optional[datetime] = None
    annotation_project_id: Optional[str] = None
    annotation_dataset_id: Optional[str] = None
    annotation_ontology: Optional[str] = None
    dataset_variants: Optional[List[DbDatasetVariant]] = None
    split_annotations_reports: Optional[List[DbSplitAnnotationsReport]] = None
    dataset_images: Optional[List[DatasetImage]] = None


class DbDatasetVariant(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    model_config = ConfigDict(use_enum_values=True)  # To parse Enum values as strings
    variant_type: VariantTypeChoices  # Required
    upload_date: Optional[datetime] = None
    size_bytes: Optional[int] = None
    data_type: Optional[str] = None
    number_of_data_items: Optional[int] = None
    resolutions: Optional[List[DbResolution]] = None
    duration: Optional[float] = None
    duration_average: Optional[float] = None
    frame_rate: Optional[float] = None
    bit_rate: Optional[float] = None
    n_cameras: Optional[int] = None


class DbAnnotatedObject(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    model_config = ConfigDict(use_enum_values=True)  # To parse Enum values as strings
    name: str
    entity_type: EntityTypeChoices


class DbAnnotatedObjectReport(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    model_config = ConfigDict(use_enum_values=True)  # To parse Enum values as strings
    obj: DbAnnotatedObject
    unique_obj_ids: Optional[int] = None
    obj_instances: Optional[int] = None
    average_count_per_image: Optional[float] = None
    avg_area: Optional[float] = None
    min_area: Optional[float] = None
    max_area: Optional[float] = None
    annotation_type: Optional[List[DbAnnotationType]] = None


class DbDistributionValue(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    distribution_category: DbDistributionCategory
    percentage: Optional[float] = None

    @staticmethod
    def from_names(type_name: str, category_name: str, percentage: Optional[float]) -> "DbDistributionValue":
        dist_type = DbDistributionType(name=type_name)
        dist_category = DbDistributionCategory(distribution_type=dist_type, name=category_name)
        return DbDistributionValue(distribution_category=dist_category, percentage=percentage)


class DbSplitAnnotationsReport(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    model_config = ConfigDict(use_enum_values=True)  # To parse Enum values as strings
    variant_type: VariantTypeChoices  # Required
    split: str  # Required
    sample_count: Optional[int] = None
    annotated_object_reports: Optional[List[DbAnnotatedObjectReport]] = None
    distribution_values: Optional[List[DbDistributionValue]] = None


class DbDistributionCategory(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    distribution_type: DbDistributionType
    name: str


class DbAnnotationType(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    name: str


class AnnotationType(Enum):
    ImageClassification = "Image Classification"
    ObjectDetection = "Object Detection"
    SegmentationMask = "Segmentation Mask"
    ImageCaptioning = "Image Captioning"
    InstanceSegmentation = "Instance Segmentation"


class DbResolution(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    height: int
    width: int


class DataTypeChoices(str, Enum):  # Should match `DataTypeChoices` in `dipdatalib::src/apps/datasets/models.py`
    images = "images"
    video_frames = "video_frames"
    video_clips = "video_clips"


class VariantTypeChoices(str, Enum):  # Should match `VariantType` in `dipdatalib::src/apps/datasets/models.py`
    ORIGINAL = "original"
    HIDDEN = "hidden"
    SAMPLE = "sample"


class SplitChoices(str, Enum):  # Should match `SplitChoices` in `dipdatalib::src/apps/datasets/models.py`
    FULL = "full"
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


class EntityTypeChoices(str, Enum):  # Should match `EntityTypeChoices` in `dipdatalib::src/apps/datasets/models.py`
    OBJECT = "OBJECT"
    EVENT = "EVENT"


class DatasetImage(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    img: str


class DbDistributionType(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    name: str


VARIANT_TYPE_MAPPING: Dict[
    DatasetVariant, VariantTypeChoices
] = {  # Conider making DatasetVariant & VariantTypeChoices into one
    DatasetVariant.DUMP: VariantTypeChoices.ORIGINAL,
    DatasetVariant.HIDDEN: VariantTypeChoices.HIDDEN,
    DatasetVariant.SAMPLE: VariantTypeChoices.SAMPLE,
}

SPLIT_CHOICE_MAPPING: Dict[SplitChoices, List[str]] = {
    SplitChoices.FULL: SplitName.valid_splits(),
    SplitChoices.TRAIN: [SplitName.TRAIN],
    SplitChoices.TEST: [SplitName.TEST],
    SplitChoices.VALIDATION: [SplitName.VAL],
}


def get_folder_size(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"The path {path} does not exist.")
    return sum([path.stat().st_size for path in path.rglob("*")])


def upload_to_hafnia_dataset_detail_page(dataset_update: DbDataset) -> dict:
    cfg = Config()
    dataset_details = dataset_update.model_dump_json()
    data = upload_dataset_details(cfg=cfg, data=dataset_details, dataset_name=dataset_update.name)
    return data


def upload_dataset_details(cfg: Config, data: str, dataset_name: str) -> dict:
    dataset_endpoint = cfg.get_platform_endpoint("datasets")
    dataset_id = get_dataset_id(dataset_name, dataset_endpoint, cfg.api_key)

    import_endpoint = f"{dataset_endpoint}/{dataset_id}/import"
    headers = {"Authorization": cfg.api_key}

    user_logger.info("Importing dataset details. This may take up to 30 seconds...")
    data = post(endpoint=import_endpoint, headers=headers, data=data)  # type: ignore[assignment]
    return data  # type: ignore[return-value]


def get_resolutions(dataset: HafniaDataset, max_resolutions_selected: int = 8) -> List[DbResolution]:
    unique_resolutions = (
        dataset.samples.select([pl.col("height"), pl.col("width")]).unique().sort(by=["height", "width"])
    )
    if len(unique_resolutions) > max_resolutions_selected:
        skip_size = len(unique_resolutions) // max_resolutions_selected
        unique_resolutions = unique_resolutions.gather_every(skip_size)
    resolutions = [DbResolution(height=res["height"], width=res["width"]) for res in unique_resolutions.to_dicts()]
    return resolutions


def has_primitive(dataset: Union[HafniaDataset, pl.DataFrame], PrimitiveType: Type[Primitive]) -> bool:
    col_name = PrimitiveType.column_name()
    table = dataset.samples if isinstance(dataset, HafniaDataset) else dataset
    if col_name not in table.columns:
        user_logger.warning(f"Warning: No field called '{col_name}' was found for '{PrimitiveType.__name__}'.")
        return False

    if table[col_name].dtype == pl.Null:
        return False

    return True


def calculate_distribution_values(
    dataset_split: pl.DataFrame, distribution_tasks: Optional[List[TaskInfo]]
) -> List[DbDistributionValue]:
    distribution_tasks = distribution_tasks or []

    if len(distribution_tasks) == 0:
        return []
    classification_column = hafnia.dataset.primitives.classification.Classification.column_name()
    classifications = dataset_split.select(pl.col(classification_column).explode())
    classifications = classifications.filter(pl.col(classification_column).is_not_null()).unnest(classification_column)
    classifications = classifications.filter(
        pl.col(FieldName.TASK_NAME).is_in([task.name for task in distribution_tasks])
    )
    dist_values = []
    for (task_name,), task_group in classifications.group_by(FieldName.TASK_NAME):
        distribution_type = DbDistributionType(name=task_name)
        n_annotated_total = len(task_group)
        for (class_name,), class_group in task_group.group_by(FieldName.CLASS_NAME):
            class_count = len(class_group)

            dist_values.append(
                DbDistributionValue(
                    distribution_category=DbDistributionCategory(distribution_type=distribution_type, name=class_name),
                    percentage=class_count / n_annotated_total * 100,
                )
            )
    dist_values = sorted(
        dist_values,
        key=lambda x: (
            x.distribution_category.distribution_type.name,
            x.distribution_category.name,
        ),
    )
    return dist_values


def s3_based_fields(bucket_name: str, variant_type: DatasetVariant, session: boto3.Session) -> tuple[datetime, int]:
    client = session.client("s3")
    file_objects = client.list_objects_v2(Bucket=bucket_name, Prefix=variant_type.value)["Contents"]
    last_modified = sorted([file_obj["LastModified"] for file_obj in file_objects])[-1]
    size = sum([file_obj["Size"] for file_obj in file_objects])
    return last_modified, size


def dataset_info_from_dataset(
    dataset: HafniaDataset,
    deployment_stage: DeploymentStage,
    path_sample: Optional[Path],
    path_hidden: Optional[Path],
) -> DbDataset:
    dataset_variants = []
    dataset_reports = []
    dataset_meta_info = dataset.info.meta or {}

    path_and_variant: List[Tuple[Path, DatasetVariant]] = []
    if path_sample is not None:
        path_and_variant.append((path_sample, DatasetVariant.SAMPLE))

    if path_hidden is not None:
        path_and_variant.append((path_hidden, DatasetVariant.HIDDEN))

    if len(path_and_variant) == 0:
        raise ValueError("At least one path must be provided for sample or hidden dataset.")

    for path_dataset, variant_type in path_and_variant:
        if variant_type == DatasetVariant.SAMPLE:
            dataset_variant = dataset.create_sample_dataset()
        else:
            dataset_variant = dataset

        size_bytes = get_folder_size(path_dataset)
        dataset_variants.append(
            DbDatasetVariant(
                variant_type=VARIANT_TYPE_MAPPING[variant_type],  # type: ignore[index]
                # upload_date: Optional[datetime] = None
                size_bytes=size_bytes,
                data_type=DataTypeChoices.images,
                number_of_data_items=len(dataset_variant),
                resolutions=get_resolutions(dataset_variant, max_resolutions_selected=8),
                duration=dataset_meta_info.get("duration", None),
                duration_average=dataset_meta_info.get("duration_average", None),
                frame_rate=dataset_meta_info.get("frame_rate", None),
                # bit_rate: Optional[float] = None
                n_cameras=dataset_meta_info.get("n_cameras", None),
            )
        )

        for split_name in SplitChoices:
            split_names = SPLIT_CHOICE_MAPPING[split_name]
            dataset_split = dataset_variant.samples.filter(pl.col(ColumnName.SPLIT).is_in(split_names))

            distribution_values = calculate_distribution_values(
                dataset_split=dataset_split,
                distribution_tasks=dataset.info.distributions,
            )
            report = DbSplitAnnotationsReport(
                variant_type=VARIANT_TYPE_MAPPING[variant_type],  # type: ignore[index]
                split=split_name,
                sample_count=len(dataset_split),
                distribution_values=distribution_values,
            )

            object_reports: List[DbAnnotatedObjectReport] = []
            primitive_columns = [tPrimtive.column_name() for tPrimtive in primitives.PRIMITIVE_TYPES]
            if has_primitive(dataset_split, PrimitiveType=hafnia.dataset.primitives.bbox.Bbox):
                bbox_column_name = hafnia.dataset.primitives.bbox.Bbox.column_name()
                drop_columns = [col for col in primitive_columns if col != bbox_column_name]
                drop_columns.append(FieldName.META)
                df_per_instance = dataset_split.rename({"height": "image.height", "width": "image.width"})
                df_per_instance = df_per_instance.explode(bbox_column_name).drop(drop_columns).unnest(bbox_column_name)

                # Calculate area of bounding boxes
                df_per_instance = df_per_instance.with_columns((pl.col("height") * pl.col("width")).alias("area"))

                annotation_type = DbAnnotationType(name=AnnotationType.ObjectDetection.value)
                for (class_name,), class_group in df_per_instance.group_by(FieldName.CLASS_NAME):
                    if class_name is None:
                        continue
                    object_reports.append(
                        DbAnnotatedObjectReport(
                            obj=DbAnnotatedObject(
                                name=class_name,
                                entity_type=EntityTypeChoices.OBJECT.value,
                            ),
                            unique_obj_ids=class_group[FieldName.OBJECT_ID].n_unique(),
                            obj_instances=len(class_group),
                            annotation_type=[annotation_type],
                            avg_area=class_group["area"].mean(),
                            min_area=class_group["area"].min(),
                            max_area=class_group["area"].max(),
                            average_count_per_image=len(class_group) / class_group[ColumnName.SAMPLE_INDEX].n_unique(),
                        )
                    )

            if has_primitive(dataset_split, PrimitiveType=hafnia.dataset.primitives.classification.Classification):
                annotation_type = DbAnnotationType(name=AnnotationType.ImageClassification.value)
                col_name = hafnia.dataset.primitives.classification.Classification.column_name()
                classification_tasks = [
                    task.name
                    for task in dataset.info.tasks
                    if task.primitive == hafnia.dataset.primitives.classification.Classification
                ]
                has_classification_data = dataset_split[col_name].dtype != pl.List(pl.Null)
                if has_classification_data:
                    classification_df = dataset_split.select(col_name).explode(col_name).unnest(col_name)

                    # Include only classification tasks that are defined in the dataset info
                    classification_df = classification_df.filter(
                        pl.col(FieldName.TASK_NAME).is_in(classification_tasks)
                    )

                    for (
                        task_name,
                        class_name,
                    ), class_group in classification_df.group_by(FieldName.TASK_NAME, FieldName.CLASS_NAME):
                        if class_name is None:
                            continue
                        if task_name == hafnia.dataset.primitives.classification.Classification.default_task_name():
                            display_name = class_name  # Prefix class name with task name
                        else:
                            display_name = f"{task_name}.{class_name}"
                        object_reports.append(
                            DbAnnotatedObjectReport(
                                obj=DbAnnotatedObject(
                                    name=display_name,
                                    entity_type=EntityTypeChoices.EVENT.value,
                                ),
                                unique_obj_ids=len(
                                    class_group
                                ),  # Unique object IDs are not applicable for classification
                                obj_instances=len(class_group),
                                annotation_type=[annotation_type],
                            )
                        )

            if has_primitive(dataset_split, PrimitiveType=hafnia.dataset.primitives.segmentation.Segmentation):
                raise NotImplementedError("Not Implemented yet")

            if has_primitive(dataset_split, PrimitiveType=hafnia.dataset.primitives.bitmask.Bitmask):
                col_name = hafnia.dataset.primitives.bitmask.Bitmask.column_name()
                drop_columns = [col for col in primitive_columns if col != col_name]
                drop_columns.append(FieldName.META)
                df_per_instance = dataset_split.rename({"height": "image.height", "width": "image.width"})
                df_per_instance = df_per_instance.explode(col_name).drop(drop_columns).unnest(col_name)

                min_area = df_per_instance["area"].min() if "area" in df_per_instance.columns else None
                max_area = df_per_instance["area"].max() if "area" in df_per_instance.columns else None
                avg_area = df_per_instance["area"].mean() if "area" in df_per_instance.columns else None

                annotation_type = DbAnnotationType(name=AnnotationType.InstanceSegmentation)
                for (class_name,), class_group in df_per_instance.group_by(FieldName.CLASS_NAME):
                    if class_name is None:
                        continue
                    object_reports.append(
                        DbAnnotatedObjectReport(
                            obj=DbAnnotatedObject(
                                name=class_name,
                                entity_type=EntityTypeChoices.OBJECT.value,
                            ),
                            unique_obj_ids=class_group[FieldName.OBJECT_ID].n_unique(),
                            obj_instances=len(class_group),
                            annotation_type=[annotation_type],
                            average_count_per_image=len(class_group) / class_group[ColumnName.SAMPLE_INDEX].n_unique(),
                            avg_area=avg_area,
                            min_area=min_area,
                            max_area=max_area,
                        )
                    )

            if has_primitive(dataset_split, PrimitiveType=hafnia.dataset.primitives.polygon.Polygon):
                raise NotImplementedError("Not Implemented yet")

            # Sort object reports by name to more easily compare between versions
            object_reports = sorted(object_reports, key=lambda x: x.obj.name)  # Sort object reports by name
            report.annotated_object_reports = object_reports

            if report.distribution_values is None:
                report.distribution_values = []

            dataset_reports.append(report)
    dataset_name = dataset.info.dataset_name
    bucket_sample = generate_bucket_name(dataset_name, deployment_stage=deployment_stage)
    dataset_info = DbDataset(
        name=dataset_name,
        version=dataset.info.version,
        s3_bucket_name=bucket_sample,
        dataset_variants=dataset_variants,
        split_annotations_reports=dataset_reports,
        license_citation=dataset_meta_info.get("license_citation", None),
        data_captured_start=dataset_meta_info.get("data_captured_start", None),
        data_captured_end=dataset_meta_info.get("data_captured_end", None),
        data_received_start=dataset_meta_info.get("data_received_start", None),
        data_received_end=dataset_meta_info.get("data_received_end", None),
        annotation_project_id=dataset_meta_info.get("annotation_project_id", None),
        annotation_dataset_id=dataset_meta_info.get("annotation_dataset_id", None),
    )

    return dataset_info
