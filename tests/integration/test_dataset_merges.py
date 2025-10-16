import pytest

from hafnia.dataset.dataset_names import OPS_REMOVE_CLASS, ColumnName
from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.utils import is_hafnia_configured


def test_merge_midwest_and_coco_datasets():
    if not is_hafnia_configured():
        pytest.skip("Hafnia platform not configured. Skipping CLI integration test.")

    force_download = False
    mappings_coco = {
        "person": "Person",
        "bicycle": "Vehicle",
        "car": "Vehicle",
        "motorcycle": "Vehicle",
        "bus": "Vehicle",
        "train": "Vehicle",
        "truck": "Vehicle",
    }
    mapping_midwest = {
        "Person": "Person",
        "Vehicle.*": "Vehicle",
        "Vehicle.Trailer": OPS_REMOVE_CLASS,
    }
    coco_name = "coco-2017-tiny"
    midwest_name = "midwest-vehicle-detection-tiny"

    coco = HafniaDataset.from_name(coco_name, force_redownload=force_download)
    coco_remapped = coco.class_mapper(class_mapping=mappings_coco, method="remove_undefined", task_name="bboxes")

    midwest = HafniaDataset.from_name(midwest_name, force_redownload=force_download)
    midwest_remapped = midwest.class_mapper(class_mapping=mapping_midwest, task_name="bboxes")

    merged_dataset = HafniaDataset.merge(midwest_remapped, coco_remapped)
    merged_dataset.check_dataset()

    # Recreate as recipe
    dataset_recipe = DatasetRecipe.from_merger(
        recipes=[
            DatasetRecipe.from_name(name=midwest_name).class_mapper(class_mapping=mapping_midwest, task_name="bboxes"),
            DatasetRecipe.from_name(name=coco_name).class_mapper(
                class_mapping=mappings_coco, method="remove_undefined", task_name="bboxes"
            ),
        ]
    )

    dataset_from_recipe = dataset_recipe.build()
    dataset_from_recipe.check_dataset()

    # Ensure dataset names are
    expected_dataset_names = {coco_name, midwest_name}
    actual_dataset_names = set(merged_dataset.samples[ColumnName.DATASET_NAME].unique())
    assert actual_dataset_names == expected_dataset_names, (
        f"The '{ColumnName.DATASET_NAME}' should contain the original dataset names {expected_dataset_names}. "
        f"But found: {actual_dataset_names}"
    )
