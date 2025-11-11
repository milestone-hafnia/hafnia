from pathlib import Path

import pytest

from hafnia.dataset.dataset_details_uploader import dataset_details_from_hafnia_dataset
from hafnia.dataset.dataset_names import DeploymentStage, SampleField
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.primitives.classification import Classification
from tests import helper_testing


@pytest.mark.parametrize("dataset_name", helper_testing.MICRO_DATASETS)
def test_dataset_details_from_hafnia_dataset(dataset_name: str, tmp_path: Path):
    path_dataset = helper_testing.get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    dataset = HafniaDataset.from_path(path_dataset)
    classification_tasks = dataset.info.get_tasks_by_primitive(Classification)
    distribution_task_names = [task.name or task.primitive.default_task_name() for task in classification_tasks]
    gallery_image_names = [dataset.samples[SampleField.FILE_PATH].str.split("/").list.last().sort()[0]]
    dataset_info = dataset_details_from_hafnia_dataset(
        dataset=dataset,
        deployment_stage=DeploymentStage.STAGING,
        path_sample=path_dataset,
        path_hidden=None,
        path_gallery_images=tmp_path / "gallery_images",
        gallery_image_names=gallery_image_names,
        distribution_task_names=distribution_task_names,
    )
    # Check if dataset info can be serialized to JSON
    dataset_info_json = dataset_info.model_dump_json()  # noqa: F841
