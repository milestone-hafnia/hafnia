from pathlib import Path

import rich
from PIL import Image

from hafnia.dataset.dataset_names import SplitName
from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe
from hafnia.dataset.hafnia_dataset import HafniaDataset, Sample

path_demo = Path(".data/sprint_6_demo")

### 1) Video-based storage format ####
dataset_images = HafniaDataset.from_path(path_demo / "dataset_image_based")
for sample_dict in dataset_images:
    sample = Sample(**sample_dict)
    rich.print(sample)
    image = sample.read_image()  ## Reads image from file
    image_with_annotations = sample.draw_annotations(image=image)  ## Draw annotations on frame

    Image.fromarray(image_with_annotations).save(path_demo / "from_image.png")
    break


dataset_videos = HafniaDataset.from_path(path_demo / "dataset_video_based")

for sample_dict in dataset_videos:
    sample = Sample(**sample_dict)

    image = sample.read_image()  ## Reads frame from video
    image_with_annotations = sample.draw_annotations(image=image)  ## Draw annotations on frame

    Image.fromarray(image_with_annotations).save(path_demo / "from_video.png")
    break

# Create image-based dataset from video-based dataset (1 minute per video)
path_images_from_video = path_demo / "dataset_image_based_from_video"
dataset_images_converted = dataset_videos.convert_to_image_storage_format(
    path_images_from_video, reextract_frames=False
)
dataset_images_converted.write_annotations(path_images_from_video)

#### 2) Imports and exporters ####
##### YOLO FORMAT ######
dataset_od = HafniaDataset.from_name("midwest-vehicle-detection")

# Export to YOLO format
path_export_yolo = path_demo / "exported_in_yolo_format"
dataset_od.to_yolo_format(path_export_yolo)

# Import from YOLO format
yolo_reloaded = HafniaDataset.from_yolo_format(path_export_yolo)

##### Image classification folder #####
dataset_mnist = HafniaDataset.from_name("mnist")

# Export to image classification folder
path_export_image_classification = path_demo / "exported_in_image_classification_folder"
dataset_mnist.to_image_classification_folder(path_export_image_classification)

# Import from image classification folder
dataset_mnist_reloaded = HafniaDataset.from_image_classification_folder(path_export_image_classification, split="train")


##### COCO FORMAT #####
path_export_coco = path_demo / "exported_in_coco_format"


#### 3) Full Public dataset loaders ####

# Supported: "mnist", "caltech-101", "caltech-256", "cifar10", "cifar100"
dataset_caltech = HafniaDataset.from_name_public_dataset("caltech-101", n_samples=10)


#### 4) Open datasets are created from recipes ####
split_ratios = {SplitName.TRAIN: 0.8, SplitName.VAL: 0.1, SplitName.TEST: 0.1}
recipe = (
    DatasetRecipe.from_name_public_dataset("caltech-101", n_samples=100)
    .splits_by_ratios(split_ratios)
    .define_sample_set_by_size(n_samples=50)
)


dataset_caltech_101 = recipe.build()

print(recipe.as_python_code())
