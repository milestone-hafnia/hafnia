from typing import List, Optional

import datasets
import torch
import torchvision
from flatten_dict import flatten
from PIL import Image, ImageDraw, ImageFont
from torchvision import tv_tensors
from torchvision import utils as tv_utils
from torchvision.transforms import v2


class TorchvisionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        transforms=None,
        classification_tasks: Optional[List] = None,
        object_tasks: Optional[List] = None,
        segmentation_tasks: Optional[List] = None,
        keep_metadata: bool = False,
    ):
        self.dataset = hf_dataset
        self.transforms = transforms
        self.object_tasks = object_tasks or ["objects"]
        self.segmentation_tasks = segmentation_tasks or ["segmentation"]
        self.classification_tasks = classification_tasks or ["classification"]
        self.keep_metadata = keep_metadata

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # For now, we expect a dataset to always have an image field
        image = tv_tensors.Image(sample.pop("image"))

        img_shape = image.shape[-2:]
        target = {}

        for segmentation_task in self.segmentation_tasks:
            if segmentation_task in sample:
                target[f"{segmentation_task}.mask"] = tv_tensors.Mask(sample[segmentation_task].pop("mask"))

        for classification_task in self.classification_tasks:
            if classification_task in sample:
                target[f"{classification_task}.class_idx"] = sample[classification_task].pop("class_idx")

        for object_task in self.object_tasks:
            if object_task in sample:
                bboxes_list = sample[object_task].pop("bbox")
                bboxes = tv_tensors.BoundingBoxes(bboxes_list, format="XYWH", canvas_size=img_shape)
                if bboxes.numel() == 0:
                    bboxes = bboxes.reshape(-1, 4)
                target[f"{object_task}.bbox"] = bboxes
                target[f"{object_task}.class_idx"] = torch.tensor(sample[object_task].pop("class_idx"))

        if self.keep_metadata:
            target.update(flatten(sample, reducer="dot"))

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.dataset)


def draw_image_classification(visualize_image: torch.Tensor, text_label: str) -> torch.Tensor:
    max_dim = max(visualize_image.shape[-2:])
    font_size = max(int(max_dim * 0.1), 10)  # Minimum font size of 10
    txt_font = ImageFont.load_default(font_size)
    dummie_draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    _, _, w, h = dummie_draw.textbbox((0, 0), text=text_label, font=txt_font)  # type: ignore[arg-type]

    text_image = Image.new("RGB", (int(w), int(h)))
    draw = ImageDraw.Draw(text_image)
    draw.text((0, 0), text=text_label, font=txt_font)  # type: ignore[arg-type]
    text_tensor = v2.functional.to_image(text_image)

    height = text_tensor.shape[-2] + visualize_image.shape[-2]
    width = max(text_tensor.shape[-1], visualize_image.shape[-1])
    visualize_image_new = torch.zeros((3, height, width), dtype=visualize_image.dtype)
    shift_w = (width - visualize_image.shape[-1]) // 2
    visualize_image_new[:, : visualize_image.shape[-2], shift_w : shift_w + visualize_image.shape[-1]] = visualize_image
    shift_w = (width - text_tensor.shape[-1]) // 2
    shift_h = visualize_image.shape[-2]
    visualize_image_new[:, shift_h : shift_h + text_tensor.shape[-2], shift_w : shift_w + text_tensor.shape[-1]] = (
        text_tensor
    )
    visualize_image = visualize_image_new
    return visualize_image


def draw_image_and_targets(
    image: torch.Tensor,
    targets,
    detection_tasks: Optional[List[str]] = None,
    segmentation_tasks: Optional[List[str]] = None,
    classification_tasks: Optional[List[str]] = None,
) -> torch.Tensor:
    detection_tasks = detection_tasks or ["objects"]
    segmentation_tasks = segmentation_tasks or ["segmentation"]
    classification_tasks = classification_tasks or ["classification"]

    visualize_image = image.clone()
    if visualize_image.is_floating_point():
        visualize_image = image - torch.min(image)
        visualize_image = visualize_image / visualize_image.max()

    visualize_image = v2.functional.to_dtype(visualize_image, torch.uint8, scale=True)

    for object_task in detection_tasks:
        bbox_field = f"{object_task}.bbox"
        if bbox_field in targets:
            hugging_face_format = "xywh"
            bbox = torchvision.ops.box_convert(targets[bbox_field], in_fmt=hugging_face_format, out_fmt="xyxy")
            class_names_field = f"{object_task}.class_name"
            class_names = targets.get(class_names_field, None)
            visualize_image = tv_utils.draw_bounding_boxes(visualize_image, bbox, labels=class_names, width=2)

    for segmentation_task in segmentation_tasks:
        mask_field = f"{segmentation_task}.mask"
        if mask_field in targets:
            mask = targets[mask_field].squeeze(0)
            masks_list = [mask == value for value in mask.unique()]
            masks = torch.stack(masks_list, dim=0).to(torch.bool)
            visualize_image = tv_utils.draw_segmentation_masks(visualize_image, masks=masks, alpha=0.5)

    for classification_task in classification_tasks:
        classification_field = f"{classification_task}.class_idx"
        if classification_field in targets:
            text_label = f"[{targets[classification_field]}]"
            classification_name_field = f"{classification_task}.class_name"
            if classification_name_field in targets:
                text_label = text_label + f" {targets[classification_name_field]}"
            visualize_image = draw_image_classification(visualize_image, text_label)
    return visualize_image


class TorchVisionCollateFn:
    def __init__(self, skip_stacking: Optional[List] = None):
        if skip_stacking is None:
            skip_stacking = []
        self.skip_stacking_list = skip_stacking

    def __call__(self, batch):
        images, targets = tuple(zip(*batch, strict=False))
        if "image" not in self.skip_stacking_list:
            images = torch.stack(images)

        targets_modified = {k: [d[k] for d in targets] for k in targets[0]}
        for key_name, item_values in targets_modified.items():
            if key_name not in self.skip_stacking_list:
                first_element = item_values[0]
                if isinstance(first_element, torch.Tensor):
                    item_values = torch.stack(item_values)
                elif isinstance(first_element, (int, float)):
                    item_values = torch.tensor(item_values)
                elif isinstance(first_element, (str, list)):
                    # Skip stacking for certain types such as strings and lists
                    pass
                if isinstance(first_element, tv_tensors.Mask):
                    item_values = tv_tensors.Mask(item_values)
                elif isinstance(first_element, tv_tensors.Image):
                    item_values = tv_tensors.Image(item_values)
                elif isinstance(first_element, tv_tensors.BoundingBoxes):
                    item_values = tv_tensors.BoundingBoxes(item_values)
                targets_modified[key_name] = item_values

        return images, targets_modified
