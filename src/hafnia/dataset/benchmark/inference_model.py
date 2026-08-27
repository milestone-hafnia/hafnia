from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
from PIL import Image

from hafnia.dataset.hafnia_dataset_types import ModelInfo
from hafnia.dataset.primitives import Primitive

ImageType = Union[str, Image.Image, np.ndarray]


class InferenceModel(ABC):
    """Abstract base class for inference models."""

    @abstractmethod
    def predict(
        self,
        image: ImageType,
        sample_dict: Optional[dict] = None,
    ) -> List[Primitive]:
        """
        Perform prediction on a single image.

        Args:
            image: Input image for prediction.
            sample_dict: Optional dictionary containing additional information about the sample.

        Returns:
            Prediction results for the image.
        """
        pass

    def predict_batch(
        self,
        images: List[ImageType],
        sample_dicts: Optional[List[dict]] = None,
    ) -> List[List[Primitive]]:
        """
        Perform prediction on a batch of images.

        The default implementation calls 'predict' for each image. Override this method to
        implement actual batched inference.

        Args:
            images: Input images for prediction.
            sample_dicts: Optional list of dictionaries containing additional information about
                each sample. Must have the same length as 'images'.

        Returns:
            Prediction results for each image - one list of primitives per image.
        """
        if sample_dicts is None:
            sample_dicts = [None] * len(images)  # type: ignore[list-item]
        elif len(sample_dicts) != len(images):
            raise ValueError(
                f"Length of 'sample_dicts' ({len(sample_dicts)}) must match length of 'images' ({len(images)})."
            )
        return [self.predict(image, sample_dict=sample_dict) for image, sample_dict in zip(images, sample_dicts)]

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """
        Get the tasks that this model can perform.

        Returns:
            Tasks supported by the model.
        """
        pass
