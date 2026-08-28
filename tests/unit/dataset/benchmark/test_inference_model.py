from typing import List, Optional, Tuple

import pytest

from hafnia.dataset.benchmark.inference_model import ImageType, InferenceModel
from hafnia.dataset.hafnia_dataset_types import ModelInfo, TaskInfo
from hafnia.dataset.primitives import Classification, Primitive


class RecordingInferenceModel(InferenceModel):
    """Records each 'predict' call and returns Cat Classification"""

    def __init__(self):
        self.calls: List[Tuple[ImageType, Optional[dict]]] = []

    def predict(self, image: ImageType, sample_dict: Optional[dict] = None) -> List[Primitive]:
        self.calls.append((image, sample_dict))
        return [Classification(class_name="Cat", ground_truth=False, confidence=0.5)]

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name="RecordingModel",
            tasks=[TaskInfo.from_class_names(primitive=Classification, class_names=["Cat", "Dog"])],
        )


def test_predict_batch_default_calls_predict_per_image():
    """The default 'predict_batch' loops over 'predict' and returns one list of primitives per image."""
    model = RecordingInferenceModel()
    images = ["image0", "image1", "image2"]
    sample_dicts = [{"index": 0}, {"index": 1}, {"index": 2}]

    predictions = model.predict_batch(images, sample_dicts=sample_dicts)

    assert model.calls == list(zip(images, sample_dicts)), "Each image should be paired with its own 'sample_dict'"
    assert len(predictions) == len(images), "Expected one list of primitives per image"
    assert [[p.class_name for p in preds] for preds in predictions] == [["Cat"], ["Cat"], ["Cat"]]


def test_predict_batch_without_sample_dicts():
    """'sample_dicts=None' forwards 'None' to 'predict' for every image."""
    model = RecordingInferenceModel()
    images = ["image0", "image1"]

    predictions = model.predict_batch(images)

    assert model.calls == [("image0", None), ("image1", None)]
    assert len(predictions) == len(images)


def test_predict_batch_empty_images():
    model = RecordingInferenceModel()

    assert model.predict_batch([]) == []
    assert model.calls == []


def test_predict_batch_raises_on_sample_dicts_length_mismatch():
    model = RecordingInferenceModel()

    with pytest.raises(ValueError, match="must match length of 'images'"):
        model.predict_batch(["image0", "image1"], sample_dicts=[{"index": 0}])

    assert model.calls == [], "'predict' should not be called when the input lengths do not match"


def test_predict_batch_override_is_used():
    """A model that implements batched inference can override 'predict_batch'."""

    class BatchedInferenceModel(RecordingInferenceModel):
        def predict_batch(self, images, sample_dicts=None) -> List[List[Primitive]]:
            return [[Classification(class_name="batched", ground_truth=False, confidence=1.0)] for _ in images]

    model = BatchedInferenceModel()

    predictions = model.predict_batch(["image0", "image1"])

    assert [[p.class_name for p in preds] for preds in predictions] == [["batched"], ["batched"]]
    assert model.calls == [], "The override should not fall back to 'predict'"
