# Abstract model interface
from pydantic import BaseModel


class ModelInterface:
    def __init__(self, **kwargs):
        """Initialize the model with given parameters."""
        # Validate kwargs against model configuration schema
        ModelConfigType = self.model_config()
        model_config = ModelConfigType.model_validate(kwargs)
        self.model = self.model_from_config(model_config)

    @staticmethod
    def model_config() -> BaseModel:
        """Return the model configuration schema."""
        pass

    @staticmethod
    def model_from_config(config: BaseModel):
        """Instantiate the model from a configuration object."""
        pass

    @staticmethod
    def train_config() -> BaseModel:
        """Return the training configuration schema."""
        pass

    @staticmethod
    def train_from_config(config: BaseModel):
        """Instantiate the training process from a configuration object."""
        pass

    @staticmethod
    def inference_config() -> BaseModel:
        """Return the evaluation configuration schema."""
        pass

    @staticmethod
    def inference_from_config(config: BaseModel):
        """Instantiate the evaluation process from a configuration object."""
        pass

    @staticmethod
    def from_path(path_config: str, path_model: str):
        """Instantiate the model from a file path."""
        pass

    @staticmethod
    def load_model_weights(path: str):
        """Load the model file from the given path."""
        pass

    def to_device(self) -> None:
        """Convert the model to a configuration object."""
        pass

    def to_config(self) -> BaseModel:
        pass

    def to_path(self) -> str:
        pass

    def train(self, data):
        """Train the model with the provided data."""
        raise NotImplementedError("Train method not implemented.")

    def predict_one(self, input_data):
        """Make predictions based on the input data."""
        raise NotImplementedError("Predict method not implemented.")

    def predict_batch(self, input_batch):
        """Make predictions on a batch of input data."""
        raise NotImplementedError("Predict batch method not implemented.")

    def evaluate(self, test_data):
        """Evaluate the model's performance on test data."""
        raise NotImplementedError("Evaluate method not implemented.")
