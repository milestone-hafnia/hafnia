import json
import os
import platform
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, field_validator

from hafnia.data.factory import load_dataset
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.log import sys_logger, user_logger
from hafnia.utils import is_remote_job, now_as_str


class EntityType(Enum):
    """Types of entities that can be logged."""

    SCALAR = "scalar"
    METRIC = "metric"


class Entity(BaseModel):
    """
    Entity model for experiment logging.

    Attributes:
        step: Current step/iteration of the experiment
        ts: Timestamp when the entity was created
        name: Name of the entity
        ent_type: Type of entity (scalar, metric)
        mode: Mode of logging (train or eval)
        value: Numerical value of the entity
    """

    step: int
    ts: str
    name: str
    ent_type: str
    value: float = -1

    @field_validator("value", mode="before")
    def set_value(cls, v: Union[float, str, int]) -> float:
        """Convert input to float or default to -1.0."""
        try:
            return float(v)
        except (ValueError, TypeError) as e:
            user_logger.warning(f"Invalid value '{v}' provided, defaulting to -1.0: {e}")
            return -1.0

    @field_validator("ent_type", mode="before")
    def set_ent_type(cls, v: Union[EntityType, str]) -> str:
        """Convert EntityType enum to string value."""
        if isinstance(v, EntityType):
            return v.value
        return str(v)

    @staticmethod
    def create_schema() -> pa.Schema:
        """Create the PyArrow schema for the Parquet file."""
        return pa.schema(
            [
                pa.field("step", pa.int64()),
                pa.field("ts", pa.string()),
                pa.field("name", pa.string()),
                pa.field("ent_type", pa.string()),
                pa.field("value", pa.float64()),
            ]
        )


class HafniaLogger:
    EXPERIMENT_FILE = "experiment.parquet"

    def __init__(self, log_dir: Union[Path, str] = "./.data"):
        self._local_experiment_path = Path(log_dir) / "experiments" / now_as_str()
        create_paths = [
            self._local_experiment_path,
            self.path_model_checkpoints(),
            self._path_artifacts(),
            self.path_model(),
        ]
        for path in create_paths:
            path.mkdir(parents=True, exist_ok=True)

        self.dataset_name: Optional[str] = None
        self.log_file = self._path_artifacts() / self.EXPERIMENT_FILE
        self.schema = Entity.create_schema()
        self.log_environment()

    def load_dataset(self, dataset_name: str) -> HafniaDataset:
        """
        Load a dataset from the specified path.
        """
        self.dataset_name = dataset_name
        return load_dataset(dataset_name)

    def path_local_experiment(self) -> Path:
        """Get the path for local experiment."""
        if is_remote_job():
            raise RuntimeError("Cannot access local experiment path in remote job.")
        return self._local_experiment_path

    def path_model_checkpoints(self) -> Path:
        """Get the path for model checkpoints."""
        if "MDI_CHECKPOINT_DIR" in os.environ:
            return Path(os.environ["MDI_CHECKPOINT_DIR"])

        if is_remote_job():
            return Path("/opt/ml/checkpoints")
        return self.path_local_experiment() / "checkpoints"

    def _path_artifacts(self) -> Path:
        """Get the path for artifacts."""
        if "MDI_ARTIFACT_DIR" in os.environ:
            return Path(os.environ["MDI_ARTIFACT_DIR"])

        if is_remote_job():
            return Path("/opt/ml/output/data")

        return self.path_local_experiment() / "data"

    def path_model(self) -> Path:
        """Get the path for the model."""
        if "MDI_MODEL_DIR" in os.environ:
            return Path(os.environ["MDI_MODEL_DIR"])

        if is_remote_job():
            return Path("/opt/ml/model")

        return self.path_local_experiment() / "model"

    def log_metric(self, name: str, value: float, step: int) -> None:
        self.log_scalar(name, value, step, EntityType.METRIC)

    def log_scalar(
        self,
        name: str,
        value: float,
        step: int,
        ent_type: EntityType = EntityType.SCALAR,
    ) -> None:
        entity = Entity(
            step=step,
            ts=datetime.now().isoformat(),
            name=name,
            value=value,
            ent_type=ent_type.value,
        )
        self.write_entity(entity)

    def log_configuration(self, configurations: Dict):
        self.log_hparams(configurations, "configuration.json")

    def log_hparams(self, params: Dict, fname: str = "hparams.json"):
        file_path = self._path_artifacts() / fname
        try:
            if file_path.exists():  # New params are appended to existing params
                existing_params = json.loads(file_path.read_text())
            else:
                existing_params = {}
            existing_params.update(params)
            file_path.write_text(json.dumps(existing_params, indent=2))
            user_logger.info(f"Saved parameters to {file_path}")
        except Exception as e:
            user_logger.error(f"Failed to save parameters to {file_path}: {e}")

    def log_environment(self):
        environment_info = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "os": platform.system(),
            "os_release": platform.release(),
            "cpu_count": os.cpu_count(),
            "cuda_version": os.getenv("CUDA_VERSION", "N/A"),
            "cudnn_version": os.getenv("CUDNN_VERSION", "N/A"),
        }
        self.log_hparams(environment_info, "environment.json")

    def write_entity(self, entity: Entity) -> None:
        """
        Force writing all accumulated logs to disk.

        This method should be called at the end of an experiment or
        when you want to ensure all logs are written.
        """
        print(entity)  # Keep this line! Parsed and used for real-time logging in another process
        entities = [entity]
        try:
            log_batch = pa.Table.from_pylist([e.model_dump() for e in entities], schema=self.schema)

            if not self.log_file.exists():
                pq.write_table(log_batch, self.log_file)
            else:
                prev = pa.parquet.read_table(self.log_file)
                next_table = pa.concat_tables([prev, log_batch])
                pq.write_table(next_table, self.log_file)
        except Exception as e:
            sys_logger.error(f"Failed to flush logs: {e}")
