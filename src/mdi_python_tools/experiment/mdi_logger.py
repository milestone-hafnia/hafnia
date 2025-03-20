import json
import os
import platform
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Union

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, field_validator

from mdi_python_tools.log import logger


class EntityType(Enum):
    """Types of entities that can be logged."""

    SCALAR = "scalar"
    METRIC = "metric"


class EntityMode(Enum):
    """Modes for entity logging."""

    TRAIN = "train"
    EVAL = "eval"


class Entity(BaseModel):
    """
    Entity model for experiment logging.

    Attributes:
        step: Current step/iteration of the experiment
        ts: Timestamp when the entity was created
        name: Name of the entity
        ent_type: Type of entity (scalar, metric, image)
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
            logger.warning(f"Invalid value '{v}' provided, defaulting to -1.0: {e}")
            return -1.0

    @field_validator("ent_type", mode="before")
    def set_ent_type(cls, v: Union[EntityType, str]) -> str:
        """Convert EntityType enum to string value."""
        if isinstance(v, EntityType):
            return v.value
        return str(v)

    def __repr__(self):
        return f"{self.ts} {self.name}: {self.value} {self.ent_type}"


class MDILogger:
    EXPERIMENT_FILE = "experiment.parquet"

    def __init__(self, log_dir: Union[Path, str], update_interval: int = 5):
        self.log_dir = Path(log_dir) if isinstance(log_dir, str) else log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / self.EXPERIMENT_FILE
        self.remote_job = os.getenv("REMOTE_JOB", "False").lower() in ("true", "1", "yes")
        self.update_interval = max(1, update_interval)
        self.entities: List[Entity] = []
        self.schema = self.create_schema()
        self.log_environment()

    def create_schema(self) -> pa.Schema:
        """Create the PyArrow schema for the Parquet file."""
        return pa.schema(
            [
                pa.field("step", pa.int64()),
                pa.field("ts", pa.string()),
                pa.field("name", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("ent_type", pa.string()),
                pa.field("image", pa.string()),
            ]
        )

    def log_metric(self, name: str, value: float, is_training: bool, step: int) -> None:
        self.log_scalar(name, value, is_training, step, EntityType.METRIC)

    def log_scalar(
        self,
        name: str,
        value: float,
        is_training: bool,
        step: int,
        ent_type: EntityType = EntityType.SCALAR,
    ) -> None:
        mode = EntityMode.TRAIN if is_training else EntityMode.EVAL
        entity = Entity(
            step=step,
            ts=datetime.now().isoformat(),
            name=name,
            value=value,
            mode=mode,
            ent_type=ent_type,
        )
        self.entities.append(entity)
        print(entity)
        if len(self.entities) >= self.update_interval:
            self.flush()

    def log_configuration(self, configurations: Dict):
        self.log_hparams(configurations, "configuration.json")

    def log_hparams(self, params: Dict, fname: str = "hparams.json"):
        file_path = self.log_dir / fname
        try:
            with open(file_path, "w") as f:
                json.dump(params, f, indent=2)
            logger.info(f"Saved parameters to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save parameters to {file_path}: {e}")

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

    def flush(self) -> None:
        """
        Force writing all accumulated logs to disk.

        This method should be called at the end of an experiment or
        when you want to ensure all logs are written.
        """
        if not self.entities:
            return

        try:
            log_batch = pa.Table.from_pylist(
                [e.model_dump() for e in self.entities], schema=self.schema
            )

            if not self.log_file.exists():
                pq.write_table(log_batch, self.log_file)
            else:
                prev = pa.parquet.read_table(self.log_file)
                next_table = pa.concat_tables([prev, log_batch])
                pq.write_table(next_table, self.log_file)

            logger.info(f"Flushed {len(self.entities)} logs to {self.log_file}")
            self.entities = []
        except Exception as e:
            logger.error(f"Failed to flush logs: {e}")
