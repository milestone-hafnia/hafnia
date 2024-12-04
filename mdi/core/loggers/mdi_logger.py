import numpy as np
import base64
import json
from typing import Dict, List, Optional, Union
import os
import pyarrow as pa
from datetime import datetime
from mdi_runc.log import logger
from PIL import Image
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, field_validator


class EntityType(Enum):
    SCALAR = "scalar"
    METRIC = "metric"
    IMAGE = "image"


class EntityMode(Enum):
    TRAIN = "train"
    EVAL = "eval"


class Entity(BaseModel):
    step: int
    ts: str
    name: str
    ent_type: str
    mode: str
    value: float = -1
    image: Optional[str] = None

    @field_validator("value", mode="before")
    def set_value(cls, v: float):
        try:
            return float(v)
        except ValueError:
            return -1.0

    @field_validator("ent_type", mode="before")
    def set_ent_type(cls, v: EntityType):
        return v.value

    @field_validator("image", mode="before")
    def set_image(cls, v: str):
        return None if v is None else str(v)

    def __repr__(self):
        return f"{self.mode} {self.ts} {self.name}: {self.value} {self.ent_type}"


class MDILogger:
    def __init__(self, log_dir: Union[Path, str], update_interval: int = 5):
        self.log_dir = Path(log_dir) if isinstance(log_dir, str) else log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "experiment.parquet"
        self.remote_job = os.getenv("REMOTE_JOB", False)
        self.update_interval = max(1, update_interval)
        self.entities: List[Entity] = []
        self.create_schema()
        self.log_environment()

    def create_schema(self) -> None:
        self.schema = pa.schema(
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
        self.post_log()

    def log_image(self, name: str, value: np.ndarray, step: int):
        if self.remote_job:
            logger.warning("Images cannot be logged in the remote execution.")
            return
        try:
            image_bytes = Image.fromarray(value).tobytes()
        except ValueError:
            return
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        self.entities.append(
            Entity(
                step=step,
                ts=datetime.now().isoformat(),
                name=name,
                image=image_b64,
                ent_type=EntityType.IMAGE,
            )
        )
        self.post_log()

    def log_configuration(self, configurations: Dict):
        self.log_hparams(configurations, "configuration.json")

    def log_hparams(self, params: Dict, fname: str = "hparams.json"):
        with open(self.log_dir / f"{fname}", "w") as f:
            json.dump(params, f)

    def log_environment(self):
        import sys
        import platform

        environment_info = {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "os": platform.system(),
            "os_release": platform.release(),
            "cpu_count": os.cpu_count(),
            "cuda_version": os.getenv("CUDA_VERSION", "N/A"),
            "cudnn_version": os.getenv("CUDNN_VERSION", "N/A"),
        }
        self.log_hparams(environment_info, "environment.json")

    def post_log(self) -> None:
        if len(self.entities) != self.update_interval:
            return
        prev = (
            None if not self.log_file.exists() else pa.parquet.read_table(self.log_file)
        )
        log_batch = pa.Table.from_pylist(
            [e.dict() for e in self.entities], schema=self.schema
        )
        next_table = log_batch if prev is None else pa.concat_tables([prev, log_batch])
        pa.parquet.write_table(next_table, self.log_file)
        self.entities = []
