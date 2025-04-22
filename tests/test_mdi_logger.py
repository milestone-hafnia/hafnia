import os
from pathlib import Path

import pyarrow as pa
import pytest

from hafnia.experiment.hafnia_logger import EntityType, HafniaLogger


@pytest.fixture(scope="function")
def logger(tmpdir: Path) -> HafniaLogger:
    """Create a logger instance for testing."""
    if "HAFNIA_LOCAL_SCRIPT" not in os.environ:
        os.environ["HAFNIA_LOCAL_SCRIPT"] = "true"
    return HafniaLogger(Path(tmpdir))


def test_basic_scalar_logging(logger: HafniaLogger) -> None:
    """Test that basic scalar logging works."""

    assert not logger.log_file.exists()

    logger.log_scalar("test_scalar", 42.0, 1)

    assert logger.log_file.exists()

    logger.log_scalar("test_scalar", 43.0, 2)

    table = pa.parquet.read_table(logger.log_file)
    df = table.to_pandas()
    assert len(df) == 2
    assert df.iloc[0]["value"] == 42.0
    assert df.iloc[1]["value"] == 43.0


def test_metric_logging(logger: HafniaLogger) -> None:
    """Test that metric logging works."""
    logger.log_metric("accuracy", 0.95, 100)
    logger.log_metric("loss", 0.05, 100)
    assert logger.log_file.exists()

    # Verify the data
    table = pa.parquet.read_table(logger.log_file)
    df = table.to_pandas()

    metrics_df = df[df["ent_type"] == EntityType.METRIC.value]
    assert len(metrics_df) == 2
    assert "accuracy" in metrics_df["name"].values
    assert "loss" in metrics_df["name"].values


def test_config_logging(logger: HafniaLogger):
    """Test configuration logging."""
    config = {"learning_rate": 0.001, "batch_size": 32, "model_type": "resnet50"}

    logger.log_configuration(config)
    config_file = logger._path_artifacts() / "configuration.json"

    assert config_file.exists()

    import json

    with open(config_file, "r") as f:
        loaded_config = json.load(f)

    assert loaded_config["learning_rate"] == 0.001
    assert loaded_config["batch_size"] == 32
    assert loaded_config["model_type"] == "resnet50"
