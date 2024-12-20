import json
from typing import Union, Dict
from pathlib import Path
from mdi_runc.log import logger
import pyarrow.parquet as pa
from argparse import ArgumentParser


def mdi_to_tboard(log_dir: Union[str, Path]) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter

    except ImportError:
        raise ImportError("Please install tensorboard to use this function: `pip install torch`")
    log_dir = Path(log_dir) if not isinstance(log_dir, Path) else log_dir

    exp_file = log_dir / "experiment.parquet"
    if not exp_file.exists():
        raise FileNotFoundError(f"Experiment file not found at {exp_file}")

    sw = SummaryWriter(log_dir.as_posix())
    table = pa.read_table(exp_file)
    for row in table.to_pylist():
        sw.add_scalar(row["name"], row["value"], row["step"])

    support_files = (
        ("environment.json", "Environment"),
        ("configuration.json", "Configuration"),
    )
    for fname, tag in support_files:
        path = log_dir / fname
        if not path.exists():
            continue
        with open(path, "r") as f:
            text = "\n".join([f"{key}: {value}" for key, value in json.load(f).items()])
            sw.add_text(tag, text)

    logger.info(f"Converted MDI logs to Tensorboard format and saved to {sw.log_dir}")
    sw.close()


def mdi_to_sagemaker(log_dir: Union[str, Path]) -> None:
    """
    Convert MDI logs to a SageMaker-compatible format, including generating a training_report.json.
    """
    log_dir = Path(log_dir) if not isinstance(log_dir, Path) else log_dir

    exp_file = log_dir / "experiment.parquet"
    if not exp_file.exists():
        raise FileNotFoundError(f"Experiment file not found at {exp_file}")

    # Read experiment file
    table = pa.read_table(exp_file)
    rows = table.to_pylist()

    # Generate training report
    training_report = {
        "Metrics": [
            {
                "MetricName": row["name"],
                "Value": row["value"],
                "Timestamp": row.get("timestamp", "unknown"),  # Optional timestamp
                "Step": row["step"],
            }
            for row in rows
        ],
        "Environment": {},
        "Configuration": {},
    }

    # Add environment.json and configuration.json if available
    for fname, key in (
        ("environment.json", "Environment"),
        ("configuration.json", "Configuration"),
    ):
        file_path = log_dir / fname
        if file_path.exists():
            with open(file_path, "r") as f:
                training_report[key] = json.load(f)

    # Save training report as training_report.json
    training_report_file = log_dir / "training_report.json"
    with open(training_report_file, "w") as f:
        json.dump(training_report, f, indent=4)

    logger.info(
        f"Converted MDI logs to SageMaker format and saved training_report.json to {training_report_file}"
    )


def mdi_to_wb(log_dir: Union[str, Path], wb_config: Dict):
    raise NotImplementedError("Not implemented yet")


def main():
    parser = ArgumentParser(description="Convert MDI logs to other formats")
    parser.add_argument("log_dir", type=str, help="Path to the log directory from MDI experiment.")
    parser.add_argument("--format", type=str, default="tboard", help="Output format to convert to.")
    args = parser.parse_args()
    if args.format == "tboard":
        return mdi_to_tboard(args.log_dir)
    if args.format == "sagemaker":
        return mdi_to_sagemaker(args.log_dir)
    raise ValueError(f"Unsupported format: {args.format}")
