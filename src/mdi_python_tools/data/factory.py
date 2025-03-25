from pathlib import Path

from datasets import load_from_disk

from mdi_python_tools.log import logger


def load_dataset(data_root: str):
    if not Path(data_root).exists():
        logger.error(f"Can not load dataset, directory does not exist -- {data_root}")
        exit(1)
    return load_from_disk(data_root)
