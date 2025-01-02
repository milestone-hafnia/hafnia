from datasets import load_from_disk


def load_dataset(data_root: str):
    return load_from_disk(data_root)
