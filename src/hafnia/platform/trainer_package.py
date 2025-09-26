from pathlib import Path
from typing import Dict, List, Optional

from hafnia import http
from hafnia.log import user_logger
from hafnia.utils import archive_dir, get_trainer_package_path, pretty_print_list_as_table, timed


@timed("Uploading trainer package.")
def create_trainer_package(source_dir: Path, endpoint: str, api_key: str) -> str:
    source_dir = source_dir.resolve()  # Ensure the path is absolute to handle '.' paths are given an appropriate name.
    path_trainer = get_trainer_package_path(trainer_name=source_dir.name)
    zip_path = archive_dir(source_dir, output_path=path_trainer)
    user_logger.info(f"Trainer package created and stored in '{path_trainer}'")

    headers = {"Authorization": api_key, "accept": "application/json"}
    data = {
        "name": path_trainer.name,
        "description": "Trainer package created by Hafnia CLI",
        "file": (zip_path.name, Path(zip_path).read_bytes()),
    }
    response = http.post(endpoint, headers=headers, data=data, multipart=True)
    return response["id"]


@timed("Get trainer package.")
def get_trainer_package_by_id(id: str, endpoint: str, api_key: str) -> Dict:
    full_url = f"{endpoint}/{id}"
    headers = {"Authorization": api_key}
    response: Dict = http.fetch(full_url, headers=headers)  # type: ignore[assignment]
    return response


@timed("Get trainer packages")
def get_trainer_packages(endpoint: str, api_key: str) -> List[Dict]:
    headers = {"Authorization": api_key}
    trainers: List[Dict] = http.fetch(endpoint, headers=headers)  # type: ignore[assignment]
    return trainers


def pretty_print_trainer_packages(trainers: List[Dict[str, str]], limit: Optional[int]) -> None:
    # Sort trainer packages to have the most recent first
    trainers = sorted(trainers, key=lambda x: x["created_at"], reverse=True)
    if limit is not None:
        trainers = trainers[:limit]

    mapping = {
        "ID": "id",
        "Name": "name",
        "Description": "description",
        "Created At": "created_at",
    }
    pretty_print_list_as_table(
        table_title="Available Trainer Packages (most recent first)",
        dict_items=trainers,
        column_name_to_key_mapping=mapping,
    )
