import json
from pathlib import Path
from typing import Dict, List, Optional

from hafnia import http
from hafnia.log import user_logger
from hafnia.utils import (
    archive_dir,
    get_trainer_package_path,
    pretty_print_list_as_table,
    timed,
)


@timed("Uploading trainer package.")
def create_trainer_package(
    source_dir: Path,
    endpoint: str,
    api_key: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    cmd: Optional[str] = None,
) -> str:
    # Ensure the path is absolute to handle '.' paths are given an appropriate name.
    source_dir = Path(source_dir).resolve()

    path_trainer = get_trainer_package_path(trainer_name=source_dir.name)
    name = name or path_trainer.stem
    zip_path, package_files = archive_dir(source_dir, output_path=path_trainer)
    user_logger.info(f"Trainer package created and stored in '{path_trainer}'")

    cmd_builder_schemas = auto_discover_cmd_builder_schemas(package_files)
    cmd = cmd or "python scripts/train.py"
    description = description or f"Trainer package for '{name}'. Created with Hafnia SDK Cli."
    headers = {"Authorization": api_key, "accept": "application/json"}
    data = {
        "name": name,
        "description": description,
        "default_command": cmd,
        "file": (zip_path.name, Path(zip_path).read_bytes()),
    }
    if len(cmd_builder_schemas) == 0:
        data["command_builder_schemas"] = json.dumps(cmd_builder_schemas)
    user_logger.info(f"Uploading trainer package '{name}' to platform...")
    response = http.post(endpoint, headers=headers, data=data, multipart=True)
    user_logger.info(f"Trainer package uploaded successfully with id '{response['id']}'")
    return response["id"]


def auto_discover_cmd_builder_schemas(package_files: List[Path]) -> List[Dict]:
    """
    Auto-discover command builder schema files in the trainer package files.
    Looks for files ending with '.schema.json' and loads their content as JSON.
    """
    cmd_builder_schema_files = [file for file in package_files if file.name.endswith(".schema.json")]
    cmd_builder_schemas = []
    for cmd_builder_schema_file in cmd_builder_schema_files:
        cmd_builder_schema = json.loads(cmd_builder_schema_file.read_text())
        cmd_entrypoint = cmd_builder_schema.get("cmd", None)
        user_logger.info(f"Found command builder schema file for entry point '{cmd_entrypoint}'")
        cmd_builder_schemas.append(cmd_builder_schema)
    return cmd_builder_schemas


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
