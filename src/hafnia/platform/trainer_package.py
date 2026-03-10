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
from hafnia_cli.config import Config


def _prepare_trainer_package(source_dir: Path) -> tuple[Path, List[Path], List[Dict], str]:
    """
    Prepare a trainer package by archiving the source directory and discovering cmd builder schemas.

    Returns:
        tuple: (zip_path, package_files, cmd_builder_schemas, trainer_name)
    """
    source_dir = Path(source_dir).resolve()
    path_trainer = get_trainer_package_path(trainer_name=source_dir.name)
    zip_path, package_files = archive_dir(source_dir, output_path=path_trainer)
    user_logger.info(f"Trainer package created and stored in '{path_trainer}'")
    cmd_builder_schemas = auto_discover_cmd_builder_schemas(package_files)
    return zip_path, package_files, cmd_builder_schemas, source_dir.name


@timed("Uploading trainer package.")
def create_trainer_package(
    source_dir: Path,
    name: Optional[str] = None,
    description: Optional[str] = None,
    cmd: Optional[str] = None,
    cfg: Optional[Config] = None,
) -> Dict:
    cfg = cfg or Config()
    endpoint = cfg.get_platform_endpoint("trainers")

    zip_path, package_files, cmd_builder_schemas, trainer_name = _prepare_trainer_package(source_dir)

    name = name or trainer_name
    cmd = cmd or "python scripts/train.py"
    description = description or f"Trainer package for '{name}'. Created with Hafnia SDK Cli."
    headers = {"Authorization": cfg.api_key, "accept": "application/json"}
    data = {
        "name": name,
        "description": description,
        "default_command": cmd,
        "file": (zip_path.name, Path(zip_path).read_bytes()),
    }
    if len(cmd_builder_schemas) > 0:
        data["command_builder_schemas"] = json.dumps(cmd_builder_schemas)
    user_logger.info(f"Uploading trainer package '{name}' to platform...")
    response = http.post(endpoint, headers=headers, data=data, multipart=True)
    user_logger.info(f"Trainer package uploaded successfully with id '{response['id']}'")
    return response


@timed("Updating trainer package.")
def update_trainer_package(
    id: str,
    source_dir: Optional[Path] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    cmd: Optional[str] = None,
    cfg: Optional[Config] = None,
) -> Dict:
    """Update an existing trainer package on the platform."""
    cfg = cfg or Config()
    endpoint = cfg.get_platform_endpoint("trainers")
    full_url = f"{endpoint}/{id}"
    headers = {"Authorization": cfg.api_key, "accept": "application/json"}

    data: Dict = {}
    if name is not None:
        data["name"] = name
    if description is not None:
        data["description"] = description
    if cmd is not None:
        data["default_command"] = cmd

    if source_dir is not None:
        zip_path, package_files, cmd_builder_schemas, trainer_name = _prepare_trainer_package(source_dir)
        if len(cmd_builder_schemas) > 0:
            data["command_builder_schemas"] = json.dumps(cmd_builder_schemas)
        data["file"] = (zip_path.name, Path(zip_path).read_bytes())

    if not data:
        raise ValueError("At least one field (source_dir, name, description, or cmd) must be provided for update.")

    user_logger.info(f"Updating trainer package '{id}' on platform...")
    # Always use multipart for consistency and compatibility
    response = http.patch(full_url, headers=headers, data=data, multipart=True)
    user_logger.info(f"Trainer package '{id}' updated successfully.")
    return response


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
def get_trainer_package_by_id(id: str, cfg: Optional[Config] = None) -> Dict:
    cfg = cfg or Config()
    endpoint = cfg.get_platform_endpoint("trainers")
    full_url = f"{endpoint}/{id}"
    headers = {"Authorization": cfg.api_key}
    response: Dict = http.fetch(full_url, headers=headers)  # type: ignore[assignment]

    return response


@timed("Get trainer packages")
def get_trainer_packages(
    cfg: Optional[Config] = None,
    limit: int = 1000,
    ordering: str = "-created_at",
    search: Optional[str] = None,
    visibility: Optional[str] = None,
) -> List[Dict]:
    cfg = cfg or Config()

    endpoint = cfg.get_platform_endpoint("trainers")
    params = {"page_size": limit, "ordering": ordering}
    if visibility:
        params["visibility"] = visibility
    if search:
        params["search"] = search
    headers = {"Authorization": cfg.api_key}
    response: Dict = http.fetch(endpoint, headers=headers, params=params)
    trainers = response.get("data", [])
    return trainers


def pretty_print_trainer_packages(trainers: List[Dict[str, str]]) -> None:
    mapping = {
        "ID": "id",
        "Name": "name",
        "Description": "description",
        "Created At": "created_at",
        "Visibility": "visibility",
    }
    description_max_length = 25
    display_trainers = []
    for trainer in trainers:
        trainer = trainer.copy()
        description = trainer.get("description", None)
        if description is None:
            description = ""
        if len(description) > description_max_length:
            trainer["description"] = description[:description_max_length] + "..."

        display_trainers.append(trainer)

    pretty_print_list_as_table(
        table_title="Available Trainer Packages (most recent first)",
        dict_items=display_trainers,
        column_name_to_key_mapping=mapping,
    )
