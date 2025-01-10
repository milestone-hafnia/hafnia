from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional
from zipfile import ZipFile

from mdi_python_tools.config import CONFIG
from mdi_python_tools.http import fetch, post
from mdi_python_tools.log import logger
from mdi_python_tools.platform.codebuild import validate_recipe


def get_dataset_id(dataset_name: str) -> Optional[str]:
    headers = {"X-APIKEY": CONFIG.api_key}
    base_endpoint = CONFIG.get_platform_endpoint("datasets")
    full_url = f"{base_endpoint}?name__iexact={dataset_name}"
    try:
        dataset_info = fetch(full_url, headers=headers)
    except Exception as e:
        logger.error(f"Error fetching dataset info: {str(e)}")
        return None
    return dataset_info[0]["id"]


def create_recipe(source_dir: Path) -> Optional[str]:
    headers = {"X-APIKEY": CONFIG.api_key, "accept": "application/json"}
    endpoint = CONFIG.get_platform_endpoint("recipes")
    with TemporaryDirectory() as tdir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = Path(tdir) / f"recipe-{source_dir.name}-{ts}.zip"
        with ZipFile(zip_path, "w") as archive:
            for path in source_dir.rglob("*"):
                if any(
                    part.startswith(".") or part.startswith("__")
                    for part in path.relative_to(source_dir).parts
                ):
                    continue
                archive.write(path, path.relative_to(source_dir))
        validate_recipe(zip_path)
        with open(zip_path, "rb") as zip_file:
            fields = {
                "name": source_dir.name,
                "description": "Recipe created by MDI CLI",
                "organization": CONFIG.organization_id,
                "file": (zip_path.name, zip_file.read()),
            }
            try:
                response = post(endpoint, headers=headers, data=fields, multipart=True)
                return response["id"]
            except Exception as e:
                logger.error(f"Error creating recipe: {str(e)}")
                return None


def get_exp_environment_id(name: str) -> Optional[str]:
    headers = {"X-APIKEY": CONFIG.api_key}
    endoint = CONFIG.get_platform_endpoint("experiment_environments")
    try:
        env_info = fetch(endoint, headers=headers)
    except Exception as e:
        logger.error(f"Error fetching environment info: {str(e)}")
        return None
    return next((env["id"] for env in env_info if env["name"] == name), None)


def create_experiment(
    exp_name: str, dataset_id: str, recipe_id: str, exec_cmd: str, environment_id: str
) -> Optional[str]:
    headers = {"X-APIKEY": CONFIG.api_key}
    endpoint = CONFIG.get_platform_endpoint("experiments")
    try:
        response = post(
            endpoint,
            headers=headers,
            data={
                "organization": CONFIG.organization_id,
                "name": exp_name,
                "recipe": recipe_id,
                "dataset": dataset_id,
                "command": exec_cmd,
                "environment": environment_id,
            },
        )
    except Exception as e:
        logger.error(f"Error creating experiment: {str(e)}")
        return None
    return response["id"]


def get_exp_run_info(experiment_id: str) -> Optional[Dict]:
    headers = {"X-APIKEY": CONFIG.api_key, "accept": "application/json"}
    endpoint = CONFIG.get_platform_endpoint("experiment_runs")
    try:
        exp_run_info = fetch(f"{endpoint}?experiment={experiment_id}", headers=headers)
    except Exception as e:
        logger.error(f"Error fetching experiment run info: {str(e)}")
        return None
    return exp_run_info[0]


def create_experiment_run(
    exp_name: str, source_dir: str, exec_cmd: str, dataset_name: str, env_name: str
) -> Dict:
    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")

    dataset_id = get_dataset_id(dataset_name)
    recipe_id = create_recipe(source_dir)
    env_id = get_exp_environment_id(env_name)
    experiment_id = create_experiment(exp_name, dataset_id, recipe_id, exec_cmd, env_id)
    run_info = get_exp_run_info(experiment_id)
    return {
        "dataset_id": dataset_id,
        "recipe_id": recipe_id,
        "environment_id": env_id,
        "experiment_id": experiment_id,
        "event": run_info["event"],
    }
