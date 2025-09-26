from typing import Dict, List

from hafnia import http
from hafnia.utils import pretty_print_list_as_table, timed


@timed("Creating experiment.")
def create_experiment(
    experiment_name: str,
    dataset_recipe_id: str,
    trainer_id: str,
    exec_cmd: str,
    environment_id: str,
    endpoint: str,
    api_key: str,
) -> Dict:
    headers = {"Authorization": api_key}
    response = http.post(
        endpoint,
        headers=headers,
        data={
            "name": experiment_name,
            "trainer": trainer_id,
            "dataset_recipe": dataset_recipe_id,
            "command": exec_cmd,
            "environment": environment_id,
        },
    )
    return response


@timed("Fetching environment info.")
def get_environments(endpoint: str, api_key: str) -> List[Dict]:
    headers = {"Authorization": api_key}
    envs: List[Dict] = http.fetch(endpoint, headers=headers)  # type: ignore[assignment]
    return envs


def pretty_print_training_environments(envs: List[Dict]) -> None:
    ENV_FIELDS = {
        "Name": "name",
        "Instance": "instance",
        "GPU": "gpu",
        "GPU Count": "gpu_count",
        "GPU RAM": "vram",
        "CPU": "cpu",
        "CPU Count": "cpu_count",
        "RAM": "ram",
    }
    pretty_print_list_as_table(
        table_title="Available Training Environments",
        dict_items=envs,
        column_name_to_key_mapping=ENV_FIELDS,
    )


def get_exp_environment_id(name: str, endpoint: str, api_key: str) -> str:
    envs = get_environments(endpoint=endpoint, api_key=api_key)

    for env in envs:
        if env["name"] == name:
            return env["id"]

    pretty_print_training_environments(envs)

    available_envs = [env["name"] for env in envs]

    raise ValueError(f"Environment '{name}' not found. Available environments are: {available_envs}")
