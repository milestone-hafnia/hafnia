from pathlib import Path
from typing import Dict, List, Optional

from hafnia import http
from hafnia.log import user_logger
from hafnia.utils import archive_dir, get_recipe_path, pretty_print_list_as_table, timed


@timed("Uploading recipe.")
def create_training_recipe(source_dir: Path, endpoint: str, api_key: str) -> str:
    source_dir = source_dir.resolve()  # Ensure the path is absolute to handle '.' paths are given an appropriate name.
    path_recipe = get_recipe_path(recipe_name=source_dir.name)
    zip_path = archive_dir(source_dir, output_path=path_recipe)
    user_logger.info(f"Recipe created and stored in '{path_recipe}'")

    headers = {"Authorization": api_key, "accept": "application/json"}
    data = {
        "name": path_recipe.name,
        "description": "Recipe created by Hafnia CLI",
        "file": (zip_path.name, Path(zip_path).read_bytes()),
    }
    response = http.post(endpoint, headers=headers, data=data, multipart=True)
    return response["id"]


@timed("Get training recipe.")
def get_training_recipe_by_id(id: str, endpoint: str, api_key: str) -> Dict:
    full_url = f"{endpoint}/{id}"
    headers = {"Authorization": api_key}
    response: Dict = http.fetch(full_url, headers=headers)  # type: ignore[assignment]
    return response


@timed("Get training recipes")
def get_training_recipes(endpoint: str, api_key: str) -> List[Dict]:
    headers = {"Authorization": api_key}
    recipes: List[Dict] = http.fetch(endpoint, headers=headers)  # type: ignore[assignment]
    return recipes


def pretty_print_training_recipes(recipes: List[Dict[str, str]], limit: Optional[int]) -> None:
    # Sort recipes to have the most recent first
    recipes = sorted(recipes, key=lambda x: x["created_at"], reverse=True)
    if limit is not None:
        recipes = recipes[:limit]

    mapping = {
        "ID": "id",
        "Name": "name",
        "Description": "description",
        "Created At": "created_at",
    }
    pretty_print_list_as_table(
        table_title="Available Training Recipes (most recent first)",
        dict_items=recipes,
        column_name_to_key_mapping=mapping,
    )
