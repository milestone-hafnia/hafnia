import json
from pathlib import Path
from typing import Dict, List, Optional

import click
from flatten_dict import flatten
from rich import print as rprint

from cli.config import Config
from hafnia.platform.experiment import delete_dataset_recipe_by_id, delete_dataset_recipe_by_name, get_dataset_recipes
from hafnia.utils import pretty_print_list_as_table


@click.group(name="dataset-recipe")
def dataset_recipe() -> None:
    """Dataset recipe commands"""
    pass


@dataset_recipe.command(name="create")
@click.option(
    "-p",
    "--path",
    type=click.Path(writable=True),
    required=True,
    help="Output dataset recipe path.",
)
@click.option(
    "-n",
    "--name",
    type=str,
    default=None,
    show_default=True,
    help="Name of the dataset recipe.",
)
@click.pass_obj
def dataset_recipe_create(cfg: Config, path: Path, name: Optional[str]) -> None:
    """Create Hafnia dataset recipe from local path"""
    from hafnia.platform.experiment import get_or_create_dataset_recipe_from_path

    endpoint = cfg.get_platform_endpoint("dataset_recipes")
    recipe = get_or_create_dataset_recipe_from_path(path, endpoint=endpoint, api_key=cfg.api_key, name=name)

    if recipe is None:
        raise click.ClickException("Failed to create dataset recipe.")

    rprint(recipe)


@dataset_recipe.command(name="ls")
@click.pass_obj
@click.option("-l", "--limit", type=int, default=None, help="Limit number of listed dataset recipes.")
def list_dataset_recipes(cfg: Config, limit: Optional[int]) -> None:
    """List available dataset recipes"""
    endpoint = cfg.get_platform_endpoint("dataset_recipes")
    recipes = get_dataset_recipes(endpoint=endpoint, api_key=cfg.api_key)
    # Sort recipes to have the most recent first
    recipes = sorted(recipes, key=lambda x: x["created_at"], reverse=True)
    if limit is not None:
        recipes = recipes[:limit]
    pretty_print_dataset_recipes(recipes)


@dataset_recipe.command(name="rm")
@click.option("-i", "--id", type=str, help="Dataset recipe ID to delete.")
@click.option("-n", "--name", type=str, help="Dataset recipe name to delete.")
@click.pass_obj
def delete_dataset_recipe(cfg: Config, id: Optional[str], name: Optional[str]) -> Dict:
    """Delete a dataset recipe by ID or name"""
    endpoint = cfg.get_platform_endpoint("dataset_recipes")

    if id is not None:
        return delete_dataset_recipe_by_id(id=id, endpoint=endpoint, api_key=cfg.api_key)
    if name is not None:
        dataset_recipe = delete_dataset_recipe_by_name(name=name, endpoint=endpoint, api_key=cfg.api_key)
        if dataset_recipe is None:
            raise click.ClickException(f"Dataset recipe with name '{name}' was not found.")

        return dataset_recipe

    raise click.MissingParameter(
        "No dataset recipe identifier have been given. Provide either --id or --name. "
        "Get available recipes with 'hafnia dataset-recipe ls'."
    )


def pretty_print_dataset_recipes(recipes: List[Dict]) -> None:
    recipes = [flatten(recipe, reducer="dot", max_flatten_depth=2) for recipe in recipes]  # noqa: F821
    for recipe in recipes:
        recipe["recipe_json"] = json.dumps(recipe["template.body"])[:20]

    RECIPE_FIELDS = {
        "ID": "id",
        "Name": "name",
        "Recipe": "recipe_json",
        "Created": "created_at",
        "IsDataset": "template.is_direct_dataset_reference",
    }
    pretty_print_list_as_table(
        table_title="Available Dataset Recipes",
        dict_items=recipes,
        column_name_to_key_mapping=RECIPE_FIELDS,
    )
