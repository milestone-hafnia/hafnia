from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe


def test_dataset_recipe():
    # Create a recipe using the FromName operation
    recipe = DatasetRecipe.from_name("mnist").shuffle().select_samples(n_samples=10)

    recipe.build()

    recipe.as_json_str()
