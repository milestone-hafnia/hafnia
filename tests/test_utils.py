from pathlib import Path

from hafnia.utils import view_recipe_content


def test_view_recipe_content():
    # Recreates 'recipe.zip' from the `recipe-classification` example.
    # Below command assumes that `recipe-classification` is available in same directory as the `hafnia` repo.
    #       hafnia recipe create ../recipe-classification/ && mv recipe.zip tests/data/
    tree_str = view_recipe_content(Path("tests/data/recipe.zip"))

    assert "train.py" in tree_str
    assert "Dockerfile" in tree_str

    print(tree_str)
