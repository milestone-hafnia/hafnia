from pathlib import Path

from hafnia.utils import view_recipe_content


def test_view_recipe_content():
    tree_str = view_recipe_content(Path("./recipe.zip"))

    assert "train.py" in tree_str
    assert "Dockerfile" in tree_str

    print(tree_str)
