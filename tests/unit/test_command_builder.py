import copy
import inspect
from pathlib import Path
from typing import Annotated, Callable, Literal, Optional

import cyclopts
import pytest
import typer
from pydantic import BaseModel
from pydantic import Field as PydanticField

from hafnia.experiment import command_builder
from hafnia.experiment.command_builder import CommandBuilderSchema, write_cmd_builder_schema_to_file_from_cli_function


def test_docstring_description_example():
    """Test to extract parameter descriptions from docstrings."""

    def docstring_description_example(param0: str, param1: int):
        """
        Example function to demonstrate docstring description extraction.
        Below descriptions for each parameter are automatically extracted and used in the schema.

        Parameters
        ----------
        param0 : str
            Parameter docstring 0.
        param1 : int
            Parameter docstring 1.
        """
        pass

    schema = command_builder.schema_from_cli_function(docstring_description_example)
    for idx, (name, prop) in enumerate(schema["properties"].items()):
        assert "description" in prop, f"Parameter '{name}' is missing description in schema."
        assert prop["description"] == f"Parameter docstring {idx}.", (
            f"Parameter '{name}' has incorrect description in schema."
        )
        assert "type" in prop, f"Parameter '{name}' is missing type in schema."
        assert "title" in prop, f"Parameter '{name}' is missing title in schema."


def test_typer_argument_and_option_example():
    """Test to extract parameter descriptions from typer.Argument and typer.Option."""

    def typer_argument_and_option_example(
        param1: Annotated[str, typer.Option(help="Param without default.")],
        param2: Annotated[int, typer.Option(help="Param with default")] = 10,
    ):
        pass

    has_default_value = ["param2"]
    schema = command_builder.schema_from_cli_function(typer_argument_and_option_example)
    for name, prop in schema["properties"].items():
        assert "description" in prop, f"Parameter '{name}' is missing description in schema."
        assert "type" in prop, f"Parameter '{name}' is missing type in schema."
        assert "title" in prop, f"Parameter '{name}' is missing title in schema."
        if name in has_default_value:
            assert "default" in prop, f"Parameter '{name}' is missing default in schema."


def test_typer_argument_not_supported():
    """Test to extract parameter descriptions from typer.Argument and typer.Option."""

    def typer_argument_and_option_example(
        param1: Annotated[str, typer.Argument(help="This is an example parameter using typer.Argument.")],
        param2: Annotated[int, typer.Option(help="Another parameter with typer.Option description.")] = 10,
    ):
        pass

    with pytest.raises(TypeError, match="typer.Argument is not supported"):
        command_builder.schema_from_cli_function(typer_argument_and_option_example)


def test_cyclopts_parameter_example():
    """Test to extract parameter descriptions from cyclopts.Parameter."""

    def cyclopts_parameter_example(
        param1: Annotated[str, cyclopts.Parameter(help="This is an example parameter using cyclopts.Parameter.")],
        param2: Annotated[int, cyclopts.Parameter(help="Another parameter with cyclopts.Parameter description.")] = 10,
    ):
        pass

    has_default_value = ["param2"]
    schema = command_builder.schema_from_cli_function(cyclopts_parameter_example)
    for name, prop in schema["properties"].items():
        assert "description" in prop, f"Parameter '{name}' is missing description in schema."
        assert "type" in prop, f"Parameter '{name}' is missing type in schema."
        assert "title" in prop, f"Parameter '{name}' is missing title in schema."
        if name in has_default_value:
            assert "default" in prop, f"Parameter '{name}' is missing default in schema."


def test_pydantic_field_example():
    """Test to extract parameter descriptions from Pydantic Field."""

    def pydantic_field_example(
        param1: Annotated[str, PydanticField(description="This is an example parameter using Pydantic Field.")],
        param2: Annotated[int, PydanticField(description="Another parameter with Pydantic Field description.")] = 10,
    ):
        pass

    has_default_value = ["param2"]
    schema = command_builder.schema_from_cli_function(pydantic_field_example)
    for name, prop in schema["properties"].items():
        assert "description" in prop, f"Parameter '{name}' is missing description in schema."
        assert "type" in prop, f"Parameter '{name}' is missing type in schema."
        assert "title" in prop, f"Parameter '{name}' is missing title in schema."
        if name in has_default_value:
            assert "default" in prop, f"Parameter '{name}' is missing default in schema."


def test_default_mismatch_failure():
    """Test to extract parameter descriptions from docstrings."""

    def default_value_mismatch(
        param0: Annotated[str, PydanticField(default="asdf", description="Desc 0.")],
        param1: Annotated[str, PydanticField(default="One", description="Desc 1.")] = "Two",
    ):
        pass

    with pytest.raises(ValueError, match="The default value for 'param1' have been defined in two ways"):
        command_builder.schema_from_cli_function(default_value_mismatch)


def test_bool_not_supported_failure():
    """Test to check that bool parameters raise an error."""

    def bool_parameter_example(
        param0: bool,
    ):
        pass

    with pytest.raises(TypeError, match="Boolean types are not supported "):
        command_builder.schema_from_cli_function(bool_parameter_example)


def example_advanced_cli_example() -> Callable:
    """Advanced test function to generate and print Pydantic schema from function signature."""

    class NestedNestedModel(BaseModel):
        optimizer_name: Optional[str] = PydanticField(..., description="Name of the optimizer")
        learning_rate: float = PydanticField(0.01, description="Learning rate for optimizer")

    class ExampleNestedModel(BaseModel):
        model_name: Optional[str] = PydanticField(..., description="Name of the model")
        epochs: int = PydanticField(3, description="Number of epochs to train")
        optimizer: Optional[NestedNestedModel] = PydanticField(None, description="Nested nested model configuration")

    EXAMPLE_NESTED_MODEL_DEFAULT = ExampleNestedModel(
        model_name="default_model",
        epochs=5,
        optimizer=NestedNestedModel(optimizer_name="adam", learning_rate=0.01),
    )

    def cli_function_example(
        test: str,  # Required parameter without annotation
        string_description_from_doc: str,  # Description is automatically taken from docstring
        required_string: Annotated[
            str, "This description is used", PydanticField(description="This description is ignored", min_length=2)
        ],
        nested_model: Annotated[ExampleNestedModel, "Nested model configuration"],
        string_description_from_doc_with_default: str = "blah",  # Description is automatically taken from docstring
        string_with_default_underscore: str = "blah_underscore",  # We should not convert underscore values to kebab-case
        string_min_length: Annotated[
            str, PydanticField(description="SomeDescription", min_length=2)
        ] = "default_string",
        int_or_string: Annotated[int | str, "An int or string value"] = 1,
        project_name1: Annotated[str, PydanticField(default="asdf", description="Project name1")] = "asdf",
        project_name2: Annotated[str, PydanticField(description="Project name2")] = "asdf",
        project_name3: Annotated[str, cyclopts.Parameter(help="Project name2")] = "asdf",
        name_typer: Annotated[str, typer.Option(help="This is a test typer option")] = "default_name",
        epochs: Annotated[int, cyclopts.Parameter(help="Number of epochs to train")] = 3,
        learning_rate: Annotated[float, "Learning rate for optimizer"] = 0.001,
        resize: Annotated[Optional[int], "Resize image to specified size. If None, will drop unless needed"] = None,
        batch_size: Annotated[int, PydanticField(description="Batch size for training", gt=0, lt=1000)] = 128,
        num_workers: Annotated[int, "Number of workers for DataLoader"] = 8,
        log_interval: Annotated[int, "Interval for logging"] = 5,
        max_steps_per_epoch: Annotated[int, "Max steps per epoch"] = 20,
        models: Annotated[Literal["resnet18", "vgg16"], "Model name"] = "resnet18",
        nested_model_with_default: Annotated[
            ExampleNestedModel, "Nested model configuration"
        ] = EXAMPLE_NESTED_MODEL_DEFAULT,
    ):
        """
        Example function with the same signature as train.py main()

        Parameters
        ----------
        string_description_from_doc : str
            Description is automatically taken from docstring.
        string_description_from_doc_with_default : str
            Description is automatically taken from docstring.
        """
        pass

    return cli_function_example


def test_command_builder_schema(tmp_path: Path):
    """Test function to generate and print Pydantic schema from function signature."""

    ## Setup example function with various parameter types and annotations
    cli_function_example = example_advanced_cli_example()
    path_schema_json = tmp_path / "command_schema.json"
    write_cmd_builder_schema_to_file_from_cli_function(
        cli_function=cli_function_example,
        path_schema=path_schema_json,
    )

    assert path_schema_json.exists(), "Schema JSON file was not created."

    schema = CommandBuilderSchema.from_json_file(path_schema_json)
    schema.model_dump(mode="json")


def test_command_builder_from_function():
    """Test function to generate and print Pydantic schema from function signature."""

    ## Setup example function with various parameter types and annotations
    cli_function_example = example_advanced_cli_example()

    # Test: Generate schema
    command_builder_schema = CommandBuilderSchema.from_function(cli_function_example)
    schema_dict = command_builder_schema.json_schema

    # Assert:
    no_description_params = ["test", "string_with_default_underscore"]
    no_default_params = ["test", "string_description_from_doc", "required_string", "nested_model"]
    required_params = no_default_params

    for name, prop in schema_dict["properties"].items():
        if "$ref" in prop:  # Skip $ref properties
            continue
        if name not in no_description_params:
            assert "description" in prop, f"Parameter '{name}' is missing description in schema."

        if "anyOf" not in prop:  # Skip complex types
            assert "type" in prop, f"Parameter '{name}' is missing type in schema."

        assert "title" in prop, f"Parameter '{name}' is missing title in schema."

        if name in no_default_params:
            assert "default" not in prop, f"Parameter '{name}' should not have a default in schema."
        else:
            assert "default" in prop, f"Parameter '{name}' is missing default in schema."

    props = schema_dict["properties"]
    assert props["string_with_default_underscore"]["default"] == "blah_underscore", (
        "Default value for 'string_with_default_underscore' is incorrect."
    )
    assert required_params == schema_dict.get("required", []), "Required parameters do not match schema."

    # Check nested model properties
    assert props["nested_model"]["$ref"] == "#/$defs/ExampleNestedModel", "Nested model $ref is incorrect."
    nested_model_def = schema_dict["$defs"]["ExampleNestedModel"]
    assert set(nested_model_def["properties"]) == {"model_name", "epochs", "optimizer"}, (
        "Nested model properties are incorrect."
    )

    # Check nested model with default
    nested_model_with_default = props["nested_model_with_default"]
    assert nested_model_with_default["$ref"] == "#/$defs/ExampleNestedModel", (
        "Nested model with default $ref is incorrect."
    )

    expected_defaults = {
        "model_name": "default_model",
        "epochs": 5,
        "optimizer": {"optimizer_name": "adam", "learning_rate": 0.01},
    }
    assert nested_model_with_default["default"] == expected_defaults, "Nested model with default value is incorrect."

    # Check union types
    union_field_key = [
        "properties.resize",
        "properties.int_or_string",
        "$defs.ExampleNestedModel.properties.model_name",
    ]
    for field_key in union_field_key:
        union_prop = copy.deepcopy(schema_dict)
        nested_field_keys = field_key.split(".")
        for key in nested_field_keys:
            if key not in union_prop:
                raise KeyError(f"Key '{key}' not found in schema while accessing field '{field_key}'.")
            union_prop = union_prop[key]

        assert "anyOf" in union_prop, "Union type should have 'anyOf' in schema."
        assert len(union_prop["anyOf"]) == 2, "Union type 'float_or_int' should have two types in 'anyOf'."
        assert all("type" in item for item in union_prop["anyOf"]), "All union types should anyOf should have a type."
        assert all("title" in item for item in union_prop["anyOf"]), "All union type should anyOf should have a title."


def test_command_args_from_form_data():
    cli_function_example = example_advanced_cli_example()
    update_params = {
        "test": "my_test_value",
        "required_string": "my_required_string",
        "nested_model": {"model_name": "custom_model"},
        "string_description_from_doc": "custom description",
    }
    params = inspect.signature(cli_function_example).parameters
    n_root_params = len(params)
    form_dataset = command_builder.simulate_form_data(cli_function_example, update_params)
    assert len(form_dataset) == n_root_params, "Form dataset does not have the correct number of parameters."

    cmd_builder_schema = CommandBuilderSchema.from_function(cli_function_example)
    commands_args = cmd_builder_schema.command_args_from_form_data(form_dataset)
    command_str = " ".join(commands_args)
    assert command_str.count(" --") > n_root_params


def test_command_args_from_form_data_simple():
    class NestedModel(BaseModel):
        name: str

    def some_function(param_value1: str, param_value2: int = 10, nested: NestedModel = NestedModel(name="default")):
        pass

    update_params = {
        "param_value1": "custom_value",
    }

    params = inspect.signature(some_function).parameters
    n_root_params = len(params)
    form_dataset = command_builder.simulate_form_data(some_function, update_params)
    assert len(form_dataset) == n_root_params, "Form dataset does not have the correct number of parameters."

    # Use case 1: Using default settings
    cmd_builder1 = CommandBuilderSchema.from_function(some_function)
    commands_args = cmd_builder1.command_args_from_form_data(form_dataset)
    cmd_string = " ".join(commands_args)
    assert cmd_string.count(" --") == n_root_params - cmd_builder1.n_positional_args
    assert "nested.name" in cmd_string, "Nested parameter not correctly represented. Expected '.' separator."
    assert "param-value1" in cmd_string, "Parameter value was not converted to kebab-case."

    # Use case 2: Custom settings
    cmd_builder2 = CommandBuilderSchema.from_function(
        some_function,
        parameter_prefix="++",
        nested_parameter_separator="__",
        n_positional_args=1,
        kebab_case=False,
        assignment_separator="equals",
    )

    commands_args = cmd_builder2.command_args_from_form_data(form_dataset)
    commands_args[0] = "'custom_value'"  # Adjust for positional argument
    cmd_string = " ".join(commands_args)
    assert cmd_string.count(" ++") == n_root_params - cmd_builder2.n_positional_args
    assert "nested__name" in cmd_string, "Nested parameter not correctly represented. Expected '__' separator."
    assert "param_value2" in cmd_string, "Parameter value was incorrectly converted to kebab-case."
    assert "++nested__name='default'" in cmd_string, "Assignment separator '=' not correctly used."
