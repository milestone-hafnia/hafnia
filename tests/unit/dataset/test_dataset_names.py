from hafnia.dataset.dataset_names import SampleField
from hafnia.dataset.hafnia_dataset import Sample


def test_sample_fields():
    column_variable_names = list(SampleField.__annotations__)
    sample_fields = Sample.model_fields.keys()
    for column_variable_name in column_variable_names:
        column_name = getattr(SampleField, column_variable_name)
        assert column_name in sample_fields, (
            f"Column name '{column_name}' defined in 'FieldName.{column_variable_name}' "
            f"not found in '{Sample.__name__}' fields. Available fields are: {list(sample_fields)}"
        )
