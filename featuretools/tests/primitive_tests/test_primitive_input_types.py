import pytest
import featuretools as ft
from featuretools.primitives import MultiplyNumericScalar, Year

@pytest.mark.parametrize("primitive, expected_dtype", [
    (MultiplyNumericScalar(2), "float"),
    (Year(), "int")
])
def test_primitive_input_types(primitive, expected_dtype):
    es = ft.demo.load_retail(nrows=10)
    fm, features = ft.dfs(
        entityset=es,
        target_dataframe_name="orders",
        trans_primitives=[primitive],
        max_depth=1,
    )
    col = features[-1].get_name()
    dtype = fm[col].dtype
    assert expected_dtype in str(dtype), f"{primitive} returned dtype {dtype}"
