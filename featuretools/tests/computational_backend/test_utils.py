import numpy as np

from featuretools import dfs
from featuretools.computational_backends import replace_inf_values
from featuretools.primitives import DivideByFeature, DivideNumericScalar


def test_replace_inf_values(divide_by_zero_es):
    div_by_scalar = DivideNumericScalar(value=0)
    div_by_feature = DivideByFeature(value=1)
    div_by_feature_neg = DivideByFeature(value=-1)
    for primitive in [
        "divide_numeric",
        div_by_scalar,
        div_by_feature,
        div_by_feature_neg,
    ]:
        fm, _ = dfs(
            entityset=divide_by_zero_es,
            target_dataframe_name="zero",
            trans_primitives=[primitive],
            max_depth=1,
        )
        assert np.inf in fm.values or -np.inf in fm.values
        replaced_fm = replace_inf_values(fm)
        assert np.inf not in replaced_fm.values
        assert -np.inf not in replaced_fm.values

        custom_value_fm = replace_inf_values(fm, replacement_value="custom_val")
        assert np.inf not in custom_value_fm.values
        assert -np.inf not in replaced_fm.values
        assert "custom_val" in custom_value_fm.values


def test_replace_inf_values_specify_cols(divide_by_zero_es):
    div_by_scalar = DivideNumericScalar(value=0)
    fm, _ = dfs(
        entityset=divide_by_zero_es,
        target_dataframe_name="zero",
        trans_primitives=[div_by_scalar],
        max_depth=1,
    )

    assert np.inf in fm["col1 / 0"].values
    replaced_fm = replace_inf_values(fm, columns=["col1 / 0"])
    assert np.inf not in replaced_fm["col1 / 0"].values
    assert np.inf in replaced_fm["col2 / 0"].values
