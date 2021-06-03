import numpy as np
import pytest

from featuretools.primitives import (
    AggregationPrimitive,
    Count,
    Hour,
    IsIn,
    TransformPrimitive
)
from featuretools.synthesis.get_valid_primitives import get_valid_primitives
from featuretools.utils.gen_utils import Library
from featuretools.variable_types import Categorical, Numeric


def test_get_valid_primitives_selected_primitives(es):
    agg_prims, trans_prims = get_valid_primitives(es, "log",
                                                  selected_primitives=[Hour, Count])
    assert set(agg_prims) == set([Count])
    assert set(trans_prims) == set([Hour])

    agg_prims, trans_prims = get_valid_primitives(es, "products",
                                                  selected_primitives=[Hour],
                                                  max_depth=1)
    assert set(agg_prims) == set()
    assert set(trans_prims) == set()


def test_get_valid_primitives_selected_primitives_strings(es):
    agg_prims, trans_prims = get_valid_primitives(es, "log",
                                                  selected_primitives=["hour", "count"])
    assert set(agg_prims) == set([Count])
    assert set(trans_prims) == set([Hour])

    agg_prims, trans_prims = get_valid_primitives(es, "products",
                                                  selected_primitives=["hour"],
                                                  max_depth=1)
    assert set(agg_prims) == set()
    assert set(trans_prims) == set()


def test_invalid_primitive_string(es):
    with pytest.raises(ValueError, match="'foobar' is not a recognized primitive name"):
        get_valid_primitives(es, target_entity='log', selected_primitives=['foobar'])


def test_get_valid_primitives_custom_primitives(pd_es):
    class ThreeMostCommonCat(AggregationPrimitive):
        name = "n_most_common_categorical"
        input_types = [Categorical]
        return_type = Categorical
        number_output_features = 3

        def get_function(self, agg_type='pandas'):
            def pd_top3(x):
                array = np.array(x.value_counts()[:3].index)
                if len(array) < 3:
                    filler = np.full(3 - len(array), np.nan)
                    array = np.append(array, filler)
                return array
            return pd_top3

    class AddThree(TransformPrimitive):
        name = 'add_three'
        input_types = [Numeric, Numeric, Numeric]
        return_type = Numeric
        commutative = True
        compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]

        def generate_name(self, base_feature_names):
            return "%s + %s + %s" % (base_feature_names[0], base_feature_names[1], base_feature_names[2])

    agg_prims, trans_prims = get_valid_primitives(pd_es, "log")
    assert ThreeMostCommonCat not in agg_prims
    assert AddThree not in trans_prims

    with pytest.raises(ValueError, match="'add_three' is not a recognized primitive name"):
        agg_prims, trans_prims = get_valid_primitives(pd_es, "log", 2, [ThreeMostCommonCat, "add_three"])


def test_get_valid_primitives_all_primitives(es):
    agg_prims, trans_prims = get_valid_primitives(es, "customers")
    assert Count in agg_prims
    assert Hour in trans_prims


def test_get_valid_primitives_single_table(transform_es):
    msg = "Only one entity in entityset, changing max_depth to 1 since deeper features cannot be created"
    with pytest.warns(UserWarning, match=msg):
        agg_prims, trans_prims = get_valid_primitives(transform_es, "first")

    assert set(agg_prims) == set()
    assert IsIn in trans_prims
