import pytest
from pympler.asizeof import asizeof

from ..testing_utils import make_ecommerce_entityset

from featuretools.primitives import Last, Mode, Sum
from featuretools.primitives.base import Feature, IdentityFeature
from featuretools.variable_types import Categorical, Datetime, Id, Numeric


@pytest.fixture(scope='module')
def es():
    return make_ecommerce_entityset()


@pytest.fixture(scope='module')
def es_numeric():
    return make_ecommerce_entityset(with_integer_time_index=True)


def test_copy_features_does_not_copy_entityset(es):
    agg = Sum(es['log']['value'], es['sessions'])
    agg_where = Sum(es['log']['value'], es['sessions'],
                    where=IdentityFeature(es['log']['value']) == 2)
    agg_use_previous = Sum(es['log']['value'], es['sessions'],
                           use_previous='4 days')
    agg_use_previous_where = Sum(es['log']['value'], es['sessions'],
                                 where=IdentityFeature(es['log']['value']) == 2,
                                 use_previous='4 days')
    features = [agg, agg_where, agg_use_previous, agg_use_previous_where]
    in_memory_size = asizeof(locals())
    copied = [f.copy() for f in features]
    new_in_memory_size = asizeof(locals())
    assert new_in_memory_size < 2 * in_memory_size

    for f, c in zip(features, copied):
        assert f.entityset
        assert c.entityset
        assert id(f.entityset) == id(c.entityset)
        if f.where:
            assert c.where
            assert id(f.where.entityset) == id(c.where.entityset)
        for bf, bf_c in zip(f.base_features, c.base_features):
            assert id(bf.entityset) == id(bf_c.entityset)
            if bf.where:
                assert bf_c.where
                assert id(bf.where.entityset) == id(bf_c.where.entityset)


def test_get_dependencies(es):
    f = Feature(es['log']['value'])
    agg1 = Sum(f, es['sessions'])
    agg2 = Sum(agg1, es['customers'])
    d1 = Feature(agg2, es['sessions'])
    shallow = d1.get_dependencies(deep=False, ignored=None)
    deep = d1.get_dependencies(deep=True, ignored=None)
    ignored = set([agg1.hash()])
    deep_ignored = d1.get_dependencies(deep=True, ignored=ignored)
    assert [s.hash() for s in shallow] == [agg2.hash()]
    assert [d.hash() for d in deep] == [agg2.hash(), agg1.hash(), f.hash()]
    assert [d.hash() for d in deep_ignored] == [agg2.hash()]


def test_get_depth(es):
    es = make_ecommerce_entityset()
    f = Feature(es['log']['value'])
    g = Feature(es['log']['value'])
    agg1 = Last(f, es['sessions'])
    agg2 = Last(agg1, es['customers'])
    d1 = Feature(agg2, es['sessions'])
    d2 = Feature(d1, es['log'])
    assert d2.get_depth() == 4
    # Make sure this works if we pass in two of the same
    # feature. This came up when user supplied duplicates
    # in the seed_features of DFS.
    assert d2.get_depth(stop_at=[f, g]) == 4
    assert d2.get_depth(stop_at=[f, g, agg1]) == 3
    assert d2.get_depth(stop_at=[f, g, agg1]) == 3
    assert d2.get_depth(stop_at=[f, g, agg2]) == 2
    assert d2.get_depth(stop_at=[f, g, d1]) == 1
    assert d2.get_depth(stop_at=[f, g, d2]) == 0


def test_squared(es):
    feature = Feature(es['log']['value'])
    squared = feature * feature
    assert len(squared.base_features) == 1
    assert squared.base_features[0].hash() == feature.hash()


def test_return_type_inference(es):
    mode = Mode(es["log"]["priority_level"], es["customers"])
    assert mode.variable_type == es["log"]["priority_level"].__class__


def test_return_type_inference_direct_feature(es):
    mode = Mode(es["log"]["priority_level"], es["customers"])
    mode_session = Feature(mode, es["sessions"])
    assert mode_session.variable_type == es["log"]["priority_level"].__class__


def test_return_type_inference_datetime_time_index(es):
    last = Last(es["log"]["datetime"], es["customers"])
    assert last.variable_type == Datetime


def test_return_type_inference_numeric_time_index(es_numeric):
    last = Last(es_numeric["log"]["datetime"], es_numeric["customers"])
    assert last.variable_type == Numeric


def test_return_type_inference_id(es):
    # direct features should keep Id variable type
    direct_id_feature = Feature(es["sessions"]["customer_id"], es["log"])
    assert direct_id_feature.variable_type == Id

    # aggregations of Id variable types should get converted
    mode = Mode(es["log"]["session_id"], es["customers"])
    assert mode.variable_type == Categorical

    # also test direct feature of aggregation
    mode_direct = Feature(mode, es["sessions"])
    assert mode_direct.variable_type == Categorical
