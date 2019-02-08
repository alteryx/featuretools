import pytest

from ..testing_utils import make_ecommerce_entityset

import featuretools as ft

from featuretools.primitives import (  # CumCount,; CumMax,; CumMean,; CumMin,; CumSum,
    AddNumeric,
    AddNumericScalar,
    Count,
    DivideByFeature,
    DivideNumeric,
    DivideNumericScalar,
    Equal,
    EqualScalar,
    GreaterThan,
    GreaterThanEqualTo,
    GreaterThanEqualToScalar,
    GreaterThanScalar,
    LessThan,
    LessThanEqualTo,
    LessThanEqualToScalar,
    LessThanScalar,
    ModuloByFeature,
    ModuloNumeric,
    ModuloNumericScalar,
    MultiplyNumeric,
    MultiplyNumericScalar,
    Negate,
    NotEqual,
    NotEqualScalar,
    ScalarSubtractNumericFeature,
    SubtractNumeric,
    SubtractNumericScalar,
    Sum
)


@pytest.fixture
def es():
    return make_ecommerce_entityset()


def test_overrides(es):
    value = ft.Feature(es['log']['value'])
    value2 = ft.Feature(es['log']['value_2'])

    feats = [AddNumeric, SubtractNumeric, MultiplyNumeric, DivideNumeric,
             ModuloNumeric, GreaterThan, LessThan, Equal, NotEqual,
             GreaterThanEqualTo, LessThanEqualTo]
    assert ft.Feature(value, primitive=Negate).hash() == (-value).hash()

    compares = [(value, value), (value, value2)]
    overrides = [
        value + value,
        value - value,
        value * value,
        value / value,
        value % value,
        value > value,
        value < value,
        value == value,
        value != value,
        value >= value,
        value <= value,

        value + value2,
        value - value2,
        value * value2,
        value / value2,
        value % value2,
        value > value2,
        value < value2,
        value == value2,
        value != value2,
        value >= value2,
        value <= value2,
    ]

    for left, right in compares:
        for feat in feats:
            f = ft.Feature([left, right], primitive=feat)
            o = overrides.pop(0)
            assert o.hash() == f.hash()


def test_override_boolean(es):
    count = ft.Feature(es['log']['id'], parent_entity=es['sessions'], primitive=Count)
    count_lo = ft.Feature(count, primitive=GreaterThanScalar(1))
    count_hi = ft.Feature(count, primitive=LessThanScalar(10))

    to_test = [[True, True, True],
               [True, True, False],
               [False, False, True]]

    features = []
    features.append(count_lo.OR(count_hi))
    features.append(count_lo.AND(count_hi))
    features.append(~(count_lo.AND(count_hi)))

    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=[0, 1, 2])
    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert v == test


def test_scalar_overrides(es):
    value = ft.Feature(es['log']['value'])

    feats = [
        AddNumericScalar, SubtractNumericScalar, MultiplyNumericScalar,
        DivideNumericScalar, ModuloNumericScalar, GreaterThanScalar,
        LessThanScalar, EqualScalar, NotEqualScalar, GreaterThanEqualToScalar,
        LessThanEqualToScalar
    ]

    overrides = [
        value + 2,
        value - 2,
        value * 2,
        value / 2,
        value % 2,
        value > 2,
        value < 2,
        value == 2,
        value != 2,
        value >= 2,
        value <= 2,
    ]

    for feat in feats:
        f = ft.Feature(value, primitive=feat(2))
        o = overrides.pop(0)
        assert o.hash() == f.hash()

    value2 = ft.Feature(es['log']['value_2'])

    reverse_feats = [
        AddNumericScalar, ScalarSubtractNumericFeature, MultiplyNumericScalar,
        DivideByFeature, ModuloByFeature, GreaterThanScalar, LessThanScalar,
        EqualScalar, NotEqualScalar, GreaterThanEqualToScalar,
        LessThanEqualToScalar
    ]
    reverse_overrides = [
        2 + value2,
        2 - value2,
        2 * value2,
        2 / value2,
        2 % value2,
        2 < value2,
        2 > value2,
        2 == value2,
        2 != value2,
        2 <= value2,
        2 >= value2
    ]
    for feat in reverse_feats:
        f = ft.Feature(value2, primitive=feat(2))
        o = reverse_overrides.pop(0)
        assert o.hash() == f.hash()


def test_override_cmp_from_variable(es):
    count_lo = ft.Feature(es['log']['value']) > 1

    to_test = [False, True, True]

    features = [count_lo]

    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=[0, 1, 2])
    v = df[count_lo.get_name()].values.tolist()
    for i, test in enumerate(to_test):
        assert v[i] == test


def test_override_cmp(es):
    count = ft.Feature(es['log']['id'], parent_entity=es['sessions'], primitive=Count)
    _sum = ft.Feature(es['log']['value'], parent_entity=es['sessions'], primitive=Sum)
    gt_lo = count > 1
    gt_other = count > _sum
    ge_lo = count >= 1
    ge_other = count >= _sum
    lt_hi = count < 10
    lt_other = count < _sum
    le_hi = count <= 10
    le_other = count <= _sum
    ne_lo = count != 1
    ne_other = count != _sum

    to_test = [[True, True, False],
               [False, False, True],
               [True, True, True],
               [False, False, True],
               [True, True, True],
               [True, True, False],
               [True, True, True],
               [True, True, False]]
    features = [gt_lo, gt_other, ge_lo, ge_other, lt_hi,
                lt_other, le_hi, le_other, ne_lo, ne_other]

    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=[0, 1, 2])
    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert v == test
