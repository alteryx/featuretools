from featuretools import Feature, calculate_feature_matrix
from featuretools.primitives import (
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
    Sum,
)
from featuretools.tests.testing_utils import to_pandas


def test_overrides(es):
    value = Feature(es["log"].ww["value"])
    value2 = Feature(es["log"].ww["value_2"])

    feats = [
        AddNumeric,
        SubtractNumeric,
        MultiplyNumeric,
        DivideNumeric,
        ModuloNumeric,
        GreaterThan,
        LessThan,
        Equal,
        NotEqual,
        GreaterThanEqualTo,
        LessThanEqualTo,
    ]
    assert Feature(value, primitive=Negate).unique_name() == (-value).unique_name()

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
            f = Feature([left, right], primitive=feat)
            o = overrides.pop(0)
            assert o.unique_name() == f.unique_name()


def test_override_boolean(es):
    count = Feature(
        es["log"].ww["id"],
        parent_dataframe_name="sessions",
        primitive=Count,
    )
    count_lo = Feature(count, primitive=GreaterThanScalar(1))
    count_hi = Feature(count, primitive=LessThanScalar(10))

    to_test = [[True, True, True], [True, True, False], [False, False, True]]

    features = []
    features.append(count_lo.OR(count_hi))
    features.append(count_lo.AND(count_hi))
    features.append(~(count_lo.AND(count_hi)))

    df = calculate_feature_matrix(
        entityset=es,
        features=features,
        instance_ids=[0, 1, 2],
    )
    df = to_pandas(df, index="id", sort_index=True)
    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].tolist()
        assert v == test


def test_scalar_overrides(es):
    value = Feature(es["log"].ww["value"])

    feats = [
        AddNumericScalar,
        SubtractNumericScalar,
        MultiplyNumericScalar,
        DivideNumericScalar,
        ModuloNumericScalar,
        GreaterThanScalar,
        LessThanScalar,
        EqualScalar,
        NotEqualScalar,
        GreaterThanEqualToScalar,
        LessThanEqualToScalar,
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
        f = Feature(value, primitive=feat(2))
        o = overrides.pop(0)
        assert o.unique_name() == f.unique_name()

    value2 = Feature(es["log"].ww["value_2"])

    reverse_feats = [
        AddNumericScalar,
        ScalarSubtractNumericFeature,
        MultiplyNumericScalar,
        DivideByFeature,
        ModuloByFeature,
        GreaterThanScalar,
        LessThanScalar,
        EqualScalar,
        NotEqualScalar,
        GreaterThanEqualToScalar,
        LessThanEqualToScalar,
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
        2 >= value2,
    ]
    for feat in reverse_feats:
        f = Feature(value2, primitive=feat(2))
        o = reverse_overrides.pop(0)
        assert o.unique_name() == f.unique_name()


def test_override_cmp_from_column(es):
    count_lo = Feature(es["log"].ww["value"]) > 1

    to_test = [False, True, True]

    features = [count_lo]

    df = to_pandas(
        calculate_feature_matrix(
            entityset=es,
            features=features,
            instance_ids=[0, 1, 2],
        ),
        index="id",
        sort_index=True,
    )
    v = df[count_lo.get_name()].tolist()
    for i, test in enumerate(to_test):
        assert v[i] == test


def test_override_cmp(es):
    count = Feature(
        es["log"].ww["id"],
        parent_dataframe_name="sessions",
        primitive=Count,
    )
    _sum = Feature(
        es["log"].ww["value"],
        parent_dataframe_name="sessions",
        primitive=Sum,
    )
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

    to_test = [
        [True, True, False],
        [False, False, True],
        [True, True, True],
        [False, False, True],
        [True, True, True],
        [True, True, False],
        [True, True, True],
        [True, True, False],
    ]
    features = [
        gt_lo,
        gt_other,
        ge_lo,
        ge_other,
        lt_hi,
        lt_other,
        le_hi,
        le_other,
        ne_lo,
        ne_other,
    ]

    df = calculate_feature_matrix(
        entityset=es,
        features=features,
        instance_ids=[0, 1, 2],
    )
    df = to_pandas(df, index="id", sort_index=True)
    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].tolist()
        assert v == test
