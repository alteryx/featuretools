import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

import featuretools as ft
from featuretools.computational_backends.feature_set import FeatureSet
from featuretools.computational_backends.feature_set_calculator import (
    FeatureSetCalculator
)
from featuretools.primitives import (
    Absolute,
    AddNumeric,
    AddNumericScalar,
    Age,
    Count,
    Day,
    Diff,
    DivideByFeature,
    DivideNumeric,
    DivideNumericScalar,
    Equal,
    EqualScalar,
    GreaterThanEqualToScalar,
    GreaterThanScalar,
    Haversine,
    Hour,
    IsIn,
    IsNull,
    Latitude,
    LessThanEqualToScalar,
    LessThanScalar,
    Longitude,
    Minute,
    Mode,
    Month,
    MultiplyNumeric,
    MultiplyNumericScalar,
    Not,
    NotEqual,
    NotEqualScalar,
    NumCharacters,
    NumWords,
    Percentile,
    ScalarSubtractNumericFeature,
    Second,
    SubtractNumeric,
    SubtractNumericScalar,
    Sum,
    TransformPrimitive,
    Year,
    get_transform_primitives
)
from featuretools.primitives.base import make_trans_primitive
from featuretools.primitives.utils import (
    PrimitivesDeserializer,
    serialize_primitive
)
from featuretools.synthesis.deep_feature_synthesis import match
from featuretools.tests.testing_utils import feature_with_name
from featuretools.variable_types import Boolean, Datetime, Numeric, Variable


def test_init_and_name(es):
    log = es['log']
    rating = ft.Feature(es["products"]["rating"], es["log"])
    log_features = [ft.Feature(v) for v in log.variables] +\
        [ft.Feature(rating, primitive=GreaterThanScalar(2.5))]
    # Add Timedelta feature
    # features.append(pd.Timestamp.now() - ft.Feature(log['datetime']))
    customers_features = [ft.Feature(v) for v in es["customers"].variables]
    trans_primitives = get_transform_primitives().values()
    # If Dask EntitySet use only Dask compatible primitives
    if isinstance(es['log'].df, dd.DataFrame):
        trans_primitives = [prim for prim in trans_primitives if prim.dask_compatible]
    for transform_prim in trans_primitives:
        # skip automated testing if a few special cases
        features_to_use = log_features
        if transform_prim in [NotEqual, Equal]:
            continue
        if transform_prim in [Age]:
            features_to_use = customers_features

        # use the input_types matching function from DFS
        input_types = transform_prim.input_types
        if type(input_types[0]) == list:
            matching_inputs = match(input_types[0], features_to_use)
        else:
            matching_inputs = match(input_types, features_to_use)
        if len(matching_inputs) == 0:
            raise Exception(
                "Transform Primitive %s not tested" % transform_prim.name)
        for prim in matching_inputs:
            instance = ft.Feature(prim, primitive=transform_prim)

            # try to get name and calculate
            instance.get_name()
            ft.calculate_feature_matrix([instance], entityset=es)


def test_relationship_path(es):
    f = ft.TransformFeature(es['log']['datetime'], Hour)

    assert len(f.relationship_path) == 0


def test_serialization(es):
    value = ft.IdentityFeature(es['log']['value'])
    primitive = ft.primitives.MultiplyNumericScalar(value=2)
    value_x2 = ft.TransformFeature(value, primitive)

    dictionary = {
        'name': None,
        'base_features': [value.unique_name()],
        'primitive': serialize_primitive(primitive),
    }

    assert dictionary == value_x2.get_arguments()
    assert value_x2 == \
        ft.TransformFeature.from_dictionary(dictionary, es,
                                            {value.unique_name(): value},
                                            PrimitivesDeserializer())


def test_make_trans_feat(es):
    f = ft.Feature(es['log']['datetime'], primitive=Hour)

    feature_set = FeatureSet([f])
    calculator = FeatureSetCalculator(es, feature_set=feature_set)
    df = calculator.run(np.array([0]))
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    v = df[f.get_name()][0]
    assert v == 10


@pytest.fixture
def simple_es():
    df = pd.DataFrame({
        'id': range(4),
        'value': pd.Categorical(['a', 'c', 'b', 'd']),
        'value2': pd.Categorical(['a', 'b', 'a', 'd']),
        'object': ['time1', 'time2', 'time3', 'time4'],
        'datetime': pd.Series([pd.Timestamp('2001-01-01'),
                               pd.Timestamp('2001-01-02'),
                               pd.Timestamp('2001-01-03'),
                               pd.Timestamp('2001-01-04')])
    })

    es = ft.EntitySet('equal_test')
    es.entity_from_dataframe('values', df, index='id')

    return es


def test_equal_categorical(simple_es):
    f1 = ft.Feature([simple_es['values']['value'], simple_es['values']['value2']],
                    primitive=Equal)

    df = ft.calculate_feature_matrix(entityset=simple_es, features=[f1])

    assert set(simple_es['values'].df['value'].cat.categories) != \
        set(simple_es['values'].df['value2'].cat.categories)
    assert df['value = value2'].to_list() == [True, False, False, True]


def test_equal_different_dtypes(simple_es):
    f1 = ft.Feature([simple_es['values']['object'], simple_es['values']['datetime']],
                    primitive=Equal)
    f2 = ft.Feature([simple_es['values']['datetime'], simple_es['values']['object']],
                    primitive=Equal)

    # verify that equals works for different dtypes regardless of order
    df = ft.calculate_feature_matrix(entityset=simple_es, features=[f1, f2])

    assert df['object = datetime'].to_list() == [False, False, False, False]
    assert df['datetime = object'].to_list() == [False, False, False, False]


def test_not_equal_categorical(simple_es):
    f1 = ft.Feature([simple_es['values']['value'], simple_es['values']['value2']],
                    primitive=NotEqual)

    df = ft.calculate_feature_matrix(entityset=simple_es, features=[f1])

    assert set(simple_es['values'].df['value'].cat.categories) != \
        set(simple_es['values'].df['value2'].cat.categories)
    assert df['value != value2'].to_list() == [False, True, True, False]


def test_not_equal_different_dtypes(simple_es):
    f1 = ft.Feature([simple_es['values']['object'], simple_es['values']['datetime']],
                    primitive=NotEqual)
    f2 = ft.Feature([simple_es['values']['datetime'], simple_es['values']['object']],
                    primitive=NotEqual)

    # verify that equals works for different dtypes regardless of order
    df = ft.calculate_feature_matrix(entityset=simple_es, features=[f1, f2])

    assert df['object != datetime'].to_list() == [True, True, True, True]
    assert df['datetime != object'].to_list() == [True, True, True, True]


def test_diff(pd_es):
    value = ft.Feature(pd_es['log']['value'])
    customer_id_feat = ft.Feature(pd_es['sessions']['customer_id'], entity=pd_es['log'])
    diff1 = ft.Feature(value, groupby=pd_es['log']['session_id'], primitive=Diff)
    diff2 = ft.Feature(value, groupby=customer_id_feat, primitive=Diff)

    feature_set = FeatureSet([diff1, diff2])
    calculator = FeatureSetCalculator(pd_es, feature_set=feature_set)
    df = calculator.run(np.array(range(15)))

    val1 = df[diff1.get_name()].values.tolist()
    val2 = df[diff2.get_name()].values.tolist()
    correct_vals1 = [
        np.nan, 5, 5, 5, 5, np.nan, 1, 1, 1, np.nan, np.nan, 5, np.nan, 7, 7
    ]
    correct_vals2 = [np.nan, 5, 5, 5, 5, -20, 1, 1, 1, -3, np.nan, 5, -5, 7, 7]
    for i, v in enumerate(val1):
        v1 = val1[i]
        if np.isnan(v1):
            assert (np.isnan(correct_vals1[i]))
        else:
            assert v1 == correct_vals1[i]
        v2 = val2[i]
        if np.isnan(v2):
            assert (np.isnan(correct_vals2[i]))
        else:
            assert v2 == correct_vals2[i]


def test_diff_single_value(pd_es):
    diff = ft.Feature(pd_es['stores']['num_square_feet'], groupby=pd_es['stores'][u'région_id'], primitive=Diff)
    feature_set = FeatureSet([diff])
    calculator = FeatureSetCalculator(pd_es, feature_set=feature_set)
    df = calculator.run(np.array([4]))
    assert df[diff.get_name()][4] == 6000.0


def test_diff_reordered(pd_es):
    sum_feat = ft.Feature(pd_es['log']['value'], parent_entity=pd_es["sessions"], primitive=Sum)
    diff = ft.Feature(sum_feat, primitive=Diff)
    feature_set = FeatureSet([diff])
    calculator = FeatureSetCalculator(pd_es, feature_set=feature_set)
    df = calculator.run(np.array([4, 2]))
    assert df[diff.get_name()][4] == 16
    assert df[diff.get_name()][2] == -6


def test_diff_single_value_is_nan(pd_es):
    diff = ft.Feature(pd_es['stores']['num_square_feet'], groupby=pd_es['stores'][u'région_id'], primitive=Diff)
    feature_set = FeatureSet([diff])
    calculator = FeatureSetCalculator(pd_es, feature_set=feature_set)
    df = calculator.run(np.array([5]))
    assert df.shape[0] == 1
    assert df[diff.get_name()].dropna().shape[0] == 0


def test_compare_of_identity(es):
    to_test = [(EqualScalar, [False, False, True, False]),
               (NotEqualScalar, [True, True, False, True]),
               (LessThanScalar, [True, True, False, False]),
               (LessThanEqualToScalar, [True, True, True, False]),
               (GreaterThanScalar, [False, False, False, True]),
               (GreaterThanEqualToScalar, [False, False, True, True])]

    features = []
    for test in to_test:
        features.append(ft.Feature(es['log']['value'], primitive=test[0](10)))

    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=[0, 1, 2, 3])
    if isinstance(df, dd.DataFrame):
        df = df.compute()

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert v == test[1]


def test_compare_of_direct(es):
    log_rating = ft.Feature(es['products']['rating'], entity=es['log'])
    to_test = [(EqualScalar, [False, False, False, False]),
               (NotEqualScalar, [True, True, True, True]),
               (LessThanScalar, [False, False, False, True]),
               (LessThanEqualToScalar, [False, False, False, True]),
               (GreaterThanScalar, [True, True, True, False]),
               (GreaterThanEqualToScalar, [True, True, True, False])]

    features = []
    for test in to_test:
        features.append(ft.Feature(log_rating, primitive=test[0](4.5)))

    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=[0, 1, 2, 3])
    if isinstance(df, dd.DataFrame):
        df = df.compute()

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert v == test[1]


def test_compare_of_transform(es):
    day = ft.Feature(es['log']['datetime'], primitive=Day)
    to_test = [(EqualScalar, [False, True]),
               (NotEqualScalar, [True, False]),
               (LessThanScalar, [True, False]),
               (LessThanEqualToScalar, [True, True]),
               (GreaterThanScalar, [False, False]),
               (GreaterThanEqualToScalar, [False, True])]

    features = []
    for test in to_test:
        features.append(ft.Feature(day, primitive=test[0](10)))

    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=[0, 14])
    if isinstance(df, dd.DataFrame):
        df = df.compute()

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert v == test[1]


def test_compare_of_agg(es):
    count_logs = ft.Feature(es['log']['id'], parent_entity=es['sessions'], primitive=Count)

    to_test = [(EqualScalar, [False, False, False, True]),
               (NotEqualScalar, [True, True, True, False]),
               (LessThanScalar, [False, False, True, False]),
               (LessThanEqualToScalar, [False, False, True, True]),
               (GreaterThanScalar, [True, True, False, False]),
               (GreaterThanEqualToScalar, [True, True, False, True])]

    features = []
    for test in to_test:
        features.append(ft.Feature(count_logs, primitive=test[0](2)))

    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=[0, 1, 2, 3])
    if isinstance(df, dd.DataFrame):
        df = df.compute()

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert v == test[1]


def test_compare_all_nans(es):
    if any(isinstance(entity.df, dd.DataFrame) for entity in es.entities):
        nan_feat = ft.Feature(es['log']['value'], parent_entity=es['sessions'], primitive=ft.primitives.Min)
        compare = nan_feat == 0.0
    else:
        nan_feat = ft.Feature(es['log']['product_id'], parent_entity=es['sessions'], primitive=Mode)
        compare = nan_feat == 'brown bag'

    # before all data
    time_last = pd.Timestamp('1/1/1993')

    df = ft.calculate_feature_matrix(entityset=es, features=[nan_feat, compare], instance_ids=[0, 1, 2], cutoff_time=time_last)
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    assert df[nan_feat.get_name()].dropna().shape[0] == 0
    assert not df[compare.get_name()].any()


def test_arithmetic_of_val(es):
    to_test = [(AddNumericScalar, [2.0, 7.0, 12.0, 17.0]),
               (SubtractNumericScalar, [-2.0, 3.0, 8.0, 13.0]),
               (ScalarSubtractNumericFeature, [2.0, -3.0, -8.0, -13.0]),
               (MultiplyNumericScalar, [0, 10, 20, 30]),
               (DivideNumericScalar, [0, 2.5, 5, 7.5]),
               (DivideByFeature, [np.inf, 0.4, 0.2, 2 / 15.0])]

    features = []
    for test in to_test:
        features.append(ft.Feature(es['log']['value'], primitive=test[0](2)))

    features.append(ft.Feature(es['log']['value']) / 0)

    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=[0, 1, 2, 3])
    if isinstance(df, dd.DataFrame):
        df = df.compute()

    for f, test in zip(features, to_test):
        v = df[f.get_name()].values.tolist()
        assert v == test[1]

    test = [np.nan, np.inf, np.inf, np.inf]
    v = df[features[-1].get_name()].values.tolist()
    assert (np.isnan(v[0]))
    assert v[1:] == test[1:]


def test_arithmetic_two_vals_fails(es):
    error_text = "Not a feature"
    with pytest.raises(Exception, match=error_text):
        ft.Feature([2, 2], primitive=AddNumeric)


def test_arithmetic_of_identity(es):
    logs = es['log']

    to_test = [(AddNumeric, [0., 7., 14., 21.]),
               (SubtractNumeric, [0, 3, 6, 9]),
               (MultiplyNumeric, [0, 10, 40, 90]),
               (DivideNumeric, [np.nan, 2.5, 2.5, 2.5])]

    features = []
    for test in to_test:
        features.append(ft.Feature([logs['value'], logs['value_2']], primitive=test[0]))

    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=[0, 1, 2, 3])
    if isinstance(df, dd.DataFrame):
        df = df.compute()

    for i, test in enumerate(to_test[:-1]):
        v = df[features[i].get_name()].values.tolist()
        assert v == test[1]
    i, test = 3, to_test[-1]
    v = df[features[i].get_name()].values.tolist()
    assert (np.isnan(v[0]))
    assert v[1:] == test[1][1:]


def test_arithmetic_of_direct(es):
    rating = es['products']['rating']
    log_rating = ft.Feature(rating, entity=es['log'])
    customer_age = es['customers']['age']
    session_age = ft.Feature(customer_age, entity=es['sessions'])
    log_age = ft.Feature(session_age, entity=es['log'])

    to_test = [(AddNumeric, [38, 37, 37.5, 37.5]),
               (SubtractNumeric, [28, 29, 28.5, 28.5]),
               (MultiplyNumeric, [165, 132, 148.5, 148.5]),
               (DivideNumeric, [6.6, 8.25, 22. / 3, 22. / 3])]

    features = []
    for test in to_test:
        features.append(ft.Feature([log_age, log_rating], primitive=test[0]))

    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=[0, 3, 5, 7])
    if isinstance(df, dd.DataFrame):
        df = df.compute().set_index('id').sort_index()

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert v == test[1]


@pytest.fixture(params=['pd_boolean_mult_es', 'dask_boolean_mult_es'])
def boolean_mult_es(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def pd_boolean_mult_es():
    es = ft.EntitySet()
    df = pd.DataFrame({"index": [0, 1, 2],
                       "bool": [True, False, True],
                       "numeric": [2, 3, np.nan]})

    es.entity_from_dataframe(entity_id="test",
                             dataframe=df,
                             index="index")

    return es


@pytest.fixture
def dask_boolean_mult_es(pd_boolean_mult_es):
    entities = {}
    for entity in pd_boolean_mult_es.entities:
        entities[entity.id] = (dd.from_pandas(entity.df, npartitions=2), entity.index, None, entity.variable_types)

    return ft.EntitySet(id=pd_boolean_mult_es.id, entities=entities)


def test_boolean_multiply(boolean_mult_es):
    es = boolean_mult_es
    to_test = [
        ('numeric', 'numeric'),
        ('numeric', 'bool'),
        ('bool', 'numeric'),
        ('bool', 'bool')
    ]
    features = []
    for row in to_test:
        features.append(ft.Feature(es["test"][row[0]]) * ft.Feature(es["test"][row[1]]))

    fm = ft.calculate_feature_matrix(entityset=es, features=features)

    if isinstance(fm, dd.DataFrame):
        fm = fm.compute()

    df = es['test'].df
    if isinstance(df, dd.DataFrame):
        df = df.compute()

    for row in to_test:
        col_name = '{} * {}'.format(row[0], row[1])
        if row[0] == 'bool' and row[1] == 'bool':
            assert fm[col_name].equals(df[row[0]] & df[row[1]])
        else:
            assert fm[col_name].equals(df[row[0]] * df[row[1]])


def test_arithmetic_of_transform(es):
    diff1 = ft.Feature([es['log']['value']], primitive=Diff)
    diff2 = ft.Feature([es['log']['value_2']], primitive=Diff)

    to_test = [(AddNumeric, [np.nan, 7., -7., 10.]),
               (SubtractNumeric, [np.nan, 3., -3., 4.]),
               (MultiplyNumeric, [np.nan, 10., 10., 21.]),
               (DivideNumeric, [np.nan, 2.5, 2.5, 2.3333333333333335])]

    features = []
    for test in to_test:
        features.append(ft.Feature([diff1, diff2], primitive=test[0]()))

    feature_set = FeatureSet(features)
    calculator = FeatureSetCalculator(es, feature_set=feature_set)
    df = calculator.run(np.array([0, 2, 12, 13]))
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert np.isnan(v.pop(0))
        assert np.isnan(test[1].pop(0))
        assert v == test[1]


def test_not_feature(es):
    not_feat = ft.Feature(es['customers']['loves_ice_cream'], primitive=Not)
    features = [not_feat]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=[0, 1])
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    v = df[not_feat.get_name()].values
    assert not v[0]
    assert v[1]


def test_arithmetic_of_agg(es):
    customer_id_feat = es['customers']['id']
    store_id_feat = es['stores']['id']
    count_customer = ft.Feature(customer_id_feat, parent_entity=es[u'régions'], primitive=Count)
    count_stores = ft.Feature(store_id_feat, parent_entity=es[u'régions'], primitive=Count)
    to_test = [(AddNumeric, [6, 2]),
               (SubtractNumeric, [0, -2]),
               (MultiplyNumeric, [9, 0]),
               (DivideNumeric, [1, 0])]

    features = []
    for test in to_test:
        features.append(ft.Feature([count_customer, count_stores], primitive=test[0]()))

    ids = ['United States', 'Mexico']
    df = ft.calculate_feature_matrix(entityset=es, features=features,
                                     instance_ids=ids)
    if isinstance(df, dd.DataFrame):
        df = df.compute().set_index('id')
    df = df.loc[ids]

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert v == test[1]


# TODO latlong is a string in entityset. Asserts in test_latlong fail
# def latlong_unstringify(latlong):
#     lat = float(latlong.split(", ")[0].replace("(", ""))
#     lon = float(latlong.split(", ")[1].replace(")", ""))
#     return (lat, lon)


def test_latlong(pd_es):
    log_latlong_feat = pd_es['log']['latlong']
    latitude = ft.Feature(log_latlong_feat, primitive=Latitude)
    longitude = ft.Feature(log_latlong_feat, primitive=Longitude)
    features = [latitude, longitude]
    df = ft.calculate_feature_matrix(entityset=pd_es, features=features, instance_ids=range(15))
    latvalues = df[latitude.get_name()].values
    lonvalues = df[longitude.get_name()].values
    assert len(latvalues) == 15
    assert len(lonvalues) == 15
    real_lats = [0, 5, 10, 15, 20, 0, 1, 2, 3, 0, 0, 5, 0, 7, 14]
    real_lons = [0, 2, 4, 6, 8, 0, 1, 2, 3, 0, 0, 2, 0, 3, 6]
    for i, v, in enumerate(real_lats):
        assert v == latvalues[i]
    for i, v, in enumerate(real_lons):
        assert v == lonvalues[i]


def test_latlong_with_nan(pd_es):
    df = pd_es['log'].df
    df['latlong'][0] = np.nan
    df['latlong'][1] = (10, np.nan)
    df['latlong'][2] = (np.nan, 4)
    df['latlong'][3] = (np.nan, np.nan)
    pd_es['log'].update_data(df)
    log_latlong_feat = pd_es['log']['latlong']
    latitude = ft.Feature(log_latlong_feat, primitive=Latitude)
    longitude = ft.Feature(log_latlong_feat, primitive=Longitude)
    features = [latitude, longitude]
    fm = ft.calculate_feature_matrix(entityset=pd_es, features=features)
    latvalues = fm[latitude.get_name()].values
    lonvalues = fm[longitude.get_name()].values
    assert len(latvalues) == 17
    assert len(lonvalues) == 17
    real_lats = [np.nan, 10, np.nan, np.nan, 20, 0, 1, 2, 3, 0, 0, 5, 0, 7, 14, np.nan, np.nan]
    real_lons = [np.nan, np.nan, 4, np.nan, 8, 0, 1, 2, 3, 0, 0, 2, 0, 3, 6, np.nan, np.nan]
    assert np.allclose(latvalues, real_lats, atol=0.0001, equal_nan=True)
    assert np.allclose(lonvalues, real_lons, atol=0.0001, equal_nan=True)


def test_haversine(pd_es):
    log_latlong_feat = pd_es['log']['latlong']
    log_latlong_feat2 = pd_es['log']['latlong2']
    haversine = ft.Feature([log_latlong_feat, log_latlong_feat2],
                           primitive=Haversine)
    features = [haversine]

    df = ft.calculate_feature_matrix(entityset=pd_es, features=features,
                                     instance_ids=range(15))
    values = df[haversine.get_name()].values
    real = [0, 525.318462, 1045.32190304, 1554.56176802, 2047.3294327, 0,
            138.16578931, 276.20524822, 413.99185444, 0, 0, 525.318462, 0,
            741.57941183, 1467.52760175]
    assert len(values) == 15
    assert np.allclose(values, real, atol=0.0001)

    haversine = ft.Feature([log_latlong_feat, log_latlong_feat2],
                           primitive=Haversine(unit='kilometers'))
    features = [haversine]
    df = ft.calculate_feature_matrix(entityset=pd_es, features=features,
                                     instance_ids=range(15))
    values = df[haversine.get_name()].values
    real_km = [0, 845.41812212, 1682.2825471, 2501.82467535, 3294.85736668,
               0, 222.35628593, 444.50926278, 666.25531268, 0, 0,
               845.41812212, 0, 1193.45638714, 2361.75676089]
    assert len(values) == 15
    assert np.allclose(values, real_km, atol=0.0001)
    error_text = "Invalid unit inches provided. Must be one of"
    with pytest.raises(ValueError, match=error_text):
        Haversine(unit='inches')


def test_haversine_with_nan(pd_es):
    # Check some `nan` values
    df = pd_es['log'].df
    df['latlong'][0] = np.nan
    df['latlong'][1] = (10, np.nan)
    pd_es['log'].update_data(df)
    log_latlong_feat = pd_es['log']['latlong']
    log_latlong_feat2 = pd_es['log']['latlong2']
    haversine = ft.Feature([log_latlong_feat, log_latlong_feat2],
                           primitive=Haversine)
    features = [haversine]

    df = ft.calculate_feature_matrix(entityset=pd_es, features=features)
    values = df[haversine.get_name()].values
    real = [np.nan, np.nan, 1045.32190304, 1554.56176802, 2047.3294327, 0,
            138.16578931, 276.20524822, 413.99185444, 0, 0, 525.318462, 0,
            741.57941183, 1467.52760175, np.nan, np.nan]

    assert np.allclose(values, real, atol=0.0001, equal_nan=True)

    # Check all `nan` values
    df = pd_es['log'].df
    df['latlong2'] = np.nan
    pd_es['log'].update_data(df)
    log_latlong_feat = pd_es['log']['latlong']
    log_latlong_feat2 = pd_es['log']['latlong2']
    haversine = ft.Feature([log_latlong_feat, log_latlong_feat2],
                           primitive=Haversine)
    features = [haversine]

    df = ft.calculate_feature_matrix(entityset=pd_es, features=features)
    values = df[haversine.get_name()].values
    real = [np.nan] * pd_es['log'].df.shape[0]

    assert np.allclose(values, real, atol=0.0001, equal_nan=True)


def test_text_primitives(es):
    words = ft.Feature(es['log']['comments'], primitive=NumWords)
    chars = ft.Feature(es['log']['comments'], primitive=NumCharacters)

    features = [words, chars]

    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(15))
    if isinstance(df, dd.DataFrame):
        df = df.compute()

    word_counts = [514, 3, 3, 644, 1268, 1269, 177, 172, 79,
                   240, 1239, 3, 3, 3, 3]
    char_counts = [3392, 10, 10, 4116, 7961, 7580, 992, 957,
                   437, 1325, 6322, 10, 10, 10, 10]
    word_values = df[words.get_name()].values
    char_values = df[chars.get_name()].values
    assert len(word_values) == 15
    for i, v in enumerate(word_values):
        assert v == word_counts[i]
    for i, v in enumerate(char_values):
        assert v == char_counts[i]


def test_isin_feat(es):
    isin = ft.Feature(es['log']['product_id'], primitive=IsIn(list_of_outputs=["toothpaste", "coke zero"]))
    features = [isin]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(8))
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    true = [True, True, True, False, False, True, True, True]
    v = df[isin.get_name()].values.tolist()
    assert true == v


def test_isin_feat_other_syntax(es):
    isin = ft.Feature(es['log']['product_id']).isin(["toothpaste", "coke zero"])
    features = [isin]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(8))
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    true = [True, True, True, False, False, True, True, True]
    v = df[isin.get_name()].values.tolist()
    assert true == v


def test_isin_feat_other_syntax_int(es):
    isin = ft.Feature(es['log']['value']).isin([5, 10])
    features = [isin]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(8))
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    true = [False, True, True, False, False, False, False, False]
    v = df[isin.get_name()].values.tolist()
    assert true == v


def test_isin_feat_custom(es):
    def pd_is_in(array, list_of_outputs=None):
        if list_of_outputs is None:
            list_of_outputs = []
        if isinstance(array, dd.Series):
            return array.isin(list_of_outputs)
        return pd.Series(array).isin(list_of_outputs)

    def isin_generate_name(self, base_feature_names):
        return u"%s.isin(%s)" % (base_feature_names[0],
                                 str(self.kwargs['list_of_outputs']))

    IsIn = make_trans_primitive(
        pd_is_in,
        [Variable],
        Boolean,
        name="is_in",
        description="For each value of the base feature, checks whether it is "
        "in a list that is provided.",
        cls_attributes={"generate_name": isin_generate_name})

    isin = ft.Feature(es['log']['product_id'], primitive=IsIn(list_of_outputs=["toothpaste", "coke zero"]))
    features = [isin]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(8))
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    true = [True, True, True, False, False, True, True, True]
    v = df[isin.get_name()].values.tolist()
    assert true == v

    isin = ft.Feature(es['log']['product_id']).isin(["toothpaste", "coke zero"])
    features = [isin]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(8))
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    true = [True, True, True, False, False, True, True, True]
    v = df[isin.get_name()].values.tolist()
    assert true == v

    isin = ft.Feature(es['log']['value']).isin([5, 10])
    features = [isin]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(8))
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    true = [False, True, True, False, False, False, False, False]
    v = df[isin.get_name()].values.tolist()
    assert true == v


def test_isnull_feat(pd_es):
    value = ft.Feature(pd_es['log']['value'])
    diff = ft.Feature(value, groupby=pd_es['log']['session_id'], primitive=Diff)
    isnull = ft.Feature(diff, primitive=IsNull)
    features = [isnull]
    df = ft.calculate_feature_matrix(entityset=pd_es, features=features, instance_ids=range(15))
    # correct_vals_diff = [
    #     np.nan, 5, 5, 5, 5, np.nan, 1, 1, 1, np.nan, np.nan, 5, np.nan, 7, 7]
    correct_vals = [True, False, False, False, False, True, False, False,
                    False, True, True, False, True, False, False]
    values = df[isnull.get_name()].values.tolist()
    assert correct_vals == values


def test_percentile(pd_es):
    v = ft.Feature(pd_es['log']['value'])
    p = ft.Feature(v, primitive=Percentile)
    feature_set = FeatureSet([p])
    calculator = FeatureSetCalculator(pd_es, feature_set)
    df = calculator.run(np.array(range(10, 17)))
    true = pd_es['log'].df[v.get_name()].rank(pct=True)
    true = true.loc[range(10, 17)]
    for t, a in zip(true.values, df[p.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_dependent_percentile(pd_es):
    v = ft.Feature(pd_es['log']['value'])
    p = ft.Feature(v, primitive=Percentile)
    p2 = ft.Feature(p - 1, primitive=Percentile)
    feature_set = FeatureSet([p, p2])
    calculator = FeatureSetCalculator(pd_es, feature_set)
    df = calculator.run(np.array(range(10, 17)))
    true = pd_es['log'].df[v.get_name()].rank(pct=True)
    true = true.loc[range(10, 17)]
    for t, a in zip(true.values, df[p.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_agg_percentile(pd_es):
    v = ft.Feature(pd_es['log']['value'])
    p = ft.Feature(v, primitive=Percentile)
    agg = ft.Feature(p, parent_entity=pd_es['sessions'], primitive=Sum)
    feature_set = FeatureSet([agg])
    calculator = FeatureSetCalculator(pd_es, feature_set)
    df = calculator.run(np.array([0, 1]))
    log_vals = pd_es['log'].df[[v.get_name(), 'session_id']]
    log_vals['percentile'] = log_vals[v.get_name()].rank(pct=True)
    true_p = log_vals.groupby('session_id')['percentile'].sum()[[0, 1]]
    for t, a in zip(true_p.values, df[agg.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_percentile_agg_percentile(pd_es):
    v = ft.Feature(pd_es['log']['value'])
    p = ft.Feature(v, primitive=Percentile)
    agg = ft.Feature(p, parent_entity=pd_es['sessions'], primitive=Sum)
    pagg = ft.Feature(agg, primitive=Percentile)
    feature_set = FeatureSet([pagg])
    calculator = FeatureSetCalculator(pd_es, feature_set)
    df = calculator.run(np.array([0, 1]))

    log_vals = pd_es['log'].df[[v.get_name(), 'session_id']]
    log_vals['percentile'] = log_vals[v.get_name()].rank(pct=True)
    true_p = log_vals.groupby('session_id')['percentile'].sum().fillna(0)
    true_p = true_p.rank(pct=True)[[0, 1]]

    for t, a in zip(true_p.values, df[pagg.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_percentile_agg(pd_es):
    v = ft.Feature(pd_es['log']['value'])
    agg = ft.Feature(v, parent_entity=pd_es['sessions'], primitive=Sum)
    pagg = ft.Feature(agg, primitive=Percentile)
    feature_set = FeatureSet([pagg])
    calculator = FeatureSetCalculator(pd_es, feature_set)
    df = calculator.run(np.array([0, 1]))

    log_vals = pd_es['log'].df[[v.get_name(), 'session_id']]
    true_p = log_vals.groupby('session_id')[v.get_name()].sum().fillna(0)
    true_p = true_p.rank(pct=True)[[0, 1]]

    for t, a in zip(true_p.values, df[pagg.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_direct_percentile(pd_es):
    v = ft.Feature(pd_es['customers']['age'])
    p = ft.Feature(v, primitive=Percentile)
    d = ft.Feature(p, pd_es['sessions'])
    feature_set = FeatureSet([d])
    calculator = FeatureSetCalculator(pd_es, feature_set)
    df = calculator.run(np.array([0, 1]))

    cust_vals = pd_es['customers'].df[[v.get_name()]]
    cust_vals['percentile'] = cust_vals[v.get_name()].rank(pct=True)
    true_p = cust_vals['percentile'].loc[[0, 0]]
    for t, a in zip(true_p.values, df[d.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_direct_agg_percentile(pd_es):
    v = ft.Feature(pd_es['log']['value'])
    p = ft.Feature(v, primitive=Percentile)
    agg = ft.Feature(p, parent_entity=pd_es['customers'], primitive=Sum)
    d = ft.Feature(agg, pd_es['sessions'])
    feature_set = FeatureSet([d])
    calculator = FeatureSetCalculator(pd_es, feature_set)
    df = calculator.run(np.array([0, 1]))

    log_vals = pd_es['log'].df[[v.get_name(), 'session_id']]
    log_vals['percentile'] = log_vals[v.get_name()].rank(pct=True)
    log_vals['customer_id'] = [0] * 10 + [1] * 5 + [2] * 2
    true_p = log_vals.groupby('customer_id')['percentile'].sum().fillna(0)
    true_p = true_p[[0, 0]]
    for t, a in zip(true_p.values, df[d.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or round(t, 3) == round(a, 3)


def test_percentile_with_cutoff(pd_es):
    v = ft.Feature(pd_es['log']['value'])
    p = ft.Feature(v, primitive=Percentile)
    feature_set = FeatureSet([p])
    calculator = FeatureSetCalculator(pd_es, feature_set, pd.Timestamp('2011/04/09 10:30:13'))
    df = calculator.run(np.array([2]))
    assert df[p.get_name()].tolist()[0] == 1.0


def test_two_kinds_of_dependents(pd_es):
    v = ft.Feature(pd_es['log']['value'])
    product = ft.Feature(pd_es['log']['product_id'])
    agg = ft.Feature(v, parent_entity=pd_es['customers'], where=product == 'coke zero', primitive=Sum)
    p = ft.Feature(agg, primitive=Percentile)
    g = ft.Feature(agg, primitive=Absolute)
    agg2 = ft.Feature(v, parent_entity=pd_es['sessions'], where=product == 'coke zero', primitive=Sum)
    agg3 = ft.Feature(agg2, parent_entity=pd_es['customers'], primitive=Sum)
    feature_set = FeatureSet([p, g, agg3])
    calculator = FeatureSetCalculator(pd_es, feature_set)
    df = calculator.run(np.array([0, 1]))
    assert df[p.get_name()].tolist() == [2. / 3, 1.0]
    assert df[g.get_name()].tolist() == [15, 26]


def test_make_transform_restricts_time_keyword():
    make_trans_primitive(
        lambda x, time=False: x,
        [Datetime],
        Numeric,
        name="AllowedPrimitive",
        description="This primitive should be accepted",
        uses_calc_time=True)

    error_text = "'time' is a restricted keyword.  Please use a different keyword."
    with pytest.raises(ValueError, match=error_text):
        make_trans_primitive(
            lambda x, time=False: x,
            [Datetime],
            Numeric,
            name="BadPrimitive",
            description="This primitive should error")


def test_make_transform_restricts_time_arg():
    make_trans_primitive(
        lambda time: time,
        [Datetime],
        Numeric,
        name="AllowedPrimitive",
        description="This primitive should be accepted",
        uses_calc_time=True)

    error_text = "'time' is a restricted keyword.  Please use a different keyword."
    with pytest.raises(ValueError, match=error_text):
        make_trans_primitive(
            lambda time: time,
            [Datetime],
            Numeric,
            name="BadPrimitive",
            description="This primitive should erorr")


def test_make_transform_sets_kwargs_correctly(es):
    def pd_is_in(array, list_of_outputs=None):
        if list_of_outputs is None:
            list_of_outputs = []
        return pd.Series(array).isin(list_of_outputs)

    def isin_generate_name(self, base_feature_names):
        return u"%s.isin(%s)" % (base_feature_names[0],
                                 str(self.kwargs['list_of_outputs']))

    IsIn = make_trans_primitive(
        pd_is_in,
        [Variable],
        Boolean,
        name="is_in",
        description="For each value of the base feature, checks whether it is "
        "in a list that is provided.",
        cls_attributes={"generate_name": isin_generate_name})

    isin_1_list = ["toothpaste", "coke_zero"]
    isin_1_base_f = ft.Feature(es['log']['product_id'])
    isin_1 = ft.Feature(isin_1_base_f, primitive=IsIn(list_of_outputs=isin_1_list))
    isin_2_list = ["coke_zero"]
    isin_2_base_f = ft.Feature(es['log']['session_id'])
    isin_2 = ft.Feature(isin_2_base_f, primitive=IsIn(list_of_outputs=isin_2_list))
    assert isin_1_base_f == isin_1.base_features[0]
    assert isin_1_list == isin_1.primitive.kwargs['list_of_outputs']
    assert isin_2_base_f == isin_2.base_features[0]
    assert isin_2_list == isin_2.primitive.kwargs['list_of_outputs']


def test_make_transform_multiple_output_features(pd_es):
    def test_time(x):
        times = pd.Series(x)
        units = ["year", "month", "day", "hour", "minute", "second"]
        return [times.apply(lambda x: getattr(x, unit)) for unit in units]

    def gen_feat_names(self):
        subnames = ["Year", "Month", "Day", "Hour", "Minute", "Second"]
        return ["Now.%s(%s)" % (subname, self.base_features[0].get_name())
                for subname in subnames]

    TestTime = make_trans_primitive(
        function=test_time,
        input_types=[Datetime],
        return_type=Numeric,
        number_output_features=6,
        cls_attributes={"get_feature_names": gen_feat_names},
    )

    join_time_split = ft.Feature(pd_es["log"]["datetime"], primitive=TestTime)
    alt_features = [ft.Feature(pd_es["log"]["datetime"], primitive=Year),
                    ft.Feature(pd_es["log"]["datetime"], primitive=Month),
                    ft.Feature(pd_es["log"]["datetime"], primitive=Day),
                    ft.Feature(pd_es["log"]["datetime"], primitive=Hour),
                    ft.Feature(pd_es["log"]["datetime"], primitive=Minute),
                    ft.Feature(pd_es["log"]["datetime"], primitive=Second)]
    fm, fl = ft.dfs(
        entityset=pd_es,
        target_entity="log",
        agg_primitives=[],
        trans_primitives=[TestTime, Year, Month, Day, Hour, Minute, Second, Diff],
        max_depth=5)

    subnames = join_time_split.get_feature_names()
    altnames = [f.get_name() for f in alt_features]
    for col1, col2 in zip(subnames, altnames):
        assert (fm[col1] == fm[col2]).all()

    for i in range(6):
        f = 'sessions.customers.DIFF(TEST_TIME(date_of_birth)[%d])' % i
        assert feature_with_name(fl, f)
        assert ('DIFF(TEST_TIME(datetime)[%d])' % i) in fl


def test_feature_names_inherit_from_make_trans_primitive():
    # R TODO
    pass


def test_get_filepath(es):
    class Mod4(TransformPrimitive):
        '''Return base feature modulo 4'''
        name = "mod4"
        input_types = [Numeric]
        return_type = Numeric
        dask_compatible = True

        def get_function(self):
            filepath = self.get_filepath("featuretools_unit_test_example.csv")
            reference = pd.read_csv(filepath, header=None, squeeze=True)

            def map_to_word(x):
                def _map(x):
                    if pd.isnull(x):
                        return x
                    return reference[int(x) % 4]
                return x.apply(_map)
            return map_to_word

    feat = ft.Feature(es['log']['value'], primitive=Mod4)
    df = ft.calculate_feature_matrix(features=[feat],
                                     entityset=es,
                                     instance_ids=range(17))
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    assert pd.isnull(df["MOD4(value)"][15])
    assert df["MOD4(value)"][0] == 0
    assert df["MOD4(value)"][14] == 2

    fm, fl = ft.dfs(entityset=es,
                    target_entity="log",
                    agg_primitives=[],
                    trans_primitives=[Mod4])
    if isinstance(fm, dd.DataFrame):
        fm = fm.compute().set_index('id')
    assert fm["MOD4(value)"][0] == 0
    assert fm["MOD4(value)"][14] == 2
    assert pd.isnull(fm["MOD4(value)"][15])


def test_override_multi_feature_names(pd_es):
    def gen_custom_names(primitive, base_feature_names):
        return ['Above18(%s)' % base_feature_names,
                'Above21(%s)' % base_feature_names,
                'Above65(%s)' % base_feature_names]

    def is_greater(x):
        return x > 18, x > 21, x > 65

    num_features = 3
    IsGreater = make_trans_primitive(function=is_greater,
                                     input_types=[Numeric],
                                     return_type=Numeric,
                                     number_output_features=num_features,
                                     cls_attributes={"generate_names": gen_custom_names})

    fm, features = ft.dfs(entityset=pd_es,
                          target_entity="customers",
                          instance_ids=[0, 1, 2],
                          agg_primitives=[],
                          trans_primitives=[IsGreater])

    expected_names = gen_custom_names(IsGreater, ['age'])

    for name in expected_names:
        assert name in fm.columns
