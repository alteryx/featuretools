# -*- coding: utf-8 -*-

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ..testing_utils import feature_with_name, make_ecommerce_entityset

import featuretools as ft
from featuretools.primitives import (
    Count,
    Mean,
    Median,
    NMostCommon,
    NumTrue,
    Sum,
    TimeSinceLast,
    get_aggregation_primitives
)
from featuretools.primitives.base import (
    AggregationPrimitive,
    Feature,
    make_agg_primitive
)
from featuretools.synthesis.deep_feature_synthesis import (
    DeepFeatureSynthesis,
    check_stacking,
    match
)
from featuretools.variable_types import (
    Datetime,
    DatetimeTimeIndex,
    Discrete,
    Index,
    Numeric,
    Variable
)


@pytest.fixture(scope='module')
def es():
    return make_ecommerce_entityset()


@pytest.fixture
def test_primitive():
    class TestAgg(AggregationPrimitive):
        name = "test"
        input_types = [Numeric]
        return_type = Numeric
        stack_on = []

        def get_function(self):
            return None

    return TestAgg


def test_get_depth(es):
    log_id_feat = es['log']['id']
    customer_id_feat = es['customers']['id']
    count_logs = Count(log_id_feat,
                       parent_entity=es['sessions'])
    sum_count_logs = Sum(count_logs,
                         parent_entity=es['customers'])
    num_logs_greater_than_5 = sum_count_logs > 5
    count_customers = Count(customer_id_feat,
                            parent_entity=es[u'régions'],
                            where=num_logs_greater_than_5)
    num_customers_region = Feature(count_customers, es["customers"])

    depth = num_customers_region.get_depth()
    assert depth == 5


def test_makes_count(es):
    dfs = DeepFeatureSynthesis(target_entity_id='sessions',
                               entityset=es,
                               agg_primitives=[Count],
                               trans_primitives=[])

    features = dfs.build_features()
    assert feature_with_name(features, 'device_type')
    assert feature_with_name(features, 'customer_id')
    assert feature_with_name(features, u'customers.région_id')
    assert feature_with_name(features, 'customers.age')
    assert feature_with_name(features, 'COUNT(log)')
    assert feature_with_name(features, 'customers.COUNT(sessions)')
    assert feature_with_name(features, u'customers.régions.language')
    assert feature_with_name(features, 'customers.COUNT(log)')


def test_count_null_and_make_agg_primitive(es):
    def count_func(values, count_null=False):
        if len(values) == 0:
            return 0

        if count_null:
            values = values.fillna(0)

        return values.count()

    def count_generate_name(self):
        where_str = self._where_str()
        use_prev_str = self._use_prev_str()
        return u"COUNT(%s%s%s)" % (self.child_entity.id,
                                   where_str,
                                   use_prev_str)

    Count = make_agg_primitive(count_func, [[Index], [Variable]], Numeric,
                               name="count", stack_on_self=False,
                               cls_attributes={"generate_name": count_generate_name})
    count_null = Count(es['log']['value'], es['sessions'], count_null=True)
    feature_matrix = ft.calculate_feature_matrix([count_null], entityset=es)
    values = [5, 4, 1, 2, 3, 2]
    assert (values == feature_matrix[count_null.get_name()]).all()


def test_check_input_types(es):
    count = Count(es["sessions"]["id"], es["customers"])
    mean = Mean(count, es[u"régions"])
    assert mean._check_input_types()

    boolean = count > 3
    mean = Mean(count, es[u"régions"], where=boolean)
    assert mean._check_input_types()


def test_base_of_and_stack_on_heuristic(es, test_primitive):
    child = Count(es["sessions"]["id"], es["customers"])
    test_primitive.stack_on = []
    child.base_of = []
    assert not (check_stacking(test_primitive, [child]))

    test_primitive.stack_on = []
    child.base_of = None
    assert (check_stacking(test_primitive, [child]))

    test_primitive.stack_on = []
    child.base_of = [test_primitive]
    assert (check_stacking(test_primitive, [child]))

    test_primitive.stack_on = None
    child.base_of = []
    assert (check_stacking(test_primitive, [child]))

    test_primitive.stack_on = None
    child.base_of = None
    assert (check_stacking(test_primitive, [child]))

    test_primitive.stack_on = None
    child.base_of = [test_primitive]
    assert (check_stacking(test_primitive, [child]))

    test_primitive.stack_on = [child]
    child.base_of = []
    assert (check_stacking(test_primitive, [child]))

    test_primitive.stack_on = [child]
    child.base_of = None
    assert (check_stacking(test_primitive, [child]))

    test_primitive.stack_on = [child]
    child.base_of = [test_primitive]
    assert (check_stacking(test_primitive, [child]))


def test_stack_on_self(es, test_primitive):
    # test stacks on self
    child = test_primitive(es['log']['value'], es[u'régions'])
    test_primitive.stack_on = []
    child.base_of = []
    test_primitive.stack_on_self = False
    child.stack_on_self = False
    assert not (check_stacking(test_primitive, [child]))

    test_primitive.stack_on_self = True
    assert (check_stacking(test_primitive, [child]))

    test_primitive.stack_on = None
    test_primitive.stack_on_self = False
    assert not (check_stacking(test_primitive, [child]))


def test_stack_expanding(es, test_primitive):
    test_primitive.input_types = [Discrete]
    expanding_primitive = NMostCommon(es['sessions']['device_type'], es[u'régions'])
    assert not (check_stacking(test_primitive, [expanding_primitive]))


# P TODO: this functionality is currently missing
# def test_max_depth_heuristic(es, parent_class, parent_entity, parent):
#     grandparent = make_parent_instance(parent_class, parent_entity,
#                                        parent)
#     for f in [parent, grandparent]:
#         f.stack_on = ['child']
#         f.stack_on_self = True
#         f.base_of = ['parent']
#         f.apply_to = [(Numeric,)]
#         f.max_stack_depth = 2

#     assert parent.can_apply(parent_entity, 'customers')
#     assert not grandparent.can_apply(parent_entity, 'customers')

#     grandparent.max_stack_depth = 3
#     assert grandparent.can_apply(parent_entity, 'customers')

def test_init_and_name(es):
    session = es['sessions']
    log = es['log']

    features = [Feature(v) for v in log.variables]
    for agg_prim in get_aggregation_primitives().values():

        input_types = agg_prim.input_types
        if type(input_types[0]) != list:
            input_types = [input_types]

        # test each allowed input_types for this primitive
        for it in input_types:
            # use the input_types matching function from DFS
            matching_types = match(it, features)
            if len(matching_types) == 0:
                raise Exception("Agg Primitive %s not tested" % agg_prim.name)
            for t in matching_types:
                instance = agg_prim(t, parent_entity=session)

                # try to get name and calculate
                instance.get_name()
                ft.calculate_feature_matrix([instance], entityset=es).head(5)


def test_time_since_last(es):
    f = TimeSinceLast(es["log"]["datetime"], es["customers"])
    fm = ft.calculate_feature_matrix([f],
                                     entityset=es,
                                     instance_ids=[0, 1, 2],
                                     cutoff_time=datetime(2015, 6, 8))

    correct = [131376600, 131289600, 131287800]
    # note: must round to nearest second
    assert all(fm[f.get_name()].round().values == correct)


def test_median(es):
    f = Median(es["log"]["value_many_nans"], es["customers"])
    fm = ft.calculate_feature_matrix([f],
                                     entityset=es,
                                     instance_ids=[0, 1, 2],
                                     cutoff_time=datetime(2015, 6, 8))

    correct = [1, 3, np.nan]
    np.testing.assert_equal(fm[f.get_name()].values, correct)


def test_agg_same_method_name(es):
    """
        Pandas relies on the function name when calculating aggregations. This means if a two
        primitives with the same function name are applied to the same column, pandas
        can't differentiate them. We have a work around to this based on the name property
        that we test here.
    """

    # test with normally defined functions
    def custom_primitive(x):
        return x.sum()

    Sum = make_agg_primitive(custom_primitive, input_types=[Numeric],
                             return_type=Numeric, name="sum")

    def custom_primitive(x):
        return x.max()

    Max = make_agg_primitive(custom_primitive, input_types=[Numeric],
                             return_type=Numeric, name="max")

    f_sum = Sum(es["log"]["value"], es["customers"])
    f_max = Max(es["log"]["value"], es["customers"])

    fm = ft.calculate_feature_matrix([f_sum, f_max], entityset=es)
    assert fm.columns.tolist() == [f_sum.get_name(), f_max.get_name()]

    # test with lambdas
    Sum = make_agg_primitive(lambda x: x.sum(), input_types=[Numeric],
                             return_type=Numeric, name="sum")
    Max = make_agg_primitive(lambda x: x.max(), input_types=[Numeric],
                             return_type=Numeric, name="max")

    f_sum = Sum(es["log"]["value"], es["customers"])
    f_max = Max(es["log"]["value"], es["customers"])
    fm = ft.calculate_feature_matrix([f_sum, f_max], entityset=es)
    assert fm.columns.tolist() == [f_sum.get_name(), f_max.get_name()]


def test_time_since_last_custom(es):
    def time_since_last(values, time=None):
        time_since = time - values.iloc[0]
        return time_since.total_seconds()

    TimeSinceLast = make_agg_primitive(time_since_last,
                                       [DatetimeTimeIndex],
                                       Numeric,
                                       name="time_since_last",
                                       uses_calc_time=True)
    f = TimeSinceLast(es["log"]["datetime"], es["customers"])
    fm = ft.calculate_feature_matrix([f],
                                     entityset=es,
                                     instance_ids=[0, 1, 2],
                                     cutoff_time=datetime(2015, 6, 8))

    correct = [131376600, 131289600, 131287800]
    # note: must round to nearest second
    assert all(fm[f.get_name()].round().values == correct)

    error_text = "'time' is a restricted keyword.  Please use a different keyword."
    with pytest.raises(ValueError, match=error_text):
        TimeSinceLast = make_agg_primitive(time_since_last,
                                           [DatetimeTimeIndex],
                                           Numeric,
                                           uses_calc_time=False)


def test_custom_primitive_time_as_arg(es):
    def time_since_last(values, time):
        time_since = time - values.iloc[0]
        return time_since.total_seconds()

    TimeSinceLast = make_agg_primitive(time_since_last,
                                       [DatetimeTimeIndex],
                                       Numeric,
                                       uses_calc_time=True)
    assert TimeSinceLast.name == "time_since_last"
    f = TimeSinceLast(es["log"]["datetime"], es["customers"])
    fm = ft.calculate_feature_matrix([f],
                                     entityset=es,
                                     instance_ids=[0, 1, 2],
                                     cutoff_time=datetime(2015, 6, 8))

    correct = [131376600, 131289600, 131287800]
    # note: must round to nearest second
    assert all(fm[f.get_name()].round().values == correct)

    error_text = "'time' is a restricted keyword.  Please use a different keyword."
    with pytest.raises(ValueError, match=error_text):
        make_agg_primitive(time_since_last,
                           [DatetimeTimeIndex],
                           Numeric,
                           uses_calc_time=False)


def test_custom_primitive_multiple_inputs(es):
    def mean_sunday(numeric, datetime):
        '''
        Finds the mean of non-null values of a feature that occurred on Sundays
        '''
        days = pd.DatetimeIndex(datetime).weekday.values
        df = pd.DataFrame({'numeric': numeric, 'time': days})
        return df[df['time'] == 6]['numeric'].mean()

    MeanSunday = make_agg_primitive(function=mean_sunday,
                                    input_types=[Numeric, Datetime],
                                    return_type=Numeric)

    fm, features = ft.dfs(entityset=es,
                          target_entity="sessions",
                          agg_primitives=[MeanSunday],
                          trans_primitives=[])
    mean_sunday_value = pd.Series([None, None, None, 2.5, 7, None])
    iterator = zip(fm["MEAN_SUNDAY(log.value, datetime)"], mean_sunday_value)
    for x, y in iterator:
        assert ((pd.isnull(x) and pd.isnull(y)) or (x == y))

    es.add_interesting_values()
    mean_sunday_value_priority_0 = pd.Series([None, None, None, 2.5, 0, None])
    fm, features = ft.dfs(entityset=es,
                          target_entity="sessions",
                          agg_primitives=[MeanSunday],
                          trans_primitives=[],
                          where_primitives=[MeanSunday])
    where_feat = "MEAN_SUNDAY(log.value, datetime WHERE priority_level = 0)"
    for x, y in zip(fm[where_feat], mean_sunday_value_priority_0):
        assert ((pd.isnull(x) and pd.isnull(y)) or (x == y))


def test_custom_primitive_default_kwargs(es):
    def sum_n_times(numeric, n=1):
        return np.nan_to_num(numeric).sum(dtype=np.float) * n

    SumNTimes = make_agg_primitive(function=sum_n_times,
                                   input_types=[Numeric],
                                   return_type=Numeric)

    sum_n_1_n = 1
    sum_n_1_base_f = Feature(es['log']['value'])
    sum_n_1 = SumNTimes([sum_n_1_base_f], es['sessions'], n=sum_n_1_n)
    sum_n_2_n = 2
    sum_n_2_base_f = Feature(es['log']['value_2'])
    sum_n_2 = SumNTimes([sum_n_2_base_f], es['sessions'], n=sum_n_2_n)
    assert sum_n_1_base_f == sum_n_1.base_features[0]
    assert sum_n_1_n == sum_n_1.kwargs['n']
    assert sum_n_2_base_f == sum_n_2.base_features[0]
    assert sum_n_2_n == sum_n_2.kwargs['n']


def test_makes_numtrue(es):
    dfs = DeepFeatureSynthesis(target_entity_id='sessions',
                               entityset=es,
                               agg_primitives=[NumTrue],
                               trans_primitives=[])
    features = dfs.build_features()
    assert feature_with_name(features, 'customers.NUM_TRUE(log.purchased)')
    assert feature_with_name(features, 'NUM_TRUE(log.purchased)')
