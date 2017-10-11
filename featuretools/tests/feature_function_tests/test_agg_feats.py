import pytest

from featuretools.synthesis.deep_feature_synthesis import (DeepFeatureSynthesis,
                                                           check_stacking, match)
from featuretools.primitives import(Feature, Count, Mean, Sum, TimeSinceLast,
                                    AggregationPrimitive, NumTrue,
                                    get_aggregation_primitives)
from featuretools.variable_types import (Discrete, Numeric, Categorical,
                                         Ordinal, Boolean, Text, Datetime)
from featuretools import calculate_feature_matrix
from ..testing_utils import make_ecommerce_entityset, feature_with_name
from datetime import datetime


@pytest.fixture(scope='module')
def es():
    return make_ecommerce_entityset()


@pytest.fixture
def child_entity(es):
    return es['customers']


@pytest.fixture
def grandchild_entity(es):
    return es['sessions']


@pytest.fixture
def child(es, child_entity):
    return Count(es['sessions']['id'],
                 parent_entity=child_entity)


@pytest.fixture
def parent_class():
    return Mean


@pytest.fixture
def parent_entity(es):
    return es['regions']


@pytest.fixture
def parent(parent_class, parent_entity, child):
    return make_parent_instance(parent_class,
                                parent_entity, child)


@pytest.fixture
def test_primitive():
    class TestAgg(AggregationPrimitive):
        name = "test"
        input_types =  [Numeric]
        return_type = Numeric
        stack_on = []

        def get_function(self):
            return None

    return TestAgg


def make_parent_instance(parent_class, parent_entity, base_feature,
                         where=None):
    return parent_class(base_feature, parent_entity, where=where)


def test_get_depth(es):
    log_id_feat = es['log']['id']
    customer_id_feat = es['customers']['id']
    count_logs = Count(log_id_feat,
                       parent_entity=es['sessions'])
    sum_count_logs = Sum(count_logs,
                         parent_entity=es['customers'])
    num_logs_greater_than_5 = sum_count_logs > 5
    count_customers = Count(customer_id_feat,
                            parent_entity=es['regions'],
                            where=num_logs_greater_than_5)
    num_customers_region = Feature(count_customers, es["customers"])

    depth = num_customers_region.get_depth()
    assert depth == 5


def test_makes_count(es):
    dfs = DeepFeatureSynthesis(target_entity_id='sessions',
                               entityset=es,
                               filters=[],
                               agg_primitives=[Count],
                               trans_primitives=[])

    features = dfs.build_features()
    assert feature_with_name(features, 'device_type')
    assert feature_with_name(features, 'customer_id')
    assert feature_with_name(features, 'customers.region_id')
    assert feature_with_name(features, 'customers.age')
    assert feature_with_name(features, 'COUNT(log)')
    assert feature_with_name(features, 'customers.COUNT(sessions)')
    assert feature_with_name(features, 'customers.regions.language')
    assert feature_with_name(features, 'customers.COUNT(log)')


def test_check_input_types(es, child, parent):
    mean = parent
    assert mean._check_input_types()
    boolean = child > 3
    mean = make_parent_instance(Mean, es['regions'],
                                child, where=boolean)
    assert mean._check_input_types()


def test_base_of_and_stack_on_heuristic(es, test_primitive, child):
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


def test_stack_on_self(es, test_primitive, parent_entity):
    # test stacks on self
    child = test_primitive(es['log']['value'], parent_entity)
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


# P TODO: this functionality is currently missing
# def test_max_depth_heuristic(es, parent_class, parent_entity, parent):
#     grandparent = make_parent_instance(parent_class, parent_entity,
#                                        parent)
#     for f in [parent, grandparent]:
#         f.stack_on = ['child']
#         f.stacks_on_self = True
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
    for agg_prim in get_aggregation_primitives():

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
                instance = agg_prim(*t, parent_entity=session)

                # try to get name and calculate
                instance.get_name()
                instance.head()


def test_time_since_last(es):
    f = TimeSinceLast(es["log"]["datetime"], es["customers"])
    fm = calculate_feature_matrix([f], instance_ids=[0, 1, 2], cutoff_time=datetime(2015, 6, 8))

    correct = [131376600, 131289600, 131287800]
    # note: must round to nearest second
    assert all(fm[f.get_name()].round().values == correct)


def test_makes_numtrue(es):
    dfs = DeepFeatureSynthesis(target_entity_id='sessions',
                               entityset=es,
                               filters=[],
                               agg_primitives=[NumTrue],
                               trans_primitives=[])
    features = dfs.build_features()
    assert feature_with_name(features, 'customers.NUM_TRUE(log.purchased)')
    assert feature_with_name(features, 'NUM_TRUE(log.purchased)')
