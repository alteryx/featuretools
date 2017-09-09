import pytest
from featuretools.primitives import Count, Mode, Feature, DirectFeature, Compare
from featuretools import variable_types
from ..testing_utils import make_ecommerce_entityset
from featuretools.synthesis import dfs_filters as filt


@pytest.fixture(scope='module')
def es():
    return make_ecommerce_entityset()


@pytest.fixture
def session_id_feat(es):
    return Feature(es['sessions']['id'])


@pytest.fixture
def product_id_feat(es):
    return Feature(es['log']['product_id'])


@pytest.fixture
def datetime_feat(es):
    return Feature(es['log']['datetime'])


def test_limit_mode_uniques(es, session_id_feat, product_id_feat, datetime_feat):
    mode_feat = Mode(product_id_feat,
                     parent_entity=es['sessions'])

    mode_filter = filt.LimitModeUniques()

    assert mode_filter.is_valid(feature=mode_feat,
                                entity=es['sessions'],
                                target_entity_id='customers')

    # percent_unique is 6/15
    mode_filter = filt.LimitModeUniques(threshold=.3)

    assert not mode_filter.is_valid(feature=mode_feat,
                                    entity=es['sessions'],
                                    target_entity_id='customers')