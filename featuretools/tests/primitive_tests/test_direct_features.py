import pytest

from ..testing_utils import make_ecommerce_entityset

from featuretools.computational_backends import PandasBackend
from featuretools.primitives.base import DirectFeature


@pytest.fixture(scope='module')
def es():
    return make_ecommerce_entityset()


def test_direct_from_identity(es):
    device = es['sessions']['device_type']
    d = DirectFeature(base_feature=device, child_entity=es['log'])

    assert d.variable == device

    pandas_backend = PandasBackend(es, [d])
    df = pandas_backend.calculate_all_features(instance_ids=[0, 5],
                                               time_last=None)
    v = df[d.get_name()].tolist()
    assert v == [0, 1]


def test_direct_from_variable(es):
    # should be same behavior as test_direct_from_identity
    device = es['sessions']['device_type']
    d = DirectFeature(base_feature=device,
                      child_entity=es['log'])

    assert d.variable == device

    pandas_backend = PandasBackend(es, [d])
    df = pandas_backend.calculate_all_features(instance_ids=[0, 5],
                                               time_last=None)
    v = df[d.get_name()].tolist()
    assert v == [0, 1]


def test_direct_rename(es):
    # should be same behavior as test_direct_from_identity
    feat = DirectFeature(base_feature=es['sessions']['device_type'],
                         child_entity=es['log'])
    copy_feat = feat.rename("session_test")
    assert feat.hash() != copy_feat.hash()
    assert feat.get_name() != copy_feat.get_name()
    assert feat.base_features[0].generate_name() == copy_feat.base_features[0].generate_name()
    assert feat.entity == copy_feat.entity
