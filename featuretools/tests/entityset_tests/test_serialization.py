import pytest

from ..testing_utils import make_ecommerce_entityset

import featuretools as ft
from featuretools.entityset import serialization as sr


@pytest.fixture()
def entityset():
    return make_ecommerce_entityset()


def test_variable(entityset):
    keys = ['entity', 'interesting_values']
    for e in entityset.entities:
        for v in e.variables:
            d = v.create_data_description()
            id, var = sr.from_var_descr(d)
            _e, iv = map(d['properties'].pop, keys)
            _v = var(id, entityset[_e], **d['properties'])
            _v.interesting_values = iv
            assert v.__eq__(_v)


def test_entity(entityset):
    for e in entityset.entities:
        id, d = sr.to_entity_descr(e)
        kwargs = sr.from_entity_descr(d)
        df, lti = entityset[id].df, entityset[id].last_time_index
        _e = ft.Entity(id, df, entityset, last_time_index=lti, **kwargs)
        assert e.__eq__(_e, deep=True)


def test_entityset(entityset):
    d = entityset.create_data_description()
    _es = ft.EntitySet.from_data_description(d)
    assert entityset.metadata.__eq__(_es, deep=True)


def test_relationship(entityset):
    for r in entityset.relationships:
        d = sr.to_relation_descr(r)
        parent = entityset[d['parent'][0]][d['parent'][1]]
        child = entityset[d['child'][0]][d['child'][1]]
        _r = ft.Relationship(parent, child)
        assert r.__eq__(_r)
