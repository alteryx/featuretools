import os
import shutil

import pytest

from ..testing_utils import make_ecommerce_entityset

import featuretools as ft
from featuretools.demo import load_mock_customer
from featuretools.entityset import serialization as sr
from featuretools.tests import integration_data


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


def test_to_csv(entityset):
    dirname = os.path.dirname(integration_data.__file__)
    path = os.path.join(dirname, 'test')
    p = dict(encoding='utf-8', engine='python')
    entityset.to_csv(path, params=p, encoding='utf-8')
    new_es = ft.read_entityset(path)
    assert entityset.__eq__(new_es, deep=True)
    shutil.rmtree(path)


def test_to_pickle(entityset):
    dirname = os.path.dirname(integration_data.__file__)
    path = os.path.join(dirname, 'test')
    entityset.to_pickle(path)
    new_es = ft.read_entityset(path)
    assert entityset.__eq__(new_es, deep=True)
    shutil.rmtree(path)


def test_to_parquet(entityset):
    dirname = os.path.dirname(integration_data.__file__)
    path = os.path.join(dirname, 'test')
    entityset.to_parquet(path)
    new_es = ft.read_entityset(path)
    assert entityset.__eq__(new_es, deep=True)
    shutil.rmtree(path)


def test_to_parquet_with_lti():
    entityset = load_mock_customer(return_entityset=True, random_seed=0)
    dirname = os.path.dirname(integration_data.__file__)
    path = os.path.join(dirname, 'test')
    entityset.to_parquet(path)
    new_es = ft.read_entityset(path)
    assert entityset.__eq__(new_es, deep=True)
    shutil.rmtree(path)
