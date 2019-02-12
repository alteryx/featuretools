import os
import shutil

import pytest

from ..testing_utils import make_ecommerce_entityset

import featuretools as ft
from featuretools.demo import load_mock_customer
from featuretools.entityset import serialization
from featuretools.tests import integration_data


@pytest.fixture()
def entityset():
    return make_ecommerce_entityset()


def test_variable(entityset):
    for entity in entityset.entities:
        for variable in entity.variables:
            description = variable.create_data_description()
            _variable = serialization.from_variable_description(description, entity=entity)
            assert variable.__eq__(_variable)


def test_entity(entityset):
    _entityset = ft.EntitySet(entityset.id)
    for entity in entityset.metadata.entities:
        description = serialization.to_entity_description(entity)
        serialization.from_entity_description(_entityset, description)
        _entity = _entityset[description['id']]
        _entity.last_time_index = entity.last_time_index
        assert entity.__eq__(_entity, deep=True)


def test_entityset(entityset):
    description = entityset.create_data_description()
    _entityset = ft.EntitySet.from_data_description(description)
    assert entityset.metadata.__eq__(_entityset, deep=True)


def test_relationship(entityset):
    for relationship in entityset.relationships:
        description = serialization.to_relationship_description(relationship)
        relationship = serialization.from_relationship_description(entityset, description)
        assert relationship.__eq__(relationship)


def test_to_csv(entityset):
    dirname = os.path.dirname(integration_data.__file__)
    path = os.path.join(dirname, '.cache/es')
    params = dict(encoding='utf-8', engine='python')
    entityset.to_csv(path, params=params, encoding='utf-8')
    new_es = ft.read_entityset(path)
    assert entityset.__eq__(new_es, deep=True)
    shutil.rmtree(path)


def test_to_pickle(entityset):
    dirname = os.path.dirname(integration_data.__file__)
    path = os.path.join(dirname, '.cache/es')
    entityset.to_pickle(path)
    new_es = ft.read_entityset(path)
    assert entityset.__eq__(new_es, deep=True)
    shutil.rmtree(path)


def test_to_parquet(entityset):
    dirname = os.path.dirname(integration_data.__file__)
    path = os.path.join(dirname, '.cache/es')
    entityset.to_parquet(path)
    new_es = ft.read_entityset(path)
    assert entityset.__eq__(new_es, deep=True)
    shutil.rmtree(path)


def test_to_parquet_with_lti():
    entityset = load_mock_customer(return_entityset=True, random_seed=0)
    dirname = os.path.dirname(integration_data.__file__)
    path = os.path.join(dirname, '.cache/es')
    entityset.to_parquet(path)
    new_es = ft.read_entityset(path)
    assert entityset.__eq__(new_es, deep=True)
    shutil.rmtree(path)


def test_to_pickle_id_none():
    entityset = ft.EntitySet()
    dirname = os.path.dirname(integration_data.__file__)
    path = os.path.join(dirname, '.cache/es')
    entityset.to_pickle(path)
    new_es = ft.read_entityset(path)
    assert entityset.__eq__(new_es, deep=True)
    shutil.rmtree(path)
