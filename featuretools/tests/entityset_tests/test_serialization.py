import os
import shutil

import pandas as pd
import pytest

from ...demo import load_mock_customer
from ...entityset import EntitySet, deserialize, read_entityset, serialize
from ...tests import integration_data
from ..testing_utils import make_ecommerce_entityset


@pytest.fixture()
def entityset():
    return make_ecommerce_entityset()


def test_variable_descriptions():
    entityset = EntitySet()
    dataframe = pd.DataFrame(columns=list(serialize.VARIABLE_TYPES))
    entityset.entity_from_dataframe(
        'variable_types',
        dataframe,
        index='index',
        time_index='datetime_time_index',
        variable_types=serialize.VARIABLE_TYPES,
    )
    entity = entityset['variable_types']
    for variable in entity.variables:
        description = variable.to_data_description()
        _variable = deserialize.from_variable_description(description, entity=entity)
        assert variable.__eq__(_variable)


def test_entity_descriptions(entityset):
    _entityset = EntitySet(entityset.id)
    for entity in entityset.metadata.entities:
        description = serialize.to_entity_description(entity)
        deserialize.from_entity_description(description, _entityset)
        _entity = _entityset[description['id']]
        _entity.last_time_index = entity.last_time_index
        assert entity.__eq__(_entity, deep=True)


def test_relationship_descriptions(entityset):
    for relationship in entityset.relationships:
        description = serialize.to_relationship_description(relationship)
        relationship = deserialize.from_relationship_description(description, entityset)
        assert relationship.__eq__(relationship)


def test_data_description(entityset):
    description = entityset.to_data_description()
    _entityset = EntitySet.from_data_description(description)
    assert entityset.metadata.__eq__(_entityset, deep=True)


def test_write_data_description(entityset):
    params = {'csv': {'index': False}, 'parquet': {'compression': 'gzip'}}
    for format in serialize.FORMATS:
        kwargs = params.get(format, {})
        path = os.path.join(os.path.dirname(integration_data.__file__), '.cache/es')
        serialize.write_data_description(entityset, path=path, format=format, **kwargs)
        assert os.path.exists(path)
        shutil.rmtree(path)


def test_invalid_write_data_description(entityset):
    path = os.path.join(os.path.dirname(integration_data.__file__), '.cache/es')
    error_text = 'must be one of the following formats: .*'
    with pytest.raises(ValueError, match=error_text):
        serialize.write_data_description(entityset, path=path, format='')


def test_invalid_read_entity_data():
    error_text = 'must be one of the following formats: .*'
    with pytest.raises(ValueError, match=error_text):
        entity = {'loading_info': {'location': 'data', 'type': ''}}
        deserialize.read_entity_data(entity, path='.')


def test_empty_dataframe(entityset):
    for entity in entityset.entities:
        description = serialize.to_entity_description(entity)
        dataframe = deserialize.empty_dataframe(description)
        assert dataframe.empty


def test_to_csv(entityset):
    dirname = os.path.dirname(integration_data.__file__)
    path = os.path.join(dirname, '.cache/es')
    entityset.to_csv(path, encoding='utf-8', engine='python')
    new_es = read_entityset(path)
    assert entityset.__eq__(new_es, deep=True)
    shutil.rmtree(path)


def test_to_pickle(entityset):
    dirname = os.path.dirname(integration_data.__file__)
    path = os.path.join(dirname, '.cache/es')
    entityset.to_pickle(path)
    new_es = read_entityset(path)
    assert entityset.__eq__(new_es, deep=True)
    shutil.rmtree(path)


def test_to_parquet(entityset):
    dirname = os.path.dirname(integration_data.__file__)
    path = os.path.join(dirname, '.cache/es')
    entityset.to_parquet(path)
    new_es = read_entityset(path)
    assert entityset.__eq__(new_es, deep=True)
    shutil.rmtree(path)


def test_to_parquet_with_lti():
    entityset = load_mock_customer(return_entityset=True, random_seed=0)
    dirname = os.path.dirname(integration_data.__file__)
    path = os.path.join(dirname, '.cache/es')
    entityset.to_parquet(path)
    new_es = read_entityset(path)
    assert entityset.__eq__(new_es, deep=True)
    shutil.rmtree(path)


def test_to_pickle_id_none():
    entityset = EntitySet()
    dirname = os.path.dirname(integration_data.__file__)
    path = os.path.join(dirname, '.cache/es')
    entityset.to_pickle(path)
    new_es = read_entityset(path)
    assert entityset.__eq__(new_es, deep=True)
    shutil.rmtree(path)
