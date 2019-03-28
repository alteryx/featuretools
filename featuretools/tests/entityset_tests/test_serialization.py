import os
import shutil

import pandas as pd
import pytest

from ...demo import load_mock_customer
from ...entityset import EntitySet, deserialize, serialize
from ...tests import integration_data
from ..testing_utils import make_ecommerce_entityset

CACHE = os.path.join(os.path.dirname(integration_data.__file__), '.cache')


@pytest.fixture()
def entityset():
    return make_ecommerce_entityset()


def test_all_variable_descriptions():
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
        _variable = deserialize.description_to_variable(description, entity=entity)
        assert variable.__eq__(_variable)


def test_variable_descriptions(entityset):
    for entity in entityset.entities:
        for variable in entity.variables:
            description = variable.to_data_description()
            _variable = deserialize.description_to_variable(description, entity=entity)
            assert variable.__eq__(_variable)


def test_entity_descriptions(entityset):
    _entityset = EntitySet(entityset.id)
    for entity in entityset.metadata.entities:
        description = serialize.entity_to_description(entity)
        deserialize.description_to_entity(description, _entityset)
        _entity = _entityset[description['id']]
        _entity.last_time_index = entity.last_time_index
        assert entity.__eq__(_entity, deep=True)


def test_relationship_descriptions(entityset):
    for relationship in entityset.relationships:
        description = serialize.relationship_to_description(relationship)
        _relationship = deserialize.description_to_relationship(description, entityset)
        assert relationship.__eq__(_relationship)


def test_entityset_description(entityset):
    description = serialize.entityset_to_description(entityset)
    _entityset = deserialize.description_to_entityset(description)
    assert entityset.metadata.__eq__(_entityset, deep=True)


def test_invalid_formats(entityset):
    path = os.path.join(CACHE, 'es')
    error_text = 'must be one of the following formats: {}'
    error_text = error_text.format(', '.join(serialize.FORMATS))
    with pytest.raises(ValueError, match=error_text):
        serialize.write_entity_data(entityset.entities[0], path=path, format='')
    with pytest.raises(ValueError, match=error_text):
        entity = {'loading_info': {'location': 'data', 'type': ''}}
        deserialize.read_entity_data(entity, path='.')


def test_empty_dataframe(entityset):
    for entity in entityset.entities:
        description = serialize.entity_to_description(entity)
        dataframe = deserialize.empty_dataframe(description)
        assert dataframe.empty


def test_to_csv(entityset):
    path = os.path.join(CACHE, 'es')
    os.makedirs(path)
    entityset.to_csv(path, encoding='utf-8', engine='python')
    new_es = deserialize.read_entityset(path)
    assert entityset.__eq__(new_es, deep=True)
    shutil.rmtree(path)


def test_to_pickle(entityset):
    path = os.path.join(CACHE, 'es')
    os.makedirs(path)
    entityset.to_pickle(path)
    new_es = deserialize.read_entityset(path)
    assert entityset.__eq__(new_es, deep=True)
    shutil.rmtree(path)


def test_to_parquet(entityset):
    path = os.path.join(CACHE, 'es')
    os.makedirs(path)
    entityset.to_parquet(path)
    new_es = deserialize.read_entityset(path)
    assert entityset.__eq__(new_es, deep=True)
    shutil.rmtree(path)


def test_to_parquet_with_lti():
    entityset = load_mock_customer(return_entityset=True, random_seed=0)
    path = os.path.join(CACHE, 'es')
    os.makedirs(path)
    entityset.to_parquet(path)
    new_es = deserialize.read_entityset(path)
    assert entityset.__eq__(new_es, deep=True)
    shutil.rmtree(path)


def test_to_pickle_id_none():
    path = os.path.join(CACHE, 'es')
    os.makedirs(path)
    entityset = EntitySet()
    entityset.to_pickle(path)
    new_es = deserialize.read_entityset(path)
    assert entityset.__eq__(new_es, deep=True)
    shutil.rmtree(path)
