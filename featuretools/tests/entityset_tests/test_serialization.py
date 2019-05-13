import os
import shutil

import pandas as pd
import pytest

from featuretools.demo import load_mock_customer
from featuretools.entityset import EntitySet, deserialize, serialize
from featuretools.tests import integration_data

CACHE = os.path.join(os.path.dirname(integration_data.__file__), '.cache')


def test_all_variable_descriptions():
    es = EntitySet()
    dataframe = pd.DataFrame(columns=list(serialize.VARIABLE_TYPES))
    es.entity_from_dataframe(
        'variable_types',
        dataframe,
        index='index',
        time_index='datetime_time_index',
        variable_types=serialize.VARIABLE_TYPES,
    )
    entity = es['variable_types']
    for variable in entity.variables:
        description = variable.to_data_description()
        _variable = deserialize.description_to_variable(description, entity=entity)
        assert variable.__eq__(_variable)


def test_variable_descriptions(es):
    for entity in es.entities:
        for variable in entity.variables:
            description = variable.to_data_description()
            _variable = deserialize.description_to_variable(description, entity=entity)
            assert variable.__eq__(_variable)


def test_entity_descriptions(es):
    _es = EntitySet(es.id)
    for entity in es.metadata.entities:
        description = serialize.entity_to_description(entity)
        deserialize.description_to_entity(description, _es)
        _entity = _es[description['id']]
        _entity.last_time_index = entity.last_time_index
        assert entity.__eq__(_entity, deep=True)


def test_relationship_descriptions(es):
    for relationship in es.relationships:
        description = serialize.relationship_to_description(relationship)
        _relationship = deserialize.description_to_relationship(description, es)
        assert relationship.__eq__(_relationship)


def test_entityset_description(es):
    description = serialize.entityset_to_description(es)
    _es = deserialize.description_to_entityset(description)
    assert es.metadata.__eq__(_es, deep=True)


def test_invalid_formats(es):
    path = os.path.join(CACHE, 'es')
    error_text = 'must be one of the following formats: {}'
    error_text = error_text.format(', '.join(serialize.FORMATS))
    with pytest.raises(ValueError, match=error_text):
        serialize.write_entity_data(es.entities[0], path=path, format='')
    with pytest.raises(ValueError, match=error_text):
        entity = {'loading_info': {'location': 'data', 'type': ''}}
        deserialize.read_entity_data(entity, path='.')


def test_empty_dataframe(es):
    for entity in es.entities:
        description = serialize.entity_to_description(entity)
        dataframe = deserialize.empty_dataframe(description)
        assert dataframe.empty


def test_to_csv(es):
    path = os.path.join(CACHE, 'es')
    os.makedirs(path)
    es.to_csv(path, encoding='utf-8', engine='python')
    new_es = deserialize.read_entityset(path)
    assert es.__eq__(new_es, deep=True)
    shutil.rmtree(path)


def test_to_pickle(es):
    path = os.path.join(CACHE, 'es')
    os.makedirs(path)
    es.to_pickle(path)
    new_es = deserialize.read_entityset(path)
    assert es.__eq__(new_es, deep=True)
    shutil.rmtree(path)


def test_to_parquet(es):
    path = os.path.join(CACHE, 'es')
    os.makedirs(path)
    es.to_parquet(path)
    new_es = deserialize.read_entityset(path)
    assert es.__eq__(new_es, deep=True)
    shutil.rmtree(path)


def test_to_parquet_with_lti():
    es = load_mock_customer(return_entityset=True, random_seed=0)
    path = os.path.join(CACHE, 'es')
    os.makedirs(path)
    es.to_parquet(path)
    new_es = deserialize.read_entityset(path)
    assert es.__eq__(new_es, deep=True)
    shutil.rmtree(path)


def test_to_pickle_id_none():
    path = os.path.join(CACHE, 'es')
    os.makedirs(path)
    es = EntitySet()
    es.to_pickle(path)
    new_es = deserialize.read_entityset(path)
    assert es.__eq__(new_es, deep=True)
    shutil.rmtree(path)
