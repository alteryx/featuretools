import os
import json
import shutil
from featuretools.variable_types import ALL_VARIABLE_TYPES as vtypes

TYPES = ['csv', 'pickle', 'parquet']
SCHEMA = {
    'entities': ['id', 'index', 'time_index', 'variables', 'properties'],
    'relationships': {
        'parent': ['parent_entity', 'parent_variable'],
        'child': ['child_entity', 'child_variable']
    }
}
VTYPES = {v.__name__.lower(): v for v in vtypes}
_error = '"{}" is not supported'.format


def entity(e):
    '''serialize entity of entity set to data description'''
    descr = {}
    for key in SCHEMA.get('entities'):
        if key in ['id', 'index', 'time_index']:
            descr[key] = getattr(e, key)
        elif key == 'properties':
            descr[key] = dict(secondary_time_index=e.secondary_time_index)
        elif key == 'variables':
            descr[key] = [v.create_metadata_dict() for v in e.variables]
        else:
            raise ValueError(_error(key))
    return e.id, descr


def relation(r):
    '''serialize relationship of entity set to data description'''
    return {
        key: [getattr(r, attr).id for attr in attrs]
        for key, attrs in SCHEMA.get('relationships').items()
    }


def to_description(es):
    '''serialize entity set to data description'''
    return {
        'id': es.id,
        'entities': dict(map(entity, es.entities)),
        'relationships': list(map(relation, es.relationships))
    }


def save_data(e, abspath, type='csv', **params):
    '''save data of entity to disk'''
    location = os.path.join('data', e.id)
    loading_info = dict(location=location, type=type.lower())
    assert loading_info['type'] in TYPES, _error(type)
    attr = 'to_{}'.format(loading_info['type'])
    file = os.path.join(abspath, location)
    getattr(e.df, attr)(file, **params)
    return loading_info


def save(es, path, **params):
    '''save entity set to disk'''
    abspath = os.path.abspath(path)
    if os.path.exists(abspath):
        shutil.rmtree(abspath)
    for p in [abspath, os.path.join(abspath, 'data')]:
        os.makedirs(p)
    d = to_description(es)
    for e in es.entities:
        d['entities'][e.id]['loading_info'] = save_data(e, abspath, **params)
    with open(os.path.join(abspath, 'data_description.json'), 'w') as f:
        json.dump(d, f)


# ----------------------------------------
def from_description(es):
    '''serialize entity set from data description'''
    pass
