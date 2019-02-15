import json
import os

import pandas as pd

from .relationship import Relationship
from .serialize import FORMATS, VARIABLE_TYPES


def description_to_variable(description, entity=None):
    '''Deserialize variable from variable description.

    Args:
        description (dict) : Description of :class:`.Variable`.
        entity (Entity) : Instance of :class:`.Entity` to add :class:`.Variable`. If entity is None, :class:`.Variable` will not be instantiated.

    Returns:
        variable (Variable) : Returns :class:`.Variable`.
    '''
    is_type_string = isinstance(description['type'], str)
    type = description['type'] if is_type_string else description['type'].pop('value')
    variable = VARIABLE_TYPES.get(type, VARIABLE_TYPES.get('None'))
    if entity is not None:
        kwargs = {} if is_type_string else description['type']
        variable = variable(description['id'], entity, **kwargs)
        variable.interesting_values = description['properties']['interesting_values']
    return variable


def description_to_entity(description, entityset, path=None):
    '''Deserialize entity from entity description and add to entityset.

    Args:
        description (dict) : Description of :class:`.Entity`.
        entityset (EntitySet) : Instance of :class:`.EntitySet` to add :class:`.Entity`.
        path (str) : Root directory to serialized entityset.
    '''
    from_disk = path is not None
    dataframe = read_entity_data(description, path=path) if from_disk else empty_dataframe(description)
    variable_types = {variable['id']: description_to_variable(variable) for variable in description['variables']}
    entityset.entity_from_dataframe(
        description['id'],
        dataframe,
        index=description.get('index'),
        time_index=description.get('time_index'),
        secondary_time_index=description['properties'].get('secondary_time_index'),
        variable_types=variable_types)


def description_to_relationship(description, entityset):
    '''Deserialize parent and child variables from relationship description.

    Args:
        description (dict) : Description of :class:`.Relationship`.
        entityset (EntitySet) : Instance of :class:`.EntitySet` containing parent and child variables.

    Returns:
        item (tuple(Variable, Variable)) : Tuple containing parent and child variables.
    '''
    entity, variable = description['parent']
    parent = entityset[entity][variable]
    entity, variable = description['child']
    child = entityset[entity][variable]
    return Relationship(parent, child)


def empty_dataframe(description):
    '''Deserialize empty dataframe from entity description.

    Args:
        description (dict) : Description of :class:`.Entity`.

    Returns:
        df (DataFrame) : Empty dataframe for entity.
    '''
    columns = [variable['id'] for variable in description['variables']]
    dtypes = description['loading_info']['properties']['dtypes']
    return pd.DataFrame(columns=columns).astype(dtypes)


def read_entity_data(description, path=None):
    '''Read description data from disk.

    Args:
        description (dict) : Description of :class:`.Entity`.

    Returns:
        df (DataFrame) : Instance of dataframe.
    '''
    file = os.path.join(path, description['loading_info']['location'])
    kwargs = description['loading_info'].get('params', {})
    if description['loading_info']['type'] == 'csv':
        params = ['compression', 'engine', 'encoding']
        subset = {key: kwargs[key] for key in params if key in kwargs}
        dataframe = pd.read_csv(file, **subset)
    elif description['loading_info']['type'] == 'parquet':
        params = ['engine', 'columns']
        subset = {key: kwargs[key] for key in params if key in kwargs}
        dataframe = pd.read_parquet(file, **subset)
    elif description['loading_info']['type'] == 'pickle':
        params = ['compression']
        subset = {key: kwargs[key] for key in params if key in kwargs}
        dataframe = pd.read_pickle(file, **subset)
    else:
        error = 'must be one of the following formats: {}'
        raise ValueError(error.format(', '.join(FORMATS)))
    dtypes = description['loading_info']['properties']['dtypes']
    return dataframe.astype(dtypes)


def read_data_description(path):
    '''Read data description from disk.

        Args:
            path (str): Location on disk to read `data_description.json`.

        Returns:
            description (dict) : Description of :class:`.EntitySet`.
    '''
    path = os.path.abspath(path)
    assert os.path.exists(path), '"{}" does not exist'.format(path)
    file = os.path.join(path, 'data_description.json')
    with open(file, 'r') as file:
        description = json.load(file)
    description['root'] = path
    return description
