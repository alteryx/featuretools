import logging
import os
import shutil
import uuid
from tempfile import mkdtemp

import numpy as np
from pandas import Timestamp, read_csv
from pandas.io.pickle import read_pickle as pd_read_pickle
from pandas.io.pickle import to_pickle as pd_to_pickle

from featuretools import variable_types as vtypes

logger = logging.getLogger('featuretools.entityset')

_datetime_types = vtypes.PandasTypes._pandas_datetimes


def to_pickle(entityset, path):
    """Save the entityset at the given path.

       Args:
           entityset (:class:`featuretools.BaseEntitySet`) : EntitySet to save
           path : pathname of a directory to save the entityset
            (includes a CSV file for each entity, as well as a metadata
            pickle file)

    """
    entityset_path = os.path.abspath(os.path.expanduser(path))
    try:
        os.makedirs(entityset_path)
    except OSError:
        pass

    entity_store_dframes = {}
    entity_store_index_bys = {}

    entity_sizes = {}

    temp_dir = mkdtemp()

    for e_id, entity_store in entityset.entity_stores.items():
        entity_path = os.path.join(temp_dir, e_id)
        filename = e_id + '.csv'
        os.mkdir(entity_path)
        df = entity_store.df.copy(deep=False)
        datatypes = {'dtype': {}, 'parse_dates': [], 'to_join': {}}
        for c in df:
            if df[c].dtype == object:
                dropped = df[c].dropna()
                if not dropped.empty and isinstance(dropped.iloc[0], tuple):
                    datatypes['to_join'][c] = []
                    # Assume all tuples are of same length
                    for i in range(len(dropped.iloc[0])):
                        new_name = str(uuid.uuid1())
                        df[new_name] = np.nan
                        df.loc[dropped.index, new_name] = dropped.apply(lambda x: x[i])
                        datatypes['to_join'][c].append(new_name)
                    del df[c]
        df.to_csv(os.path.join(entity_path, filename),
                  index=False,
                  encoding=entity_store.encoding,
                  compression='gzip')
        entity_sizes[e_id] = \
            os.stat(os.path.join(entity_path, filename)).st_size
        for column in entity_store.df.columns:
            coltype = entity_store.get_column_type(column)
            if coltype in _datetime_types:
                datatypes['parse_dates'].append(column)
            else:
                datatypes['dtype'][column] = entity_store.df[column].dtype
        pd_to_pickle(datatypes, os.path.join(entity_path, 'datatypes.p'))

        entity_store_dframes[e_id] = entity_store.df
        entity_store.df = None

        pd_to_pickle(entity_store.indexed_by, os.path.join(entity_path, 'indexed_by.p'))

        entity_store_index_bys[e_id] = entity_store.indexed_by
        entity_store.indexed_by = None

    entityset.entity_sizes = entity_sizes
    timestamp = Timestamp.now().isoformat()
    with open(os.path.join(temp_dir, 'save_time.txt'), 'w') as f:
        f.write(timestamp)
    pd_to_pickle(entityset, os.path.join(temp_dir, 'entityset.p'))

    for e_id, entity_store in entityset.entity_stores.items():
        setattr(entity_store, 'df', entity_store_dframes[e_id])
        setattr(entity_store, 'indexed_by', entity_store_index_bys[e_id])

    # can use a lock here if need be
    if os.path.exists(entityset_path):
        shutil.rmtree(entityset_path)
    shutil.move(temp_dir, entityset_path)


def read_pickle(path):
    """
    Read an EntitySet from disk. Assumes EntitySet has been saved using
    :meth:`.to_pickle()`.

    Args:
        path (str): Path of directory where entityset is stored
    """
    entityset_path = os.path.abspath(os.path.expanduser(path))
    entityset = pd_read_pickle(os.path.join(entityset_path, 'entityset.p'))
    for e_id, entity_store in entityset.entity_stores.items():
        entity_path = os.path.join(entityset_path, e_id)

        datatypes = pd_read_pickle(os.path.join(entity_path, 'datatypes.p'))
        entity_store_path = os.path.join(entity_path,
                                         e_id + '.csv')
        entity_store_df = read_csv(entity_store_path,
                                   index_col=False,
                                   dtype=datatypes['dtype'],
                                   parse_dates=datatypes['parse_dates'],
                                   encoding=entity_store.encoding,
                                   compression='gzip')
        for c, to_join in datatypes['to_join'].items():
            entity_store_df[c] = entity_store_df[to_join].apply(tuple, axis=1)
            entity_store_df.drop(to_join, axis=1, inplace=True)

        entity_store_df.set_index(entity_store_df[entity_store.index],
                                  drop=False, inplace=True)
        setattr(entity_store, 'df', entity_store_df)
        indexed_by = pd_read_pickle(os.path.join(entity_path, 'indexed_by.p'))
        setattr(entity_store, 'indexed_by', indexed_by)

    assert entityset is not None, "EntitySet not loaded properly"

    return entityset
