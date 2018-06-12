import json
import logging
import os
import shutil
import uuid
import warnings
from tempfile import mkdtemp

import numpy as np
import pandas as pd
from pandas import Timestamp
from pandas.io.pickle import read_pickle as pd_read_pickle
from pandas.io.pickle import to_pickle as pd_to_pickle

from featuretools import variable_types as vtypes

logger = logging.getLogger('featuretools.entityset')

_datetime_types = vtypes.PandasTypes._pandas_datetimes


def parquet_compatible(df):
    df = df.reset_index(drop=True)
    to_join = {}
    if not df.empty:
        for c in df:
            if df[c].dtype == object:
                dropped = df[c].dropna()
                if not dropped.empty and isinstance(dropped.iloc[0], tuple):
                    to_join[c] = []
                    # Assume all tuples are of same length
                    for i in range(len(dropped.iloc[0])):
                        new_name = str(uuid.uuid1())
                        df[new_name] = np.nan
                        df.loc[dropped.index, new_name] = dropped.apply(lambda x: x[i])
                        to_join[c].append(new_name)
                    del df[c]
    return df, to_join


def write_parquet_entity_data(entity_path, entity):
    try:
        from fastparquet import write
    except ImportError:
        raise ImportError("Must install fastparquet to save EntitySet to parquet files. See https://github.com/dask/fastparquet")
    entity_size = 0
    df, to_join = parquet_compatible(entity.df)
    df_filename = os.path.join(entity_path, 'df.parq')
    write(df_filename, df)
    entity_size += os.stat(df_filename).st_size
    if entity.last_time_index:
        lti_filename = os.path.join(entity_path, 'lti.parq')
        write(lti_filename, entity.last_time_index)
        entity_size += os.stat(lti_filename).st_size
    index_path = os.path.join(entity_path, 'indexes')
    os.makedirs(index_path)
    for var_id, mapping_dict in entity.indexed_by.items():
        var_path = os.path.join(index_path, var_id)
        os.makedirs(var_path)
        for instance, index in mapping_dict.items():
            var_index_filename = os.path.join(var_path, '{}.parq'.format(instance))
            series_name = "is_str"
            if isinstance(instance, int):
                series_name = "is_int"
            write(var_index_filename, pd.Series(index).to_frame(series_name))
            entity_size += os.stat(var_index_filename).st_size
    return entity_size, to_join


def serialize(entityset, path, to_parquet=False):
    """Save the entityset at the given path.

    Args:
        entityset (:class:`featuretools.BaseEntitySet`) : EntitySet to save
        path (str): pathname of a directory to save the entityset
             (includes files for each entity's data, as well as a metadata
              json file)
        to_parquet (bool): if True, write parquet files instead of Python pickle files for the data
    """

    entityset_path = os.path.abspath(os.path.expanduser(path))
    try:
        os.makedirs(entityset_path)
    except OSError:
        pass

    entity_sizes = {}
    temp_dir = mkdtemp()
    new_metadata = {}
    try:
        for e_id, entity in entityset.entity_dict.items():
            entity_path = os.path.join(temp_dir, e_id)
            os.makedirs(entity_path)
            if to_parquet:
                entity_size, to_join = write_parquet_entity_data(entity_path, entity)
                new_metadata[e_id] = {'to_join': to_join}
                entity_sizes[e_id] = entity_size
            else:
                filename = os.path.join(entity_path, 'data.p')
                pd_to_pickle(entity.data, filename)
                entity_sizes[e_id] = os.stat(filename).st_size

        entityset.entity_sizes = entity_sizes
        timestamp = Timestamp.now().isoformat()
        with open(os.path.join(temp_dir, 'save_time.txt'), 'w') as f:
            f.write(timestamp)
        json_dict = entityset.create_metadata_json()
        for eid, m in new_metadata.items():
            if m:
                json_dict['entity_dict'][eid].update(m)
        with open(os.path.join(temp_dir, 'metadata.json'), 'w') as f:
            json.dump(json_dict, f)

        # can use a lock here if need be
        if os.path.exists(entityset_path):
            shutil.rmtree(entityset_path)
        shutil.move(temp_dir, entityset_path)
    except:
        # make sure to clean up
        shutil.rmtree(temp_dir)
        raise


def read_pickle(path):
    """Read an EntitySet from disk. Assumes EntitySet has been saved using
    :meth:`.to_parquet()` or :meth:`.to_pickle()`.

    Args:
        path (str): Path of directory where entityset is stored
    """
    return deserialize(path)


def read_parquet(path):
    """Read an EntitySet from disk. Assumes EntitySet has been saved using
    :meth:`.to_parquet()` or :meth:`.to_pickle()`.

    Args:
        path (str): Path of directory where entityset is stored
    """
    return deserialize(path)


def deserialize(path):
    """Read an EntitySet from disk. Assumes EntitySet has been saved using
    :meth:`.to_parquet()` or :meth:`.to_pickle()`.

    Args:
        path (str): Path of directory where entityset is stored
    """
    from featuretools import EntitySet
    entityset_path = os.path.abspath(os.path.expanduser(path))
    with open(os.path.join(entityset_path, 'metadata.json')) as f:
        metadata = json.load(f)
    entityset = EntitySet.from_metadata(metadata)

    for e_id, entity in entityset.entity_dict.items():
        entity_path = os.path.join(entityset_path, e_id)
        read_pickle = False
        if 'df.parq' in os.listdir(entity_path):
            try:
                from fastparquet import ParquetFile
            except ImportError:
                if 'data.p' in os.listdir(entity_path):
                    warnings.warn("Found both parquet file and pickle file in {}, reading pickle file".format(entity_path))
                    read_pickle = True
                else:
                    raise ImportError("Must install fastparquet to save EntitySet to parquet files. See https://github.com/dask/fastparquet")
        else:
            read_pickle = True
        if read_pickle:
            if 'data.p' not in os.listdir(entity_path):
                raise OSError("Could not find entity data file in {}".format(entity_path))
            data = pd_read_pickle(os.path.join(entity_path, 'data.p'))
        else:
            df_filename = os.path.join(entity_path, 'df.parq')
            pf = ParquetFile(df_filename)
            df = pf.to_pandas()
            df.index = df[entity.index]
            if getattr(entity, 'to_join', None) is not None:
                for cname, to_join_names in entity.to_join.items():
                    df[cname] = df[to_join_names].apply(tuple, axis=1)
                    df.drop(to_join_names, axis=1, inplace=True)
            df = df[[v.id for v in entity.variables]]
            lti_filename = os.path.join(entity_path, 'lti.parq')
            lti = None
            if os.path.exists(lti_filename):
                pf = ParquetFile(lti_filename)
                lti = pf.to_pandas()
            index_path = os.path.join(entity_path, 'indexes')
            indexed_by = {}
            for var_id in os.listdir(index_path):
                full_var_path = os.path.join(index_path, var_id)
                indexed_by[var_id] = {}
                for basename in os.listdir(full_var_path):
                    filename = os.path.join(full_var_path, basename)
                    instance = basename.split('.parq')[0]
                    pf = ParquetFile(filename)
                    instance_df = pf.to_pandas()
                    series = instance_df.iloc[:, 0]
                    if series.name == "is_int":
                        instance = int(instance)
                    indexed_by[var_id][instance] = series.values

            data = {'df': df,
                    'last_time_index': lti,
                    'indexed_by': indexed_by}
        # TODO: can do checks against metadata
        entity.update_data(data=data,
                           already_sorted=True,
                           reindex=False,
                           recalculate_last_time_indexes=False)
    assert entityset is not None, "EntitySet not loaded properly"
    return entityset
