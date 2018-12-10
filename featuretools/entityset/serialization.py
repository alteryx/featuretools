# -*- coding: utf-8 -*-

import json
import os
import shutil
import sys
import uuid
from tempfile import mkdtemp

import numpy as np
import pandas as pd
from pandas.io.pickle import read_pickle as pd_read_pickle
from pandas.io.pickle import to_pickle as pd_to_pickle


def read_parquet(path, load_data=True):
    '''Load an EntitySet from a path on disk, assuming
    the EntitySet was saved in the parquet format.
    '''
    return read_entityset(path, load_data=load_data)


def read_pickle(path, load_data=True):
    '''Load an EntitySet from a path on disk, assuming
    the EntitySet was saved in the pickle format.
    '''
    return read_entityset(path, load_data=load_data)


def read_entityset(path, load_data=True):
    from featuretools.entityset.entityset import EntitySet
    data_root = os.path.abspath(os.path.expanduser(path))
    with open(os.path.join(data_root, 'metadata.json')) as f:
        metadata = json.load(f)
    if not load_data:
        data_root = None
    return EntitySet.from_metadata(metadata, data_root=data_root)


def load_entity_data(metadata, root):
    '''Load an entity's data from disk.'''
    if metadata['data_files']['filetype'] == 'pickle':
        data = pd_read_pickle(os.path.join(root, metadata['data_files']['data_filename']))
        df = data['df']
    elif metadata['data_files']['filetype'] == 'parquet':
        df = pd.read_parquet(os.path.join(root,
                                          metadata['data_files']['df_filename']),
                             engine=metadata['data_files']['engine'])
        df.index = df[metadata['index']]
        to_join = metadata['data_files'].get('to_join', None)
        if to_join is not None:
            for cname, to_join_names in to_join.items():
                df[cname] = df[to_join_names].apply(tuple, axis=1)
                df.drop(to_join_names, axis=1, inplace=True)
    else:
        raise ValueError("Unknown entityset data filetype: {}".format(metadata['data_files']['filetype']))
    return df


def write_entityset(entityset, path, serialization_method='pickle',
                    engine='auto', compression='gzip'):
    '''Write entityset to disk, location specified by `path`.

        Args:
            * entityset: entityset to write to disk
            * path (str): location on disk to write to (will be created as a directory)
            * serialization_method (str, optional): Possible serialization methods are:
              - 'pickle'
              - 'parquet'
            * engine (str, optional): parquet serialization engine to be passed to underlying pd.DataFrame.to_parquet() call
            * compression (str, optional): type of compression to be used in parquet serialization,
                  passed to underlying pd.DataFrame.to_parquet() call
    '''
    metadata = entityset.create_metadata_dict()
    entityset_path = os.path.abspath(os.path.expanduser(path))
    try:
        os.makedirs(entityset_path)
    except OSError:
        pass

    temp_dir = mkdtemp()
    try:
        for e_id, entity in entityset.entity_dict.items():
            if serialization_method == 'parquet':
                metadata = _write_parquet_entity_data(temp_dir,
                                                      entity,
                                                      metadata,
                                                      engine=engine,
                                                      compression=compression)
            elif serialization_method == 'pickle':
                metadata = _write_pickle_entity_data(temp_dir,
                                                     entity,
                                                     metadata)
            else:
                raise ValueError("unknown serialization_method {}, ".format(serialization_method),
                                 "available methods are 'parquet' and 'pickle'")

        timestamp = pd.Timestamp.now().isoformat()
        with open(os.path.join(temp_dir, 'save_time.txt'), 'w') as f:
            f.write(timestamp)
        with open(os.path.join(temp_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

        # can use a lock here if need be
        if os.path.exists(entityset_path):
            shutil.rmtree(entityset_path)
        shutil.move(temp_dir, entityset_path)
    except:  # noqa
        # make sure to clean up
        shutil.rmtree(temp_dir)
        raise


def _parquet_compatible(df):
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
    if sys.version_info <= (3, 0):
        from past.builtins import unicode
        df.columns = [unicode(c) for c in df.columns]
    return df, to_join


def _write_pickle_entity_data(root, entity, metadata):
    rel_filename = os.path.join(entity.id, 'data.p')
    filename = os.path.join(root, rel_filename)
    os.makedirs(os.path.join(root, entity.id))
    pd_to_pickle(entity.data, filename)
    metadata['entity_dict'][entity.id]['data_files'] = {
        'data_filename': rel_filename,
        'filetype': 'pickle',
        'size': os.stat(filename).st_size
    }
    return metadata


def _write_parquet_entity_data(root, entity, metadata,
                               engine='auto', compression='gzip'):
    '''Write an Entity's data to the binary parquet format, using pd.DataFrame.to_parquet.

    You can choose different parquet backends, and have the option of compression.
    See the Pandas [documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_parquet.html)
    for more details on engines and compression types.
    '''
    entity_path = os.path.join(root, entity.id)
    os.makedirs(entity_path)
    entity_size = 0
    df, to_join = _parquet_compatible(entity.df)
    data_files = {}

    rel_df_filename = os.path.join(entity.id, 'df.parq')
    data_files['df_filename'] = rel_df_filename
    df_filename = os.path.join(root, rel_df_filename)
    df.to_parquet(df_filename, engine=engine, compression=compression)

    entity_size += os.stat(df_filename).st_size

    data_files[u'to_join'] = to_join
    data_files[u'filetype'] = 'parquet'
    data_files[u'engine'] = engine
    data_files[u'size'] = entity_size
    metadata['entity_dict'][entity.id][u'data_files'] = data_files
    return metadata
