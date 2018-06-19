# -*- coding: utf-8 -*-

import json
import os
import shutil
import sys
import uuid
from tempfile import mkdtemp

import numpy as np
import pandas as pd
from pandas.io.pickle import (to_pickle as pd_to_pickle,
                              read_pickle as pd_read_pickle)

import featuretools.variable_types.variable as vtypes


def read_parquet(path):
    return read_entityset(path)


def read_pickle(path):
    return read_entityset(path)


def read_entityset(path):
    from featuretools.entityset.entityset import EntitySet
    entityset_path = os.path.abspath(os.path.expanduser(path))
    with open(os.path.join(entityset_path, 'metadata.json')) as f:
        metadata = json.load(f)
    return EntitySet.from_metadata(metadata, root=entityset_path,
                                   load_data=True)


def load_entity_data(metadata, dummy=True, root=None):
    variable_types = {}
    defaults = []
    columns = []
    variable_names = {}
    for elt in dir(vtypes):
        try:
            cls = getattr(vtypes, elt)
            if issubclass(cls, vtypes.Variable):
                variable_names[cls._dtype_repr] = cls
        except TypeError:
            pass
    for vid, vmetadata in metadata['variables'].items():
        if vmetadata['dtype_repr']:
            vtype = variable_names.get(vmetadata['dtype_repr'], vtypes.Variable)
            variable_types[vid] = vtype
            defaults.append(vtypes.DEFAULT_DTYPE_VALUES[vtype._default_pandas_dtype])
        else:
            defaults.append(vtypes.DEFAULT_DTYPE_VALUES[object])
        columns.append(vid)
    df = pd.DataFrame({c: [d] for c, d in zip(columns, defaults)})
    if not dummy:
        if metadata['data_files']['filetype'] == 'pickle':
            df = pd_read_pickle(os.path.join(root, metadata['data_filename']))
        elif metadata['data_files']['filetype'] == 'parquet':
            df = pd.read_parquet(os.path.join(root,
                                              metadata['data_files']['df_filename']))
            df.index = df[metadata['index']]
            to_join = metadata['data_files'].get('to_join', None)
            if to_join is not None:
                for cname, to_join_names in to_join.items():
                    df[cname] = df[to_join_names].apply(tuple, axis=1)
                    df.drop(to_join_names, axis=1, inplace=True)
        else:
            raise ValueError("Unknown entityset data filetype: {}".format(metadata['data_files']['filetype']))
    return df, variable_types


def write_entityset(entityset, path, to_parquet=False):
    metadata = entityset.create_metadata_dict()
    entityset_path = os.path.abspath(os.path.expanduser(path))
    try:
        os.makedirs(entityset_path)
    except OSError:
        pass

    temp_dir = mkdtemp()
    try:
        for e_id, entity in entityset.entity_dict.items():
            if to_parquet:
                metadata = _write_parquet_entity_data(temp_dir,
                                                      entity,
                                                      metadata)
            else:
                rel_filename = os.path.join(e_id, 'data.p')
                filename = os.path.join(temp_dir, rel_filename)
                os.makedirs(os.path.join(temp_dir, e_id))
                pd_to_pickle(entity.data, filename)
                metadata['entity_dict'][e_id]['data_files'] = {
                    'data_filename': rel_filename,
                    'filetype': 'pickle',
                    'size': os.stat(filename).st_size
                }

        timestamp = pd.Timestamp.now().isoformat()
        with open(os.path.join(temp_dir, 'save_time.txt'), 'w') as f:
            f.write(timestamp)
        with open(os.path.join(temp_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

        # can use a lock here if need be
        if os.path.exists(entityset_path):
            shutil.rmtree(entityset_path)
        shutil.move(temp_dir, entityset_path)
    except:
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


def _write_parquet_entity_data(root, entity, metadata):
    entity_path = os.path.join(root, entity.id)
    os.makedirs(entity_path)
    entity_size = 0
    df, to_join = _parquet_compatible(entity.df)
    data_files = {}

    rel_df_filename = os.path.join(entity.id, 'df.parq')
    data_files['df_filename'] = rel_df_filename
    df_filename = os.path.join(root, rel_df_filename)
    saved = False
    for compression in ['snappy', 'gzip', None]:
        try:
            df.to_parquet(df_filename, compression=compression)
        except (ImportError, RuntimeError):
            continue
        else:
            saved = True
            break
    if not saved:
        raise ImportError("Must install pyarrow or fastparquet to save EntitySet to parquet files. ",
                          "See https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_parquet.html")
    entity_size += os.stat(df_filename).st_size
    if entity.last_time_index:
        rel_lti_filename = os.path.join(entity.id, 'lti.parq')
        lti_filename = os.path.join(root, rel_lti_filename)
        entity.last_time_index.to_parquet(lti_filename,
                                          compression=compression)
        entity_size += os.stat(lti_filename).st_size
        data_files[u'lti_filename'] = rel_lti_filename
    rel_index_path = os.path.join(entity.id, 'indexes')
    index_path = os.path.join(root, rel_index_path)
    os.makedirs(index_path)
    data_files['indexes'] = {}
    for var_id, mapping_dict in entity.indexed_by.items():
        rel_var_path = os.path.join(rel_index_path, var_id)
        var_path = os.path.join(root, rel_var_path)
        os.makedirs(var_path)
        data_files['indexes'][var_id] = []
        for instance, index in mapping_dict.items():
            rel_var_index_filename = os.path.join(rel_var_path, '{}.parq'.format(instance))
            var_index_filename = os.path.join(root, rel_var_index_filename)
            pd.Series(index).to_frame(str(instance)).to_parquet(var_index_filename,
                                                                compression=compression)
            entity_size += os.stat(var_index_filename).st_size
            data_files['indexes'][var_id].append({'instance': instance,
                                                  'filename': rel_var_index_filename})
    data_files[u'to_join'] = to_join
    data_files[u'filetype'] = 'parquet'
    data_files[u'size'] = entity_size
    metadata['entity_dict'][entity.id][u'data_files'] = data_files
    return metadata
