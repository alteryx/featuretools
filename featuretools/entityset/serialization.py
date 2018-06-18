# -*- coding: utf-8 -*-

import json
import os
import sys
import uuid

import numpy as np
import pandas as pd


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
