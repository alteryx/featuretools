import logging
import os
import shutil
from tempfile import mkdtemp

from pandas import Timestamp
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

    entity_sizes = {}
    temp_dir = mkdtemp()

    for e_id, entity in entityset.entity_dict.items():
        entity_path = os.path.join(temp_dir, e_id)
        os.makedirs(entity_path)
        data = entity.data
        filename = os.path.join(entity_path, 'data.p')
        pd_to_pickle(data, filename)
        entity_sizes[e_id] = os.stat(filename).st_size

    entityset.entity_sizes = entity_sizes
    metadata = entityset.metadata
    timestamp = Timestamp.now().isoformat()
    with open(os.path.join(temp_dir, 'save_time.txt'), 'w') as f:
        f.write(timestamp)
    pd_to_pickle(metadata, os.path.join(temp_dir, 'metadata.p'))

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
    entityset = pd_read_pickle(os.path.join(entityset_path, 'metadata.p'))
    for e_id, entity in entityset.entity_dict.items():
        entity_path = os.path.join(entityset_path, e_id)
        data = pd_read_pickle(os.path.join(entity_path, 'data.p'))
        entity.update_data(data=data,
                           already_sorted=True,
                           reindex=False,
                           recalculate_last_time_indexes=False)
    assert entityset is not None, "EntitySet not loaded properly"
    return entityset
