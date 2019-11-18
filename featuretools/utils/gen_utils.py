import json
import shutil
import sys
import warnings
from itertools import zip_longest

import s3fs
from smart_open import open
from tqdm import tqdm


def session_type():
    if 'IPython' not in sys.modules:
        # IPython hasn't been imported, definitely not
        return "python"
    from IPython import get_ipython
    # check for `kernel` attribute on the IPython instance
    if getattr(get_ipython(), 'kernel', None) is not None:
        return "kernel"
    return "ipython"


def make_tqdm_iterator(**kwargs):
    options = {
        "file": sys.stdout,
        "leave": True
    }
    options.update(kwargs)

    if session_type() == 'kernel':
        # from IPython import display
        # capture_stderr = StringIO()
        # with RedirectStdStreams(stderr=capture_stderr):
        # try:
        # iterator = tqdm_notebook(**options)
        # except:
        # failed = True
        # else:
        # failed = False
        # err_out = capture_stderr.getvalue()
        # capture_stderr.close()
        # if failed or err_out.lower().find("widget javascript not detected") \
        # >-1:
        # display.clear_output(wait=True)
        # iterator = tqdm(**options)
        iterator = tqdm(**options)

    else:
        iterator = tqdm(**options)
    return iterator


def get_relationship_variable_id(path):
    _, r = path[0]
    child_link_name = r.child_variable.id
    for _, r in path[1:]:
        parent_link_name = child_link_name
        child_link_name = '%s.%s' % (r.parent_entity.id,
                                     parent_link_name)
    return child_link_name


def find_descendents(cls):
    """
    A generator which yields all descendent classes of the given class
    (including the given class)

    Args:
        cls (Class): the class to find descendents of
    """
    yield cls
    for sub in cls.__subclasses__():
        for c in find_descendents(sub):
            yield c


def check_schema_version(cls, cls_type):
    if isinstance(cls_type, str):
        if cls_type == 'entityset':
            from featuretools.entityset.serialize import SCHEMA_VERSION
            version_string = cls.get('schema_version')
        elif cls_type == 'features':
            from featuretools.feature_base.features_serializer import SCHEMA_VERSION
            version_string = cls.features_dict['schema_version']

        current = SCHEMA_VERSION.split('.')
        saved = version_string.split('.')

        warning_text_upgrade = ('The schema version of the saved %s'
                                '(%s) is greater than the latest supported (%s). '
                                'You may need to upgrade featuretools. Attempting to load %s ...'
                                % (cls_type, version_string, SCHEMA_VERSION, cls_type))
        for c_num, s_num in zip_longest(current, saved, fillvalue=0):
            if c_num > s_num:
                break
            elif c_num < s_num:
                warnings.warn(warning_text_upgrade)
                break

        warning_text_outdated = ('The schema version of the saved %s'
                                 '(%s) is no longer supported by this version '
                                 'of featuretools. Attempting to load %s ...'
                                 % (cls_type, version_string, cls_type))
        # Check if saved has older major version.
        if current[0] > saved[0]:
            warnings.warn(warning_text_outdated)


def use_smartopen_es(file_path, path, transport_params=None, read=True):
    if read:
        with open(path, "rb", transport_params=transport_params) as fin:
            with open(file_path, 'wb') as fout:
                shutil.copyfileobj(fin, fout)
    else:
        with open(file_path, 'rb') as fin:
            with open(path, 'wb', transport_params=transport_params) as fout:
                shutil.copyfileobj(fin, fout)


def use_s3fs_es(file_path, path, read=True):
    s3 = s3fs.S3FileSystem(anon=True)
    if read:
        s3.get(path, file_path)
    else:
        s3.put(file_path, path)


def use_smartopen_features(path, features_dict=None, transport_params=None, read=True):
    if read:
        with open(path, 'r', encoding='utf-8', transport_params=transport_params) as f:
            features_dict = json.load(f)
            return features_dict
    else:
        with open(path, "w", transport_params=transport_params) as f:
            json.dump(features_dict, f)


def use_s3fs_features(file_path, features_dict=None, read=True):
    s3 = s3fs.S3FileSystem(anon=True)
    if read:
        with s3.open(file_path, "r", encoding='utf-8') as f:
            features_dict = json.load(f)
            return features_dict
    else:
        with s3.open(file_path, "w") as f:
            features = json.dumps(features_dict, ensure_ascii=False)
            f.write(features)
