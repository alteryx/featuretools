import sys

from tqdm import tqdm


def topsort(nodes, depfunc):
    queue = nodes[:]
    ordered = []
    while queue:
        next_n = queue.pop(0)
        for dep in depfunc(next_n):
            if dep in ordered:
                ordered.remove(dep)
            queue.append(dep)
        if next_n in ordered:
            ordered.remove(next_n)
        ordered.append(next_n)
    return ordered[::-1]


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


def is_string(test_value):
    """Checks for string in Python2 and Python3
       Via Stack Overflow: https://stackoverflow.com/a/22679982/9458191
    """
    try:
        python_string = basestring
    except NameError:
        python_string = str
    return isinstance(test_value, python_string)


def get_relationship_variable_id(path):
    r = path[0]
    child_link_name = r.child_variable.id
    for r in path[1:]:
        parent_link_name = child_link_name
        child_link_name = '%s.%s' % (r.parent_entity.id,
                                     parent_link_name)
    return child_link_name


def is_python_2():
    return sys.version_info.major < 3


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
    if is_string(cls_type):
        if is_python_2():
            from itertools import izip_longest as zip_longest
        else:
            from itertools import zip_longest

        if cls_type == 'entityset':
            from featuretools.entityset.serialize import SCHEMA_VERSION
            version_string = cls.get('schema_version')
        elif cls_type == 'feature':
            from featuretools.feature_base.features_serializer import SCHEMA_VERSION
            version_string = cls.features_dict['schema_version']

        current = SCHEMA_VERSION.split('.')
        saved = version_string.split('.')

        error_text_upgrade = ('Unable to load %s. The schema version of the saved '
                        '%s (%s) is greater than the latest supported (%s). '
                        'You may need to upgrade featuretools.'
                        % (cls_type, cls_type, version_string, SCHEMA_VERSION))
        for c_num, s_num in zip_longest(current, saved, fillvalue=0):
                if c_num > s_num:
                    break
                elif c_num < s_num:
                    raise RuntimeError(error_text_upgrade)

        error_text_outdated = ('Unable to load %s. The schema version '
                                'of the saved %s (%s) is no longer '
                                'supported by this version of featuretools.'
                                % (cls_type, cls_type, version_string))
        # Check if saved has older major version.
        if current[0] > saved[0]:
            raise RuntimeError(error_text_outdated)

