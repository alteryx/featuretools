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
