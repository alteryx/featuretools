import sys
from StringIO import StringIO
from tqdm import tqdm_notebook, tqdm
from numbers import Number
from collections import Set, Mapping, deque

try:  # Python 2
    zero_depth_bases = (basestring, Number, xrange, bytearray)
    iteritems = 'iteritems'
except NameError:  # Python 3
    zero_depth_bases = (str, bytes, Number, range, bytearray)
    iteritems = 'items'


def getsize(obj_0):
    """Recursively iterate to sum size of object & members.
       Credit to this stack overflow post: https://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python
    """
    def inner(obj, _seen_ids=set()):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, zero_depth_bases):
            pass  # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'):  # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size
    return inner(obj_0)


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


class RedirectStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


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
        # if failed or err_out.lower().find("widget javascript not detected") > -1:
            # display.clear_output(wait=True)
            # iterator = tqdm(**options)
        iterator = tqdm(**options)

    else:
        iterator = tqdm(**options)
    return iterator
