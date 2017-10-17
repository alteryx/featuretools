import sys
from builtins import object

from pympler.asizeof import asizeof as getsize  # noqa
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
