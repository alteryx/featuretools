from __future__ import absolute_import

import os

import numpy as np
import pandas as pd

from .utils import signature

from featuretools import config


class PrimitiveBase(object):
    """Base class for all primitives."""
    #: (str): Name of the primitive
    name = None
    #: (list): Variable types of inputs
    input_types = None
    #: (:class:`.Variable`): variable type of return
    return_type = None
    #: Default value this feature returns if no data found. Defaults to np.nan
    default_value = np.nan
    #: (bool): True if feature needs to know what the current calculation time
    # is (provided to computational backend as "time_last")
    uses_calc_time = False
    #: (bool): If True, allow where clauses in DFS
    allow_where = False
    #: (int): Maximum number of features in the largest chain proceeding
    # downward from this feature's base features.
    max_stack_depth = None
    #: (int): Number of columns in feature matrix associated with this feature
    number_output_features = 1
    # whitelist of primitives can have this primitive in input_types
    base_of = None
    # blacklist of primitives can have this primitive in input_types
    base_of_exclude = None
    # (bool) If True will only make one feature per unique set of base features
    commutative = False

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        series_args = [pd.Series(arg) for arg in args]
        try:
            return self._method(*series_args, **kwargs)
        except AttributeError:
            self._method = self.get_function()
            return self._method(*series_args, **kwargs)

    def generate_name(self):
        raise NotImplementedError("Subclass must implement")

    def get_function(self):
        raise NotImplementedError("Subclass must implement")

    def get_filepath(self, filename):
        return os.path.join(config.get("primitive_data_folder"), filename)

    def get_args_string(self):
        strings = []
        args = signature(self.__class__).parameters.items()
        for name, arg in args:
            # assert that arg is attribute of primitive
            error = '"{}" must be attribute of {}'
            assert hasattr(self, name), error.format(name, self.__class__.__name__)

            # skip if not a standard argument (e.g. excluding *args and **kwargs)
            if arg.kind != arg.POSITIONAL_OR_KEYWORD:
                continue

            value = getattr(self, name)
            # check if args are the same type
            if isinstance(value, type(arg.default)):
                # skip if default value
                if arg.default == value:
                    continue

            # format arg to string
            string = '{}={}'.format(name, str(value))
            strings.append(string)

        if len(strings) == 0:
            return ''

        string = ', '.join(strings)
        string = ', ' + string
        return string
