from __future__ import absolute_import

import os
import types
from sys import version_info

import numpy as np
import pandas as pd

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
        if not isinstance(self.__init__, types.MethodType):  # __init__ must be defined
            return ''

        v2 = version_info.major == 2
        module = __import__('funcsigs' if v2 else 'inspect')
        args = module.signature(self.__class__).parameters.values()

        def valid(arg):
            error = '"{}" must be attribute of {}'
            assert hasattr(self, arg.name), error.format(arg.name, self.__class__.__name__)
            is_positional_or_keyword = arg.kind == arg.POSITIONAL_OR_KEYWORD
            not_default = arg.default != getattr(self, arg.name)
            return is_positional_or_keyword and not_default

        string = {}
        for arg in args:
            if not valid(arg):
                continue
            string[arg.name] = str(getattr(self, arg.name))
        if len(string) == 0:
            return ''
        string = ', ' + ', '.join(map('='.join, string.items()))
        return string
