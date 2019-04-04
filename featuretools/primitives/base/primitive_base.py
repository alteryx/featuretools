from __future__ import absolute_import

import os
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
        def get_signature(primitive):
            v2 = version_info.major == 2
            module = __import__('funcsigs' if v2 else 'inspect')
            return module.signature(primitive.__class__)

        def args_to_string(primitive):
            parameters_modified = {}
            error = '"{}" must be attribute of {}'
            parameters = get_signature(primitive).parameters
            for key, parameter in parameters.items():
                assert hasattr(primitive, key), error.format(key, primitive.__class__.__name__)
                if parameter.default == getattr(primitive, key):
                    continue
                parameters_modified[key] = str(getattr(primitive, key))
            string = ', '.join(map('='.join, parameters_modified.items()))
            string = ', ' + string if len(string) else ''
            return string

        if self.__init__.__class__.__name__ == 'method-wrapper':
            return ''  # __init__ is not defined

        return args_to_string(self)
