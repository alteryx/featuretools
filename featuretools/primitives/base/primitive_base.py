from __future__ import absolute_import

import inspect
import os

import numpy as np

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

    def generate_name(self):
        raise NotImplementedError("Subclass must implement")

    def get_function(self):
        raise NotImplementedError("Subclass must implement")

    def get_filepath(self, filename):
        return os.path.join(config.get("primitive_data_folder"), filename)

    def get_args_string(self):
        if not callable(self.__init__):
            return ""

        temp = inspect.getargspec(self.__init__)
        arguments = temp[0]
        defaults = temp[-1]
        if defaults is None:
            defaults = []

        args = arguments[1:-len(defaults)]
        kwargs = arguments[len(defaults) + 1:]
        controllable_format = "{0}={1},"
        controllable_name = ""

        variables = self.__dict__
        for arg in args:
            val = variables[arg]
            controllable_name += controllable_format.format(arg, val)

        for kwarg, default in zip(kwargs, defaults):
            val = variables[kwarg]
            if val != default and (not np.isnan(val) or not np.isnan(default)):
                controllable_name += controllable_format.format(kwarg, val)

        return controllable_name[:-1]
