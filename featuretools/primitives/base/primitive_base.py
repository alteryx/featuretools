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
        # getargspec fails for non user defined inits in python2
        from sys import version_info
        if version_info.major < 3:
            try:
                parameter_names, _, _, default_values = inspect.getargspec(self.__init__)
            except TypeError:
                return ""
        else:
            parameter_names, _, _, default_values = inspect.getargspec(self.__init__)

        if default_values is None:
            default_values = []

        num_kwargs = len(default_values)
        args = parameter_names[1:-num_kwargs]
        kwargs = parameter_names[num_kwargs + 1:]

        parameter_format = "{0}={1}"
        variables = self.__dict__
        arg_strings = [parameter_format.format(x, variables[x]) for x in args]

        for kwarg, default in zip(kwargs, default_values):
            val = variables[kwarg]
            # Only add string where val != default
            # Handles case where val = np.nan and default = np.nan
            if val != default and not (np.isnan(val) and np.isnan(default)):
                arg_strings.append(parameter_format.format(kwarg, val))

        return ', '.join(arg_strings)
