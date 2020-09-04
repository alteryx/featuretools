import os

import numpy as np
import pandas as pd

from featuretools import config
from featuretools.primitives.base.utils import signature
from featuretools.utils.gen_utils import Library


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
    #: (list): Additional compatible libraries
    compatibility = [Library.PANDAS]

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        series_args = [pd.Series(arg) for arg in args]
        try:
            return self._method(*series_args, **kwargs)
        except AttributeError:
            self._method = self.get_function()
            return self._method(*series_args, **kwargs)

    def __lt__(self, other):
        return (self.name + self.get_args_string()) < (other.name + other.get_args_string())

    def generate_name(self):
        raise NotImplementedError("Subclass must implement")

    def generate_names(self):
        raise NotImplementedError("Subclass must implement")

    def get_function(self):
        raise NotImplementedError("Subclass must implement")

    def get_filepath(self, filename):
        return os.path.join(config.get("primitive_data_folder"), filename)

    def get_args_string(self):
        strings = []
        for name, value in self.get_arguments():
            # format arg to string
            string = '{}={}'.format(name, str(value))
            strings.append(string)

        if len(strings) == 0:
            return ''

        string = ', '.join(strings)
        string = ', ' + string
        return string

    def get_arguments(self):
        values = []

        args = signature(self.__class__).parameters.items()
        for name, arg in args:
            # skip if not a standard argument (e.g. excluding *args and **kwargs)
            if arg.kind != arg.POSITIONAL_OR_KEYWORD:
                continue

            # assert that arg is attribute of primitive
            error = '"{}" must be attribute of {}'
            assert hasattr(self, name), error.format(name, self.__class__.__name__)

            value = getattr(self, name)
            # check if args are the same type
            if isinstance(value, type(arg.default)):
                # skip if default value
                if arg.default == value:
                    continue

            values.append((name, value))

        return values
