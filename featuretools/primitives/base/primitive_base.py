import os
from inspect import signature

import numpy as np
import pandas as pd

from featuretools import config
from featuretools.utils.description_utils import convert_to_nth
from featuretools.utils.gen_utils import Library


class PrimitiveBase(object):
    """Base class for all primitives."""

    #: (str): Name of the primitive
    name = None
    #: (list): woodwork.ColumnSchema types of inputs
    input_types = None
    #: (woodwork.ColumnSchema): ColumnSchema type of return
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
    # whitelist of primitives that can be in input_types
    stack_on = None
    # blacklist of primitives that can be in signature
    stack_on_exclude = None
    # determines if primitive can be in input_types for self
    stack_on_self = True
    # (bool) If True will only make one feature per unique set of base features
    commutative = False
    #: (list): Additional compatible libraries
    compatibility = [Library.PANDAS]
    #: (str, list[str]): description template of the primitive. Input column
    # descriptions are passed as positional arguments to the template. Slice
    # number (if present) in "nth" form is passed to the template via the
    # `nth_slice` keyword argument. Multi-output primitives can use a list to
    # differentiate between the base description and a slice description.
    description_template = None
    series_library = Library.PANDAS

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
        return (self.name + self.get_args_string()) < (
            other.name + other.get_args_string()
        )

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
            string = "{}={}".format(name, str(value))
            strings.append(string)

        if len(strings) == 0:
            return ""

        string = ", ".join(strings)
        string = ", " + string
        return string

    def get_arguments(self):
        values = []

        args = signature(self.__class__).parameters.items()
        for name, arg in args:
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

    def get_description(
        self,
        input_column_descriptions,
        slice_num=None,
        template_override=None,
    ):
        template = template_override or self.description_template
        if template:
            if isinstance(template, list):
                if slice_num is not None:
                    slice_index = slice_num + 1
                    if slice_index < len(template):
                        return template[slice_index].format(
                            *input_column_descriptions,
                            nth_slice=convert_to_nth(slice_index),
                        )
                    else:
                        if len(template) > 2:
                            raise IndexError("Slice out of range of template")
                        return template[1].format(
                            *input_column_descriptions,
                            nth_slice=convert_to_nth(slice_index),
                        )
                else:
                    template = template[0]
            return template.format(*input_column_descriptions)

        # generic case:
        name = self.name.upper() if self.name is not None else type(self).__name__
        if slice_num is not None:
            nth_slice = convert_to_nth(slice_num + 1)
            description = "the {} output from applying {} to {}".format(
                nth_slice,
                name,
                ", ".join(input_column_descriptions),
            )
        else:
            description = "the result of applying {} to {}".format(
                name,
                ", ".join(input_column_descriptions),
            )
        return description

    @staticmethod
    def flatten_nested_input_types(input_types):
        """Flattens nested column schema inputs into a single list."""
        if isinstance(input_types[0], list):
            input_types = [
                sub_input for input_obj in input_types for sub_input in input_obj
            ]
        return input_types
