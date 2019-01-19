from __future__ import absolute_import

import numpy as np


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
