import numpy as np
from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive
from featuretools.utils.gen_utils import Library


class NMostCommon(AggregationPrimitive):
    """Determines the `n` most common elements.

    Description:
        Given a list of values, return the `n` values
        which appear the most frequently. If there are
        fewer than `n` unique values, the output will be
        filled with `NaN`.

    Args:
        n (int): defines "n" in "n most common." Defaults
            to 3.

    Examples:
        >>> n_most_common = NMostCommon(n=2)
        >>> x = ['orange', 'apple', 'orange', 'apple', 'orange', 'grapefruit']
        >>> n_most_common(x).tolist()
        ['orange', 'apple']
    """

    name = "n_most_common"
    input_types = [ColumnSchema(semantic_tags={"category"})]
    return_type = None

    def __init__(self, n=3):
        self.n = n
        self.number_output_features = n
        self.description_template = [
            "the {} most common values of {{}}".format(n),
            "the most common value of {}",
            *["the {nth_slice} most common value of {}"] * (n - 1),
        ]

    def get_function(self, agg_type=Library.PANDAS):
        def n_most_common(x):
            # Counts of 0 remain in value_counts output if dtype is category
            # so we need to remove them
            counts = x.value_counts()
            counts = counts[counts > 0]
            array = np.array(counts.index[: self.n])
            if len(array) < self.n:
                filler = np.full(self.n - len(array), np.nan)
                array = np.append(array, filler)
            return array

        return n_most_common
