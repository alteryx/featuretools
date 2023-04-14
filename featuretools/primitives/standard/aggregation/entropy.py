from scipy import stats
from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive
from featuretools.utils.gen_utils import Library


class Entropy(AggregationPrimitive):
    """Calculates the entropy for a categorical column

    Description:
        Given a list of observations from a categorical
        column return the entropy of the distribution.
        NaN values can be treated as a category or
        dropped.

    Args:
        dropna (bool): Whether to consider NaN values as a separate category
            Defaults to False.
        base (float): The logarithmic base to use
            Defaults to e (natural logarithm)

    Examples:
        >>> pd_entropy = Entropy()
        >>> pd_entropy([1, 2, 3, 4])
        1.3862943611198906
    """

    name = "entropy"
    input_types = [ColumnSchema(semantic_tags={"category"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    stack_on_self = False
    description_template = "the entropy of {}"

    def __init__(self, dropna=False, base=None):
        self.dropna = dropna
        self.base = base

    def get_function(self, agg_type=Library.PANDAS):
        def pd_entropy(s):
            distribution = s.value_counts(normalize=True, dropna=self.dropna)
            if distribution.dtype == "Float64":
                distribution = distribution.astype("float64")
            return stats.entropy(distribution.to_numpy(), base=self.base)

        return pd_entropy
