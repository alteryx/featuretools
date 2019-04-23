from featuretools.primitives.base import make_agg_primitive
from featuretools.variable_types import Numeric

CustomMean = make_agg_primitive(lambda x: sum(x) / len(x),
                                name="CustomMean",
                                input_types=[Numeric],
                                return_type=Numeric)
