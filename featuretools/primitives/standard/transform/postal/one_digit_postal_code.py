import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Categorical, PostalCode

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class OneDigitPostalCode(TransformPrimitive):
    """Returns the one digit prefix from a given postal code.

    Description:
        For a list of postal codes, return the one digit prefix for a given postal code.

    Examples:
        >>> one_digit_postal_code = OneDigitPostalCode().get_function()
        >>> one_digit_postal_code(['92432', '34514']).tolist()
        [9, 3]
    """

    name = "one_digit_postal_code"
    input_types = [ColumnSchema(logical_type=PostalCode)]
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={"numeric"})
    description_template = "The one digit postal code prefix of {}"

    def get_function(self):
        def one_digit_postal_code(postal_code):
            return pd.Series(pc[0] for pc in postal_code)

        return one_digit_postal_code
