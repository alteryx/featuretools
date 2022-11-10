import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Categorical, PostalCode

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class TwoDigitPostalCode(TransformPrimitive):
    """Returns the one digit prefix from a given postal code.

    Description:
        For a list of postal codes, return the one digit prefix for a given postal code.

    Examples:
        >>> two_digit_postal_code = TwoDigitPostalCode()
        >>> two_digit_postal_code(['92432', '34514']).tolist()
        ['92', '34']
    """

    name = "two_digit_postal_code"
    input_types = [ColumnSchema(logical_type=PostalCode)]
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={"category"})
    description_template = "The two digit postal code prefix of {}"

    def get_function(self):
        def two_digit_postal_code(postal_code):
            postal_code.apply(str)
            return pd.Series(pc[:1] for pc in postal_code)

        return two_digit_postal_code
