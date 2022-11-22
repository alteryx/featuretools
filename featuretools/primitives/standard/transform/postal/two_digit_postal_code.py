import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Categorical, PostalCode

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class TwoDigitPostalCode(TransformPrimitive):
    """Returns the two digit prefix of a given postal code.

    Description:
        Given a list of postal codes, returns the two digit prefix for each postal code.

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
        def two_digit_postal_code(postal_codes):
            def transform_postal_code(pc):
                return str(pc)[:2] if pd.notna(pc) else pd.NA

            return postal_codes.apply(transform_postal_code)

        return two_digit_postal_code
