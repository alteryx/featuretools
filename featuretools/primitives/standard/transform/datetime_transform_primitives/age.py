from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import AgeFractional, Datetime

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class Age(TransformPrimitive):
    """Calculates the age in years as a floating point number given a
       date of birth.

    Description:
        Age in years is computed by calculating the number of days between
        the date of birth and the reference time and dividing the result
        by 365.

    Examples:
        Determine the age of three people as of Jan 1, 2019
        >>> import pandas as pd
        >>> reference_date = pd.to_datetime("01-01-2019")
        >>> age = Age()
        >>> input_ages = [pd.to_datetime("01-01-2000"),
        ...               pd.to_datetime("05-30-1983"),
        ...               pd.to_datetime("10-17-1997")]
        >>> age(input_ages, time=reference_date).tolist()
        [19.013698630136986, 35.61643835616438, 21.221917808219178]
    """

    name = "age"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={"date_of_birth"})]
    return_type = ColumnSchema(logical_type=AgeFractional, semantic_tags={"numeric"})
    uses_calc_time = True
    compatibility = [Library.PANDAS, Library.DASK]
    description_template = "the age from {}"

    def get_function(self):
        def age(x, time=None):
            return (time - x).dt.days / 365

        return age
