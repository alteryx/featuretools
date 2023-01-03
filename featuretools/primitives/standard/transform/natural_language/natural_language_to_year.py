import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import NaturalLanguage, Ordinal

from featuretools.primitives.base import TransformPrimitive


class NaturalLanguageToYear(TransformPrimitive):
    """Extracts the year from a string

    Description:
        If a year is present in a string, extrac the year. This
        will only match years between 1800 and 2199. Years will not
        be extracted if immediately preceeded or followed by another
        number or letter. If there are multiple years present in
        a string, only the first year will be returned.

    Examples:
        >>> natural_language_to_year = NaturalLanguageToYear()
        >>> array = pd.Series(["The year was 1887.",
        ...                    "This string has no year",
        ...                    "Toy Story (1995)",
        ...                    "12451997abc"])
        >>> natural_language_to_year(array).tolist()
        ['1887', nan, '1995', nan]

    """

    name = "natural_language_to_year"
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(
        Ordinal(order=tuple(map(str, range(1, 3000)))),
        semantic_tags={"category"},
    )

    def get_function(self):
        def natural_language_to_year(x):
            pattern = r"((?<!\w)(?:18|19|20|21)\d{2})(?!\w)"
            df = pd.DataFrame({"text": x})
            df["years"] = df["text"].str.extract(pattern)
            return df["years"]

        return natural_language_to_year
