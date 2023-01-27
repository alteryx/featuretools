import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable, NaturalLanguage

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class NumCharacters(TransformPrimitive):
    """Calculates the number of characters in a given string, including whitespace and punctuation.

    Description:
        Returns the number of characters in a string. This is equivalent to the length of a string.

    Examples:
        >>> num_characters = NumCharacters()
        >>> num_characters(['This is a string',
        ...                 'second item',
        ...                 'final1']).tolist()
        [16, 11, 6]
    """

    name = "num_characters"
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the number of characters in {}"

    def get_function(self):
        def character_counter(array):
            def _get_num_characters(elem):
                """Returns the length of elem, or pd.NA given null input"""
                if pd.isna(elem):
                    return pd.NA
                return len(elem)

            return array.apply(_get_num_characters)

        return character_counter
