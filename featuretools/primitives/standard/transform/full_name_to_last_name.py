import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Categorical, PersonFullName

from featuretools.primitives.base import TransformPrimitive


class FullNameToLastName(TransformPrimitive):
    """Determines the first name from a person's name.

    Description:
        Given a list of names, determines the last name. If
        only a single name is provided, assume this is a first name, and
        return `nan`. This assumes all titles will be followed by a period.


    Examples:
        >>> full_name_to_last_name = FullNameToLastName()
        >>> names = ['Woolf Spector', 'Oliva y Ocana, Dona. Fermina',
        ...          'Ware, Mr. Frederick', 'Peter, Michael J', 'Mr. Brown']
        >>> full_name_to_last_name(names).to_list()
        ['Spector', 'Oliva y Ocana', 'Ware', 'Peter', 'Brown']
    """

    name = "full_name_to_last_name"
    input_types = [ColumnSchema(logical_type=PersonFullName)]
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={"category"})

    def get_function(self):
        def full_name_to_last_name(x):
            titles_pattern = r"([A-Z][a-z]+)\. "
            df = pd.DataFrame({"names": x})
            # extract initial names
            pattern = r"(^.+?,|^[A-Z][a-z]+\. [A-Z][a-z]+$| [A-Z][a-z]+$| [A-Z][a-z]+[/-][A-Z][a-z]+$)"
            df["last_name"] = df["names"].str.extract(pattern)
            # remove titles
            df["last_name"] = df["last_name"].str.replace(
                titles_pattern,
                "",
                regex=True,
            )
            # clean up white space and leftover commas
            df["last_name"] = df["last_name"].str.replace(",", "").str.strip()
            return df["last_name"]

        return full_name_to_last_name
