import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Categorical, PersonFullName

from featuretools.primitives.base import TransformPrimitive


class FullNameToFirstName(TransformPrimitive):
    """Determines the first name from a person's name.

    Description:
        Given a list of names, determines the first name. If
        only a single name is provided, assume this is a first name.
        If only a title and a single name is provided return `nan`.
        This assumes all titles will be followed by a period. Please note,
        in the current implementation, last names containing spaces may
        result in improper first name matches.


    Examples:
        >>> full_name_to_first_name = FullNameToFirstName()
        >>> names = ['Woolf Spector', 'Oliva y Ocana, Dona. Fermina',
        ...          'Ware, Mr. Frederick', 'Peter, Michael J', 'Mr. Brown']
        >>> full_name_to_first_name(names).to_list()
        ['Woolf', 'Oliva', 'Frederick', 'Michael', nan]
    """

    name = "full_name_to_first_name"
    input_types = [ColumnSchema(logical_type=PersonFullName)]
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={"category"})

    def get_function(self):
        def full_name_to_first_name(x):
            title_with_last_pattern = r"(^[A-Z][a-z]+\. [A-Z][a-z]+$)"
            titles_pattern = r"([A-Z][a-z]+)\. "
            df = pd.DataFrame({"names": x})
            # remove any entries with just a title and a name
            df["names"] = df["names"].str.replace(
                title_with_last_pattern,
                "",
                regex=True,
            )
            # remove any known titles
            df["names"] = df["names"].str.replace(titles_pattern, "", regex=True)
            # extract first names
            pattern = r"([A-Z][a-z]+ |, [A-Z][a-z]+$|^[A-Z][a-z]+$)"
            df["first_name"] = df["names"].str.extract(pattern)
            # clean up white space and leftover commas
            df["first_name"] = df["first_name"].str.replace(",", "").str.strip()
            return df["first_name"]

        return full_name_to_first_name
