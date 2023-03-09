from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Categorical, PersonFullName

from featuretools.primitives.base import TransformPrimitive


class FullNameToTitle(TransformPrimitive):
    """Determines the title from a person's name.

    Description:
        Given a list of names, determines the title, or
        prefix of each name (e.g. "Mr", "Mrs", etc). If
        no title is found, returns `NaN`.

    Examples:
        >>> full_name_to_title = FullNameToTitle()
        >>> names = ['Spector, Mr. Woolf', 'Oliva y Ocana, Dona. Fermina',
        ...          'Ware, Mr. Frederick', 'Peter, Michael J', 'Mr. Brown']
        >>> full_name_to_title(names).to_list()
        ['Mr', 'Dona', 'Mr', nan, 'Mr']
    """

    name = "full_name_to_title"
    input_types = [ColumnSchema(logical_type=PersonFullName)]
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={"category"})

    def get_function(self):
        def full_name_to_title(x):
            pattern = r"([A-Z][a-z]+)\. "
            return x.str.extract(pattern, expand=True)[0]

        return full_name_to_title
