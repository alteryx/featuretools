from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Filepath

from featuretools.primitives.base import TransformPrimitive


class FileExtension(TransformPrimitive):
    """Determines the extension of a filepath.

    Description:
        Given a list of filepaths, return the extension
        suffix of each one. If the filepath is missing
        or invalid, return `NaN`.

    Examples:
        >>> file_extension = FileExtension()
        >>> file_extension(['doc.txt', '~/documents/data.json', 'file']).tolist()
        ['.txt', '.json', nan]
    """

    name = "file_extension"
    input_types = [ColumnSchema(logical_type=Filepath)]
    return_type = ColumnSchema(semantic_tags={"category"})

    def get_function(self):
        def file_extension(x):
            p = r"(\.[a-z|A-Z]+$)"
            return x.str.extract(p, expand=False).str.lower()

        return file_extension
