import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import (
    URL,
    Boolean,
    BooleanNullable,
    Categorical,
    Datetime,
    Double,
    EmailAddress,
    NaturalLanguage,
    Timedelta,
)

from featuretools.primitives.core.transform_primitive import TransformPrimitive
from featuretools.utils.common_tld_utils import COMMON_TLDS
from featuretools.utils.gen_utils import Library

class URLToDomain(TransformPrimitive):
    """Determines the domain of a url.

    Description:
        Calculates the label to identify the network domain of a URL. Supports
        urls with or without protocol as well as international country domains.

    Examples:
        >>> url_to_domain = URLToDomain()
        >>> urls =  ['https://play.google.com',
        ...          'http://www.google.co.in',
        ...          'www.facebook.com']
        >>> url_to_domain(urls).tolist()
        ['play.google.com', 'google.co.in', 'facebook.com']
    """

    name = "url_to_domain"
    input_types = [ColumnSchema(logical_type=URL)]
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={"category"})

    def get_function(self):
        def url_to_domain(x):
            p = r"^(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)?([^:\/?\n]+)"
            return x.str.extract(p, expand=False)

        return url_to_domain
