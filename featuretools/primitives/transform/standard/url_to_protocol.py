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


class URLToProtocol(TransformPrimitive):
    """Determines the protocol (http or https) of a url.

    Description:
        Extract the protocol of a url using regex.
        It will be either https or http. Returns nan if
        the url doesn't contain a protocol.

    Examples:
        >>> url_to_protocol = URLToProtocol()
        >>> urls =  ['https://play.google.com',
        ...          'http://www.google.co.in',
        ...          'www.facebook.com']
        >>> url_to_protocol(urls).to_list()
        ['https', 'http', nan]
    """

    name = "url_to_protocol"
    input_types = [ColumnSchema(logical_type=URL)]
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={"category"})

    def get_function(self):
        def url_to_protocol(x):
            p = r"^(https|http)(?:\:)"
            return x.str.extract(p, expand=False)

        return url_to_protocol
