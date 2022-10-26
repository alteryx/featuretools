from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import URL, Categorical

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.common_tld_utils import COMMON_TLDS


class URLToTLD(TransformPrimitive):
    """Determines the top level domain of a url.

    Description:
        Extract the top level domain of a url, using regex,
        and a list of common top level domains. Returns nan if
        the url is invalid or null.
        Common top level domains were pulled from this list:
        https://www.hayksaakian.com/most-popular-tlds/

    Examples:
        >>> url_to_tld = URLToTLD()
        >>> urls = ['https://www.google.com', 'http://www.google.co.in',
        ...         'www.facebook.com']
        >>> url_to_tld(urls).to_list()
        ['com', 'in', 'com']
    """

    name = "url_to_tld"
    input_types = [ColumnSchema(logical_type=URL)]
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={"category"})

    def get_function(self):
        self.tlds_pattern = r"(?:\.({}))".format("|".join(COMMON_TLDS))

        def url_to_domain(x):
            p = r"^(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)?([^:\/?\n]+)"
            return x.str.extract(p, expand=False)

        def url_to_tld(x):
            domains = url_to_domain(x)
            df = domains.str.extractall(self.tlds_pattern)
            matches = df.groupby(level=0).last()[0]
            return matches.reindex(x.index)

        return url_to_tld
