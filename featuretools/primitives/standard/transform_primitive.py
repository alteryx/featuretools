import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import (
    URL,
    Boolean,
    BooleanNullable,
    Categorical,
    Double,
    EmailAddress,
    NaturalLanguage
)

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.common_tld_utils import COMMON_TLDS
from featuretools.utils.gen_utils import Library


class IsNull(TransformPrimitive):
    """Determines if a value is null.

    Examples:
        >>> is_null = IsNull()
        >>> is_null([1, None, 3]).tolist()
        [False, True, False]
    """
    name = "is_null"
    input_types = [ColumnSchema()]
    return_type = ColumnSchema(logical_type=Boolean)
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "whether {} is null"

    def get_function(self):
        def isnull(array):
            return array.isnull()
        return isnull


class Absolute(TransformPrimitive):
    """Computes the absolute value of a number.

    Examples:
        >>> absolute = Absolute()
        >>> absolute([3.0, -5.0, -2.4]).tolist()
        [3.0, 5.0, 2.4]
    """
    name = "absolute"
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the absolute value of {}"

    def get_function(self):
        return np.absolute


class SquareRoot(TransformPrimitive):
    """Computes the square root of a number.

    Examples:
        >>> sqrt = SquareRoot()
        >>> sqrt([9.0, 16.0, 4.0]).tolist()
        [3.0, 4.0, 2.0]
    """
    name = "square_root"
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={'numeric'})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the square root of {}"

    def get_function(self):
        return np.sqrt


class NaturalLogarithm(TransformPrimitive):
    """Computes the natural logarithm of a number.

    Examples:
        >>> log = NaturalLogarithm()
        >>> log([1.0, np.e]).tolist()
        [0.0, 1.0]
    """
    name = "natural_logarithm"
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={'numeric'})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the natural logarithm of {}"

    def get_function(self):
        return np.log


class Sine(TransformPrimitive):
    """Computes the sine of a number.

    Examples:
        >>> sin = Sine()
        >>> sin([-np.pi/2.0, 0.0, np.pi/2.0]).tolist()
        [-1.0, 0.0, 1.0]
    """
    name = "sine"
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={'numeric'})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the sine of {}"

    def get_function(self):
        return np.sin


class Cosine(TransformPrimitive):
    """Computes the cosine of a number.

    Examples:
        >>> cos = Cosine()
        >>> cos([0.0, np.pi/2.0, np.pi]).tolist()
        [1.0, 6.123233995736766e-17, -1.0]
    """
    name = "cosine"
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={'numeric'})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the cosine of {}"

    def get_function(self):
        return np.cos


class Tangent(TransformPrimitive):
    """Computes the tangent of a number.

    Examples:
        >>> tan = Tangent()
        >>> tan([-np.pi, 0.0, np.pi/2.0]).tolist()
        [1.2246467991473532e-16, 0.0, 1.633123935319537e+16]
    """
    name = "tangent"
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={'numeric'})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the tangent of {}"

    def get_function(self):
        return np.tan


class NumCharacters(TransformPrimitive):
    """Calculates the number of characters in a string.

    Examples:
        >>> num_characters = NumCharacters()
        >>> num_characters(['This is a string',
        ...                 'second item',
        ...                 'final1']).tolist()
        [16, 11, 6]
    """
    name = 'num_characters'
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the number of characters in {}"

    def get_function(self):
        def character_counter(array):
            return array.fillna('').str.len()
        return character_counter


class NumWords(TransformPrimitive):
    """Determines the number of words in a string by counting the spaces.

    Examples:
        >>> num_words = NumWords()
        >>> num_words(['This is a string',
        ...            'Two words',
        ...            'no-spaces',
        ...            'Also works with sentences. Second sentence!']).tolist()
        [4, 2, 1, 6]
    """
    name = 'num_words'
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the number of words in {}"

    def get_function(self):
        def word_counter(array):
            return array.fillna('').str.count(' ') + 1
        return word_counter


class IsIn(TransformPrimitive):
    """Determines whether a value is present in a provided list.

    Examples:
        >>> items = ['string', 10.3, False]
        >>> is_in = IsIn(list_of_outputs=items)
        >>> is_in(['string', 10.5, False]).tolist()
        [True, False, True]
    """
    name = "isin"
    input_types = [ColumnSchema()]
    return_type = ColumnSchema(logical_type=Boolean)
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]

    def __init__(self, list_of_outputs=None):
        self.list_of_outputs = list_of_outputs
        if not list_of_outputs:
            stringified_output_list = '[]'
        else:
            stringified_output_list = ', '.join([str(x) for x in list_of_outputs])
        self.description_template = "whether {{}} is in {}".format(stringified_output_list)

    def get_function(self):
        def pd_is_in(array):
            return array.isin(self.list_of_outputs or [])
        return pd_is_in

    def generate_name(self, base_feature_names):
        return u"%s.isin(%s)" % (base_feature_names[0],
                                 str(self.list_of_outputs))


class Diff(TransformPrimitive):
    """Compute the difference between the value in a list and the
    previous value in that list.

    Description:
        Given a list of values, compute the difference from the previous
        item in the list. The result for the first element of the list will
        always be `NaN`. If the values are datetimes, the output will be a
        timedelta.

    Examples:
        >>> diff = Diff()
        >>> values = [1, 10, 3, 4, 15]
        >>> diff(values).tolist()
        [nan, 9.0, -7.0, 1.0, 11.0]
    """
    name = "diff"
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    uses_full_dataframe = True
    description_template = "the difference from the previous value of {}"

    def get_function(self):
        def pd_diff(values):
            return values.diff()
        return pd_diff


class Negate(TransformPrimitive):
    """Negates a numeric value.

    Examples:
        >>> negate = Negate()
        >>> negate([1.0, 23.2, -7.0]).tolist()
        [-1.0, -23.2, 7.0]
    """
    name = "negate"
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the negation of {}"

    def get_function(self):
        def negate(vals):
            return vals * -1
        return negate

    def generate_name(self, base_feature_names):
        return "-(%s)" % (base_feature_names[0])


class Not(TransformPrimitive):
    """Negates a boolean value.

    Examples:
        >>> not_func = Not()
        >>> not_func([True, True, False]).tolist()
        [False, False, True]
    """
    name = "not"
    input_types = [[ColumnSchema(logical_type=Boolean)], [ColumnSchema(logical_type=BooleanNullable)]]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the negation of {}"

    def generate_name(self, base_feature_names):
        return u"NOT({})".format(base_feature_names[0])

    def get_function(self):
        return np.logical_not


class Percentile(TransformPrimitive):
    """Determines the percentile rank for each value in a list.

    Examples:
        >>> percentile = Percentile()
        >>> percentile([10, 15, 1, 20]).tolist()
        [0.5, 0.75, 0.25, 1.0]

        Nan values are ignored when determining rank

        >>> percentile([10, 15, 1, None, 20]).tolist()
        [0.5, 0.75, 0.25, nan, 1.0]
    """
    name = 'percentile'
    uses_full_dataframe = True
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    description_template = "the percentile rank of {}"

    def get_function(self):
        return lambda array: array.rank(pct=True)


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
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={'category'})

    def get_function(self):
        def url_to_domain(x):
            p = r'^(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)?([^:\/?\n]+)'
            return x.str.extract(p, expand=False)
        return url_to_domain


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
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={'category'})

    def get_function(self):
        def url_to_protocol(x):
            p = r'^(https|http)(?:\:)'
            return x.str.extract(p, expand=False)
        return url_to_protocol


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
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={'category'})

    def get_function(self):
        self.tlds_pattern = r'(?:\.({}))'.format('|'.join(COMMON_TLDS))

        def url_to_domain(x):
            p = r'^(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)?([^:\/?\n]+)'
            return x.str.extract(p, expand=False)

        def url_to_tld(x):
            domains = url_to_domain(x)
            df = domains.str.extractall(self.tlds_pattern)
            matches = df.groupby(level=0).last()[0]
            return matches.reindex(x.index)
        return url_to_tld


class IsFreeEmailDomain(TransformPrimitive):
    """Determines if an email address is from a free email domain.

    Description:
        EmailAddress input should be a string. Will return Nan
        if an invalid email address is provided, or if the input is
        not a string. The list of free email domains used in this primitive
        was obtained from https://github.com/willwhite/freemail/blob/master/data/free.txt.

    Examples:
        >>> is_free_email_domain = IsFreeEmailDomain()
        >>> is_free_email_domain(['name@gmail.com', 'name@featuretools.com']).tolist()
        [True, False]
    """
    name = "is_free_email_domain"
    input_types = [ColumnSchema(logical_type=EmailAddress)]
    return_type = ColumnSchema(logical_type=BooleanNullable)

    filename = "free_email_provider_domains.txt"

    def get_function(self):
        file_path = self.get_filepath(self.filename)

        free_domains = pd.read_csv(file_path, header=None, names=['domain'])
        free_domains['domain'] = free_domains.domain.str.strip()

        def is_free_email_domain(emails):
            # if the input is empty return an empty Series
            if len(emails) == 0:
                return pd.Series([])

            emails_df = pd.DataFrame({'email': emails})

            # if all emails are NaN expand won't propogate NaNs and will fail on indexing
            if emails_df['email'].isnull().all():
                emails_df['domain'] = np.nan
            else:
                # .str.strip() and .str.split() return NaN for NaN values and propogate NaNs into new columns
                emails_df['domain'] = emails_df['email'].str.strip().str.split('@', expand=True)[1]

            emails_df['is_free'] = emails_df['domain'].isin(free_domains['domain'])

            # if there are any NaN domain values, change the series type to allow for
            # both bools and NaN values and set is_free to NaN for the NaN domains
            if emails_df['domain'].isnull().values.any():
                emails_df['is_free'] = emails_df['is_free'].astype(np.object)
                emails_df.loc[emails_df['domain'].isnull(), 'is_free'] = np.nan
            return emails_df.is_free.values
        return is_free_email_domain


class EmailAddressToDomain(TransformPrimitive):
    """Determines the domain of an email

    Description:
        EmailAddress input should be a string. Will return Nan
        if an invalid email address is provided, or if the input is
        not a string.

    Examples:
        >>> email_address_to_domain = EmailAddressToDomain()
        >>> email_address_to_domain(['name@gmail.com', 'name@featuretools.com']).tolist()
        ['gmail.com', 'featuretools.com']
    """
    name = "email_address_to_domain"
    input_types = [ColumnSchema(logical_type=EmailAddress)]
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={'category'})

    def get_function(self):
        def email_address_to_domain(emails):
            # if the input is empty return an empty Series
            if len(emails) == 0:
                return pd.Series([])

            emails_df = pd.DataFrame({'email': emails})

            # if all emails are NaN expand won't propogate NaNs and will fail on indexing
            if emails_df['email'].isnull().all():
                emails_df['domain'] = np.nan
                emails_df['domain'] = emails_df['domain'].astype(object)
            else:
                # .str.strip() and .str.split() return NaN for NaN values and propogate NaNs into new columns
                emails_df['domain'] = emails_df['email'].str.strip().str.split('@', expand=True)[1]
            return emails_df.domain.values
        return email_address_to_domain


class NumericLag(TransformPrimitive):
    """Shifts an array of values by a specified number of periods.

    Args:
        periods (int): The number of periods by which to shift the input.
            Default is 1. Periods correspond to rows.

        fill_value (int, float, optional): The value to use to fill in
            the gaps left after shifting the input. Default is None.

    Examples:
        >>> lag = NumericLag()
        >>> lag(pd.Series(pd.date_range(start="2020-01-01", periods=5, freq='D')), [1, 2, 3, 4, 5]).tolist()
        [nan, 1.0, 2.0, 3.0, 4.0]

        You can specify the number of periods to shift the values

        >>> lag_periods = NumericLag(periods=3)
        >>> lag_periods(pd.Series(pd.date_range(start="2020-01-01", periods=5, freq='D')), [1, 2, 3, 4, 5]).tolist()
        [nan, nan, nan, 1.0, 2.0]

        You can specify the fill value to use

        >>> lag_fill_value = NumericLag(fill_value=100)
        >>> lag_fill_value(pd.Series(pd.date_range(start="2020-01-01", periods=4, freq='D')), [1, 2, 3, 4]).tolist()
        [100, 1, 2, 3]
    """
    name = "numeric_lag"
    input_types = [ColumnSchema(semantic_tags={'time_index'}), ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    uses_full_dataframe = True

    def __init__(self, periods=1, fill_value=None):
        self.periods = periods
        self.fill_value = fill_value

    def get_function(self):
        def lag(time_index, numeric):
            x = pd.Series(numeric.values, index=time_index.values)
            return x.shift(periods=self.periods, fill_value=self.fill_value).values
        return lag
