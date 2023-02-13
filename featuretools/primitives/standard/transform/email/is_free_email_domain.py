import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable, EmailAddress

from featuretools.primitives.base import TransformPrimitive


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

        free_domains = pd.read_csv(file_path, header=None, names=["domain"])
        free_domains["domain"] = free_domains.domain.str.strip()

        def is_free_email_domain(emails):
            # if the input is empty return an empty Series
            if len(emails) == 0:
                return pd.Series([], dtype="category")

            emails_df = pd.DataFrame({"email": emails})

            # if all emails are NaN expand won't propogate NaNs and will fail on indexing
            if emails_df["email"].isnull().all():
                emails_df["domain"] = np.nan
            else:
                # .str.strip() and .str.split() return NaN for NaN values and propogate NaNs into new columns
                emails_df["domain"] = (
                    emails_df["email"].str.strip().str.split("@", expand=True)[1]
                )

            emails_df["is_free"] = emails_df["domain"].isin(free_domains["domain"])

            # if there are any NaN domain values, change the series type to allow for
            # both bools and NaN values and set is_free to NaN for the NaN domains
            if emails_df["domain"].isnull().values.any():
                emails_df["is_free"] = emails_df["is_free"].astype("object")
                emails_df.loc[emails_df["domain"].isnull(), "is_free"] = np.nan
            return emails_df.is_free.values

        return is_free_email_domain
