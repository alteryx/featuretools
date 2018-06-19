import pandas as pd
from dask.base import tokenize

from ..testing_utils import make_ecommerce_entityset


def test_tokenize_entityset():
    es = make_ecommerce_entityset()
    dupe = make_ecommerce_entityset()
    int_es = make_ecommerce_entityset(with_integer_time_index=True)

    # check identitcal entitysets hash to same token
    assert tokenize(es) == tokenize(dupe)

    # not same if product relationship is missing
    productless = make_ecommerce_entityset()
    productless.relationships.pop()
    assert tokenize(es) != tokenize(productless)

    # not same if integer entityset
    assert tokenize(es) != tokenize(int_es)

    # add row to cohorts
    cohorts_df = dupe['cohorts'].df
    new_row = pd.DataFrame(data={'cohort': [2],
                                 'cohort_name': ['On Time Adopters'],
                                 'cohort_end': [pd.Timestamp('2011-04-08 12:00:00')]},
                           columns=['cohort', 'cohort_name', 'cohort_end'],
                           index=[2])
    more_cohorts = cohorts_df.append(new_row, ignore_index=True, sort=True)
    dupe['cohorts'].update_data(more_cohorts)
    assert tokenize(es) == tokenize(dupe)
