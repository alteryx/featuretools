import math
import re

import pandas as pd
from tqdm import tqdm
from woodwork.logical_types import Boolean, Categorical, Ordinal

import featuretools as ft


def load_weather(nrows=None,
                 return_single_table=False):
    '''
    Load the Australian daily-min-termperatures weather dataset.

    Args:

        nrows (int): Passed to nrows in ``pd.read_csv``. Used before filtering.
        return_single_table (bool): Exit the function early and return a dataframe.
        verbose (bool): Show a progress bar while loading the data.

    Returns:
        (pd.Dataframe, pd.Series): X and y
    '''
    filename = "daily-min-temperatures.csv"
    print('Downloading data ...')
    url = "https://api.featurelabs.com/datasets/{}?library=featuretools&version={}".format(filename, ft.__version__)
    data = pd.read_csv(url, index_col=None, nrows=nrows)
    if return_single_table:
        return data
    es = make_es(data)
    return es


def make_es(data):
    es = ft.EntitySet('Weather Data')

    es.add_dataframe(data,
                     dataframe_name='temperatures',
                     index='id',
                     make_index=True,
                     time_index='Date')
    return es
