import copy
import logging
from datetime import datetime

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

import featuretools as ft
from featuretools import variable_types
from featuretools.entityset.ww_entityset import EntitySet
from featuretools.tests.testing_utils import to_pandas
from featuretools.utils.gen_utils import import_or_none
from featuretools.utils.koalas_utils import pd_to_ks_clean


def test_empty_es():
    es = EntitySet('es')
    assert es.id == 'es'
    assert es.dataframe_dict == {}
    assert es.relationships == []
    assert es.time_type == None


def test_init_es_with_dataframe():
    pass


def test_init_es_with_multiple_dataframes():
    pass


def test_add_dataframe_to_es():
    pass


def test_init_es_with_relationships():
    pass


def test_add_relationships_to_es():
    pass
