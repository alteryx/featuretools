# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import pytest

import featuretools as ft
from featuretools import variable_types as vtypes
from featuretools.utils.entity_utils import (
    convert_all_variable_data,
    convert_variable_data,
    get_linked_vars,
    infer_variable_types
)


@pytest.fixture(scope='module')
def es():
    es = ft.demo.load_mock_customer(return_entityset=True)
    return es


def test_infer_variable_types(es):
    # Customers Entity
    total_variables = es['customers'].df.columns
    variable_types = ['customer_id']

    entity = es['customers']
    link_vars = get_linked_vars(entity)
    inferred_variable_types = infer_variable_types(entity.df,
                                                   link_vars,
                                                   variable_types,
                                                   'join_date',
                                                   {})

    # Check columns' number
    assert len(variable_types) + len(inferred_variable_types) == len(total_variables)

    # Check columns' types
    assert isinstance(entity['join_date'], inferred_variable_types['join_date'])
    assert inferred_variable_types['join_date'] == vtypes.Datetime

    assert isinstance(entity['date_of_birth'], inferred_variable_types['date_of_birth'])
    assert inferred_variable_types['date_of_birth'] == vtypes.Datetime

    assert isinstance(entity['zip_code'], inferred_variable_types['zip_code'])
    assert inferred_variable_types['zip_code'] == vtypes.Categorical

    # Sessions Entity
    entity = es['sessions']
    link_vars = get_linked_vars(entity)
    total_variables = entity.df.columns
    variable_types = ['session_id']
    inferred_variable_types = infer_variable_types(entity.df,
                                                   link_vars,
                                                   variable_types,
                                                   'session_start',
                                                   {})

    # Check columns' number
    assert len(variable_types) + len(inferred_variable_types) == len(total_variables)

    # Check columns' types
    assert inferred_variable_types['customer_id'] == vtypes.Ordinal

    assert isinstance(entity['device'], inferred_variable_types['device'])
    assert inferred_variable_types['device'] == vtypes.Categorical

    assert isinstance(entity['session_start'], inferred_variable_types['session_start'])
    assert inferred_variable_types['session_start'] == vtypes.Datetime


def test_convert_all_variable_data(es):
    variable_types = {
        'transaction_id': vtypes.Numeric,
        'session_id': vtypes.Numeric,
        'transaction_time': vtypes.Datetime,
        'amount': vtypes.Numeric,
        'product_id': vtypes.Numeric
    }
    es['transactions'].df = convert_all_variable_data(es['transactions'].df, variable_types)
    assert es['transactions'].df['transaction_id'].dtype.name in vtypes.PandasTypes._pandas_numerics
    assert es['transactions'].df['session_id'].dtype.name in vtypes.PandasTypes._pandas_numerics
    assert es['transactions'].df['transaction_time'].dtype.name in vtypes.PandasTypes._pandas_datetimes
    assert es['transactions'].df['amount'].dtype.name in vtypes.PandasTypes._pandas_numerics
    assert es['transactions'].df['product_id'].dtype.name in vtypes.PandasTypes._pandas_numerics


def test_convert_variable_data(es):
    init_dtype = es['products'].df['product_id'].dtype.name
    es['products'].df = convert_variable_data(es['products'].df,
                                              'product_id',
                                              vtypes.Numeric)
    assert init_dtype != es['products'].df['product_id'].dtype.name
    assert es['products'].df['product_id'].dtype.name in vtypes.PandasTypes._pandas_numerics

    init_dtype = es['customers'].df['zip_code'].dtype.name
    es['customers'].df = convert_variable_data(es['customers'].df,
                                               'zip_code',
                                               vtypes.Numeric)
    assert init_dtype != es['customers'].df['zip_code'].dtype.name
    assert es['customers'].df['zip_code'].dtype.name in vtypes.PandasTypes._pandas_numerics
