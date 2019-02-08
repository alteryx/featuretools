# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import pytest

import featuretools as ft
from featuretools import variable_types as vtypes
from featuretools.utils.entity_utils import (
    convert_all_variable_data,
    convert_variable_data,
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
    inferred_variable_types = infer_variable_types(es['customers'],
                                                   variable_types,
                                                   'join_date',
                                                   {})

    # Check columns' number
    assert len(variable_types) + len(inferred_variable_types) == len(total_variables)

    # Check columns' types
    assert isinstance(es['customers']['join_date'], inferred_variable_types['join_date'])
    assert inferred_variable_types['join_date'] == vtypes.Datetime

    assert isinstance(es['customers']['date_of_birth'], inferred_variable_types['date_of_birth'])
    assert inferred_variable_types['date_of_birth'] == vtypes.Datetime

    assert isinstance(es['customers']['zip_code'], inferred_variable_types['zip_code'])
    assert inferred_variable_types['zip_code'] == vtypes.Categorical

    # Sessions Entity
    total_variables = es['sessions'].df.columns
    variable_types = ['session_id']
    inferred_variable_types = infer_variable_types(es['sessions'],
                                                   variable_types,
                                                   'session_start',
                                                   {})

    # Check columns' number
    assert len(variable_types) + len(inferred_variable_types) == len(total_variables)

    # Check columns' types
    assert inferred_variable_types['customer_id'] == vtypes.Ordinal

    assert isinstance(es['sessions']['device'], inferred_variable_types['device'])
    assert inferred_variable_types['device'] == vtypes.Categorical

    assert isinstance(es['sessions']['session_start'], inferred_variable_types['session_start'])
    assert inferred_variable_types['session_start'] == vtypes.Datetime


def test_convert_all_variable_data(es):
    variable_types = {
        'transaction_id': vtypes.Numeric,
        'session_id': vtypes.Numeric,
        'transaction_time': vtypes.Datetime,
        'amount': vtypes.Numeric,
        'product_id': vtypes.Numeric
    }
    convert_all_variable_data(es['transactions'], variable_types)
    assert es['transactions'].df['transaction_id'].dtype.name in vtypes.PandasTypes._pandas_numerics
    assert es['transactions'].df['session_id'].dtype.name in vtypes.PandasTypes._pandas_numerics
    assert es['transactions'].df['transaction_time'].dtype.name in vtypes.PandasTypes._pandas_datetimes
    assert es['transactions'].df['amount'].dtype.name in vtypes.PandasTypes._pandas_numerics
    assert es['transactions'].df['product_id'].dtype.name in vtypes.PandasTypes._pandas_numerics


def test_convert_variable_data(es):
    init_dtype = es['products'].df['product_id'].dtype.name
    convert_variable_data(es['products'], 'product_id', vtypes.Numeric)
    assert init_dtype != es['products'].df['product_id'].dtype.name
    assert es['products'].df['product_id'].dtype.name in vtypes.PandasTypes._pandas_numerics

    init_dtype = es['customers'].df['zip_code'].dtype.name
    convert_variable_data(es['customers'], 'zip_code', vtypes.Numeric)
    assert init_dtype != es['customers'].df['zip_code'].dtype.name
    assert es['customers'].df['zip_code'].dtype.name in vtypes.PandasTypes._pandas_numerics
