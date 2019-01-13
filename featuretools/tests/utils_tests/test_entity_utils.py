# -*- coding: utf-8 -*-
import featuretools as ft
from featuretools import variable_types as vtypes
from featuretools.utils.entity_utils import (
    convert_all_variable_data,
    convert_variable_data,
    infer_variable_types
)


def test_infer_variable_types():
    es = ft.demo.load_mock_customer(return_entityset=True)
    total_variables = es['products'].df.columns
    variable_types = ['product_id']
    inferred_variable_types = infer_variable_types(es['products'],
                                                   variable_types, None, {})
    assert len(variable_types) + len(inferred_variable_types) == len(total_variables)


def test_convert_all_variable_data():
    es = ft.demo.load_mock_customer(return_entityset=True)
    init_dtypes = es['products'].df.dtypes
    variable_types = {
        'product_id': vtypes.Numeric,
        'brand': vtypes.Categorical
    }
    convert_all_variable_data(es['products'], variable_types)
    assert (init_dtypes != es['products'].df.dtypes).any()


def test_convert_variable_data():
    es = ft.demo.load_mock_customer(return_entityset=True)
    init_dtypes = es['products'].df['product_id'].dtypes
    convert_variable_data(es['products'], 'product_id', vtypes.Numeric)
    assert init_dtypes != es['products'].df['product_id'].dtypes
