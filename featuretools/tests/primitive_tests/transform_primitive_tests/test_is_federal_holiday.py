from datetime import datetime

import numpy as np
import pandas as pd
from pytest import raises

from featuretools.primitives import IsFederalHoliday


def test_regular():
    primitive_instance = IsFederalHoliday()
    primitive_func = primitive_instance.get_function()
    case = pd.Series(
        [
            "2016-01-01",
            "2016-02-29",
            "2017-05-29",
            datetime(2019, 7, 4, 10, 0, 30),
        ],
    ).astype("datetime64[ns]")
    answer = pd.Series([True, False, True, True])
    given_answer = pd.Series(primitive_func(case))
    assert given_answer.equals(answer)


def test_nat():
    primitive_instance = IsFederalHoliday()
    primitive_func = primitive_instance.get_function()
    case = pd.Series(
        [
            "2019-10-14",
            "NaT",
            "2016-02-29",
            "NaT",
        ],
    ).astype("datetime64[ns]")
    answer = pd.Series([True, np.nan, False, np.nan])
    given_answer = pd.Series(primitive_func(case))
    assert given_answer.equals(answer)


def test_valid_country():
    primitive_instance = IsFederalHoliday(country="Canada")
    primitive_func = primitive_instance.get_function()
    case = pd.Series(
        [
            "2016-07-01",
            "2016-11-11",
            "2018-09-03",
        ],
    ).astype("datetime64[ns]")
    answer = pd.Series([True, False, True])
    given_answer = pd.Series(primitive_func(case))
    assert given_answer.equals(answer)


def test_invalid_country():
    error_text = "must be one of the available countries"
    with raises(ValueError, match=error_text):
        IsFederalHoliday(country="")


def test_multiple_countries():
    primitive_mexico = IsFederalHoliday(country="Mexico")
    primitive_func = primitive_mexico.get_function()
    case = pd.Series([datetime(2000, 9, 16), datetime(2005, 1, 1)])
    assert len(primitive_func(case)) > 1
    primitive_india = IsFederalHoliday(country="IND")
    primitive_func = primitive_mexico.get_function()
    case = pd.Series([datetime(2048, 1, 1), datetime(2048, 10, 2)])
    primitive_func = primitive_india.get_function()
    assert len(primitive_func(case)) > 1
    primitive_uk = IsFederalHoliday(country="UK")
    primitive_func = primitive_uk.get_function()
    case = pd.Series([datetime(2048, 3, 17), datetime(2048, 4, 6)])
    assert len(primitive_func(case)) > 1
    countries = [
        "Argentina",
        "AU",
        "Austria",
        "BY",
        "Belgium",
        "Brazil",
        "Canada",
        "Colombia",
        "Croatia",
        "England",
        "Finland",
        "FRA",
        "Germany",
        "Germany",
        "Italy",
        "NewZealand",
        "PortugalExt",
        "PTE",
        "Spain",
        "ES",
        "Switzerland",
        "UnitedStates",
        "US",
        "UK",
        "UA",
        "CH",
        "SE",
        "ZA",
    ]
    for x in countries:
        IsFederalHoliday(country=x)
