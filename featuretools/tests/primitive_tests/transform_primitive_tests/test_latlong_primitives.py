import numpy as np
import pandas as pd
import pytest

from featuretools.primitives import CityblockDistance, GeoMidpoint, IsInGeoBox


def test_cityblock():
    primitive_instance = CityblockDistance()
    latlong_1 = pd.Series([(i, i) for i in range(3)])
    latlong_2 = pd.Series([(i, i) for i in range(3, 6)])
    answer = pd.Series([414.56051391, 414.52893691, 414.43421555])
    given_answer = primitive_instance(latlong_1, latlong_2)
    np.testing.assert_allclose(given_answer, answer, rtol=1e-09)

    primitive_instance = CityblockDistance(unit="kilometers")
    answer = primitive_instance(latlong_1, latlong_2)
    given_answer = pd.Series([667.1704814, 667.11966315, 666.96722389])
    np.testing.assert_allclose(given_answer, answer, rtol=1e-09)


def test_cityblock_nans():
    primitive_instance = CityblockDistance()
    lats_longs_1 = [(i, i) for i in range(2)]
    lats_longs_2 = [(i, i) for i in range(2, 4)]
    lats_longs_1 += [(1, 1), (np.nan, 3), (4, np.nan), (np.nan, np.nan)]
    lats_longs_2 += [(np.nan, np.nan), (np.nan, 5), (6, np.nan), (np.nan, np.nan)]
    given_answer = pd.Series(list([276.37367594, 276.35262728] + [np.nan] * 4))
    answer = primitive_instance(lats_longs_1, lats_longs_2)
    np.testing.assert_allclose(given_answer, answer, rtol=1e-09)


def test_cityblock_error():
    error_text = "Invalid unit given"
    with pytest.raises(ValueError, match=error_text):
        CityblockDistance(unit="invalid")


def test_midpoint():
    latlong1 = pd.Series([(-90, -180), (90, 180)])
    latlong2 = pd.Series([(+90, +180), (-90, -180)])
    function = GeoMidpoint().get_function()
    answer = function(latlong1, latlong2)
    for lat, longi in answer:
        assert lat == 0.0
        assert longi == 0.0


def test_midpoint_floating():
    latlong1 = pd.Series([(-45.5, -100.5), (45.5, 100.5)])
    latlong2 = pd.Series([(+45.5, +100.5), (-45.5, -100.5)])
    function = GeoMidpoint().get_function()
    answer = function(latlong1, latlong2)
    for lat, longi in answer:
        assert lat == 0.0
        assert longi == 0.0


def test_midpoint_zeros():
    latlong1 = pd.Series([(0, 0), (0, 0)])
    latlong2 = pd.Series([(0, 0), (0, 0)])
    function = GeoMidpoint().get_function()
    answer = function(latlong1, latlong2)
    for lat, longi in answer:
        assert lat == 0.0
        assert longi == 0.0


def test_midpoint_nan():
    all_nan = pd.Series([(np.nan, np.nan), (np.nan, np.nan)])
    latlong1 = pd.Series([(0, 0), (0, 0)])
    function = GeoMidpoint().get_function()
    answer = function(all_nan, latlong1)
    for lat, longi in answer:
        assert np.isnan(lat)
        assert np.isnan(longi)


def test_isingeobox():
    latlong = pd.Series(
        [
            (1, 2),
            (5, 7),
            (-5, 4),
            (2, 3),
            (0, 0),
            (np.nan, np.nan),
            (-2, np.nan),
            (np.nan, 1),
        ],
    )
    bottomleft = (-5, -5)
    topright = (5, 5)
    primitive = IsInGeoBox(bottomleft, topright)
    function = primitive.get_function()
    primitive_answer = function(latlong)
    answer = pd.Series([True, False, True, True, True, False, False, False])
    assert np.array_equal(primitive_answer, answer)


def test_boston():
    NYC = (40.7128, -74.0060)
    SF = (37.7749, -122.4194)
    Somerville = (42.3876, -71.0995)
    Bejing = (39.9042, 116.4074)
    CapeTown = (-33.9249, 18.4241)
    latlong = pd.Series([NYC, SF, Somerville, Bejing, CapeTown])
    LynnMA = (42.4668, -70.9495)
    DedhamMA = (42.2436, -71.1677)
    primitive = IsInGeoBox(LynnMA, DedhamMA)
    function = primitive.get_function()
    primitive_answer = function(latlong)
    answer = pd.Series([False, False, True, False, False])
    assert np.array_equal(primitive_answer, answer)
