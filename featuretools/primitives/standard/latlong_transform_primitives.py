import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable, Double, LatLong

from featuretools.primitives.base import TransformPrimitive
from featuretools.primitives.utils import (
    _deconstrct_latlongs,
    _haversine_calculate
)


class CityblockDistance(TransformPrimitive):
    """Calculates the distance between points in a city road grid.

    Description:
        This distance is calculated using the haversine formula, which
        takes into account the curvature of the Earth.
        If either input data contains `NaN`s, the calculated
        distance with be `NaN`.
        This calculation is also known as the Mahnattan distance.

    Args:
        unit (str): Determines the unit value to output. Could
            be miles or kilometers. Default is miles.

    Examples:
        >>> cityblock_distance = CityblockDistance()
        >>> DC = (38, -77)
        >>> Boston = (43, -71)
        >>> NYC = (40, -74)
        >>> distances_mi = cityblock_distance([DC, DC], [NYC, Boston])
        >>> np.round(distances_mi, 3).tolist()
        [301.519, 672.089]

        We can also change the units in which the distance is calculated.

        >>> cityblock_distance_kilometers = CityblockDistance(unit='kilometers')
        >>> distances_km = cityblock_distance_kilometers([DC, DC], [NYC, Boston])
        >>> np.round(distances_km, 3).tolist()
        [485.248, 1081.622]
    """
    name = "cityblock_distance"
    input_types = [ColumnSchema(logical_type=LatLong), ColumnSchema(logical_type=LatLong)]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={'numeric'})
    commutative = True

    def __init__(self, unit='miles'):
        if unit not in ['miles', 'kilometers']:
            raise ValueError("Invalid unit given")
        self.unit = unit

    def get_function(self):
        def cityblock(latlong_1, latlong_2):
            lat_1s, lon_1s = _deconstrct_latlongs(latlong_1)
            lat_2s, lon_2s = _deconstrct_latlongs(latlong_2)
            lon_dis = _haversine_calculate(lat_1s, lon_1s, lat_1s, lon_2s,
                                           self.unit)
            lat_dist = _haversine_calculate(lat_1s, lon_1s, lat_2s, lon_1s,
                                            self.unit)
            return pd.Series(lon_dis + lat_dist)
        return cityblock


class GeoMidpoint(TransformPrimitive):
    """Determines the geographic center of two coordinates.

    Examples:
        >>> geomidpoint = GeoMidpoint()
        >>> geomidpoint([(42.4, -71.1)], [(40.0, -122.4)])
        [(41.2, -96.75)]
    """
    name = "geomidpoint"
    input_types = [ColumnSchema(logical_type=LatLong), ColumnSchema(logical_type=LatLong)]
    return_type = ColumnSchema(logical_type=LatLong)
    commutative = True

    def get_function(self):
        def geomidpoint_func(latlong_1, latlong_2):
            lat_1s, lon_1s = _deconstrct_latlongs(latlong_1)
            lat_2s, lon_2s = _deconstrct_latlongs(latlong_2)
            lat_middle = np.array([lat_1s, lat_2s]).transpose().mean(axis=1)
            lon_middle = np.array([lon_1s, lon_2s]).transpose().mean(axis=1)
            return list(zip(lat_middle, lon_middle))
        return geomidpoint_func


class Haversine(TransformPrimitive):
    """Calculates the approximate haversine distance between two LatLong columns.

        Args:
            unit (str): Determines the unit value to output. Could
                be `miles` or `kilometers`. Default is `miles`.

        Examples:
            >>> haversine = Haversine()
            >>> distances = haversine([(42.4, -71.1), (40.0, -122.4)],
            ...                       [(40.0, -122.4), (41.2, -96.75)])
            >>> np.round(distances, 3).tolist()
            [2631.231, 1343.289]

            Output units can be specified

            >>> haversine_km = Haversine(unit='kilometers')
            >>> distances_km = haversine_km([(42.4, -71.1), (40.0, -122.4)],
            ...                             [(40.0, -122.4), (41.2, -96.75)])
            >>> np.round(distances_km, 3).tolist()
            [4234.555, 2161.814]
    """
    name = 'haversine'
    input_types = [ColumnSchema(logical_type=LatLong), ColumnSchema(logical_type=LatLong)]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    commutative = True

    def __init__(self, unit='miles'):
        valid_units = ['miles', 'kilometers']
        if unit not in valid_units:
            error_message = 'Invalid unit %s provided. Must be one of %s' % (unit, valid_units)
            raise ValueError(error_message)
        self.unit = unit
        self.description_template = "the haversine distance in {} between {{}} and {{}}".format(self.unit)

    def get_function(self):
        def haversine(latlong_1, latlong_2):
            lat_1s, lon_1s = _deconstrct_latlongs(latlong_1)
            lat_2s, lon_2s = _deconstrct_latlongs(latlong_2)
            distance = _haversine_calculate(lat_1s, lon_1s, lat_2s, lon_2s, self.unit)
            return distance
        return haversine

    def generate_name(self, base_feature_names):
        name = u"{}(".format(self.name.upper())
        name += u", ".join(base_feature_names)
        if self.unit != 'miles':
            name += u", unit={}".format(self.unit)
        name += u")"
        return name


class IsInGeoBox(TransformPrimitive):
    """Determines if coordinates are inside a box defined by two
    corner coordinate points.

    Description:
        Coordinate values should be specified as (latitude, longitude)
        tuples. This primitive is unable to handle coordinates and boxes
        at the poles, and near +/- 180 degrees latitude.

    Args:
        point1 (tuple(float, float)): The coordinates
            of the first corner of the box. Defaults to (0, 0).
        point2 (tuple(float, float)): The coordinates
            of the diagonal corner of the box. Defaults to (0, 0).

    Example:
        >>> is_in_geobox = IsInGeoBox((40.7128, -74.0060), (42.2436, -71.1677))
        >>> is_in_geobox([(41.034, -72.254), (39.125, -87.345)]).tolist()
        [True, False]
    """
    name = "is_in_geobox"
    input_types = [ColumnSchema(logical_type=LatLong)]
    return_type = ColumnSchema(logical_type=BooleanNullable)

    def __init__(self, point1=(0, 0), point2=(0, 0)):
        self.point1 = point1
        self.point2 = point2
        self.lats = np.sort(np.array([point1[0], point2[0]]))
        self.lons = np.sort(np.array([point1[1], point2[1]]))

    def get_function(self):
        def geobox(latlongs):
            if latlongs.hasnans:
                latlongs = np.where(latlongs.isnull(), pd.Series([(np.nan, np.nan)] * len(latlongs)), latlongs)
            transposed = np.transpose([list(latlon) for latlon in latlongs])
            lats = (self.lats[0] <= transposed[0]) & \
                   (self.lats[1] >= transposed[0])
            longs = (self.lons[0] <= transposed[1]) & \
                    (self.lons[1] >= transposed[1])
            return lats & longs
        return geobox


class Latitude(TransformPrimitive):
    """Returns the first tuple value in a list of LatLong tuples.
       For use with the LatLong logical type.

    Examples:
        >>> latitude = Latitude()
        >>> latitude([(42.4, -71.1),
        ...            (40.0, -122.4),
        ...            (41.2, -96.75)]).tolist()
        [42.4, 40.0, 41.2]
    """
    name = 'latitude'
    input_types = [ColumnSchema(logical_type=LatLong)]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    description_template = "the latitude of {}"

    def get_function(self):
        def latitude(latlong):
            return latlong.map(lambda x: x[0] if isinstance(x, tuple) else np.nan)
        return latitude


class Longitude(TransformPrimitive):
    """Returns the second tuple value in a list of LatLong tuples.
       For use with the LatLong logical type.

    Examples:
        >>> longitude = Longitude()
        >>> longitude([(42.4, -71.1),
        ...            (40.0, -122.4),
        ...            (41.2, -96.75)]).tolist()
        [-71.1, -122.4, -96.75]
    """
    name = 'longitude'
    input_types = [ColumnSchema(logical_type=LatLong)]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    description_template = "the longitude of {}"

    def get_function(self):
        def longitude(latlong):
            return latlong.map(lambda x: x[1] if isinstance(x, tuple) else np.nan)
        return longitude
