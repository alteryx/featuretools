import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double, LatLong

from featuretools.primitives.base import TransformPrimitive
from featuretools.primitives.standard.transform.latlong.utils import (
    _haversine_calculate,
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
    input_types = [
        ColumnSchema(logical_type=LatLong),
        ColumnSchema(logical_type=LatLong),
    ]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    commutative = True

    def __init__(self, unit="miles"):
        if unit not in ["miles", "kilometers"]:
            raise ValueError("Invalid unit given")
        self.unit = unit

    def get_function(self):
        def cityblock(latlong_1, latlong_2):
            latlong_1 = np.array(latlong_1.tolist())
            latlong_2 = np.array(latlong_2.tolist())
            lat_1s = latlong_1[:, 0]
            lat_2s = latlong_2[:, 0]
            lon_1s = latlong_1[:, 1]
            lon_2s = latlong_2[:, 1]
            lon_dis = _haversine_calculate(lat_1s, lon_1s, lat_1s, lon_2s, self.unit)
            lat_dist = _haversine_calculate(lat_1s, lon_1s, lat_2s, lon_1s, self.unit)
            return pd.Series(lon_dis + lat_dist)

        return cityblock
