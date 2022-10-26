import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable, LatLong

from featuretools.primitives.base import TransformPrimitive


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
            transposed = np.transpose(np.array(latlongs.tolist()))
            lats = (self.lats[0] <= transposed[0]) & (self.lats[1] >= transposed[0])
            longs = (self.lons[0] <= transposed[1]) & (self.lons[1] >= transposed[1])
            return lats & longs

        return geobox
