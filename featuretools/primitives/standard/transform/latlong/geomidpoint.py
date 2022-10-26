import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import LatLong

from featuretools.primitives.base import TransformPrimitive


class GeoMidpoint(TransformPrimitive):
    """Determines the geographic center of two coordinates.

    Examples:
        >>> geomidpoint = GeoMidpoint()
        >>> geomidpoint([(42.4, -71.1)], [(40.0, -122.4)])
        [(41.2, -96.75)]
    """

    name = "geomidpoint"
    input_types = [
        ColumnSchema(logical_type=LatLong),
        ColumnSchema(logical_type=LatLong),
    ]
    return_type = ColumnSchema(logical_type=LatLong)
    commutative = True

    def get_function(self):
        def geomidpoint_func(latlong_1, latlong_2):
            latlong_1 = np.array(latlong_1.tolist())
            latlong_2 = np.array(latlong_2.tolist())
            lat_1s = latlong_1[:, 0]
            lat_2s = latlong_2[:, 0]
            lon_1s = latlong_1[:, 1]
            lon_2s = latlong_2[:, 1]

            lat_middle = np.array([lat_1s, lat_2s]).transpose().mean(axis=1)
            lon_middle = np.array([lon_1s, lon_2s]).transpose().mean(axis=1)
            return list(zip(lat_middle, lon_middle))

        return geomidpoint_func
