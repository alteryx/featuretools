import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import LatLong

from featuretools.primitives.base import TransformPrimitive
from featuretools.primitives.standard.transform.latlong.utils import (
    _haversine_calculate,
)


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

    name = "haversine"
    input_types = [
        ColumnSchema(logical_type=LatLong),
        ColumnSchema(logical_type=LatLong),
    ]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    commutative = True

    def __init__(self, unit="miles"):
        valid_units = ["miles", "kilometers"]
        if unit not in valid_units:
            error_message = "Invalid unit %s provided. Must be one of %s" % (
                unit,
                valid_units,
            )
            raise ValueError(error_message)
        self.unit = unit
        self.description_template = (
            "the haversine distance in {} between {{}} and {{}}".format(self.unit)
        )

    def get_function(self):
        def haversine(latlong_1, latlong_2):
            latlong_1 = np.array(latlong_1.tolist())
            latlong_2 = np.array(latlong_2.tolist())
            lat_1s = latlong_1[:, 0]
            lat_2s = latlong_2[:, 0]
            lon_1s = latlong_1[:, 1]
            lon_2s = latlong_2[:, 1]

            distance = _haversine_calculate(lat_1s, lon_1s, lat_2s, lon_2s, self.unit)
            return distance

        return haversine

    def generate_name(self, base_feature_names):
        name = "{}(".format(self.name.upper())
        name += ", ".join(base_feature_names)
        if self.unit != "miles":
            name += ", unit={}".format(self.unit)
        name += ")"
        return name
