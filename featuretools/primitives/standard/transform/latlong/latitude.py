import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import LatLong

from featuretools.primitives.base import TransformPrimitive


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

    name = "latitude"
    input_types = [ColumnSchema(logical_type=LatLong)]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    description_template = "the latitude of {}"

    def get_function(self):
        def latitude(latlong):
            latlong = np.array(latlong.tolist())
            return latlong[:, 0]

        return latitude
