import pandas as pd
from woodwork.column_schema import ColumnSchema
from featuretools.primitives.base.transform_primitive_base import TransformPrimitive


class AddNumericScalar(TransformPrimitive):
    """Adds a scalar or pandas Timedelta to each value in the list.

    Description:
        Given a list/column of numeric or datetime values and a scalar (numeric or pandas Timedelta),
        this primitive adds the scalar to each value in the list.

    Examples:
        >>> add_numeric_scalar = AddNumericScalar(value=2)
        >>> add_numeric_scalar([3, 1, 2]).tolist()
        [5, 3, 4]

        >>> import pandas as pd
        >>> add_timedelta = AddNumericScalar(value=pd.Timedelta(days=365))
        >>> add_timedelta(pd.to_datetime(["2020-01-01", "2021-01-01"])).tolist()
        [Timestamp('2020-12-31 00:00:00'), Timestamp('2021-12-31 00:00:00')]
    """

    name = "add_numeric_scalar"
    input_types = [
        ColumnSchema(semantic_tags={"numeric"}),
        ColumnSchema(semantic_tags={"datetime"})
    ]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    commutative = True

    def __init__(self, value=0):
        self.value = value
        self.description_template = "the sum of {{}} and {}".format(self.value)

    def get_function(self):
        def add_scalar(vals):
            # ✅ Handle datetime + pandas.Timedelta
            if isinstance(self.value, pd.Timedelta):
                return vals + self.value
            # ✅ Handle numeric or datetime + numeric (default behavior)
            return vals + self.value

        return add_scalar

    def generate_name(self, base_feature_names):
        return "%s + %s" % (base_feature_names[0], str(self.value))
