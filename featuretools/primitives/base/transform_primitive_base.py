import copy
import functools
import inspect

from featuretools.primitives.base.primitive_base import PrimitiveBase
from featuretools.primitives.base.utils import inspect_function_args


class TransformPrimitive(PrimitiveBase):
    """Feature for dataframe that is a based off one or more other features
        in that dataframe."""
    # (bool) If True, feature function depends on all values of dataframe
    #   (and will receive these values as input, regardless of specified instance ids)
    uses_full_dataframe = False

    def generate_name(self, base_feature_names):
        return u"%s(%s%s)" % (
            self.name.upper(),
            u", ".join(base_feature_names),
            self.get_args_string(),
        )

    def generate_names(self, base_feature_names):
        n = self.number_output_features
        base_name = self.generate_name(base_feature_names)
        return [base_name + "[%s]" % i for i in range(n)]
