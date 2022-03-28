from featuretools.primitives.base.primitive_base import PrimitiveBase


class AggregationPrimitive(PrimitiveBase):
    stack_on = None  # whitelist of primitives that can be in input_types
    stack_on_exclude = None  # blacklist of primitives that can be insigniture
    base_of = None  # whitelist of primitives this prim can be input for
    base_of_exclude = None  # primitives this primitive can't be input for
    stack_on_self = True  # whether or not it can be in input_types of self

    def generate_name(
        self,
        base_feature_names,
        relationship_path_name,
        parent_dataframe_name,
        where_str,
        use_prev_str,
    ):
        base_features_str = ", ".join(base_feature_names)
        return "%s(%s.%s%s%s%s)" % (
            self.name.upper(),
            relationship_path_name,
            base_features_str,
            where_str,
            use_prev_str,
            self.get_args_string(),
        )

    def generate_names(
        self,
        base_feature_names,
        relationship_path_name,
        parent_dataframe_name,
        where_str,
        use_prev_str,
    ):
        n = self.number_output_features
        base_name = self.generate_name(
            base_feature_names,
            relationship_path_name,
            parent_dataframe_name,
            where_str,
            use_prev_str,
        )
        return [base_name + "[%s]" % i for i in range(n)]
