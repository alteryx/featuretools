from .primitive_base import PrimitiveBase


class AggregationPrimitive(PrimitiveBase):
    """Feature for a parent entity that summarizes
        related instances in a child entity"""
    stack_on = None # whitelist of primitives that can be in input_types
    stack_on_exclude = None # blacklist of primitives that can be insigniture
    base_of = None  # whitelist of primitives can have this primitive in input_types
    base_of_exclude = None # blacklist of primitives can have this primitive in input_types
    stack_on_self = True# whether or not it can be in input_types of self
    allow_where = True # whether or not DFS can apply where clause to this primitive

    def __init__(self, base_features, parent_entity, use_previous=None,
                 where=None):
        if not hasattr(base_features, '__iter__'):
            base_features = [self._check_feature(base_features)]
        else:
            base_features = [self._check_feature(bf) for bf in base_features]
            msg = "all base features must share the same entity"
            assert len(set([bf.entity for bf in base_features])) == 1, msg
        self.base_features = base_features[:]

        self.child_entity = base_features[0].entity

        if where is not None:
            self.where = self._check_feature(where)
            msg = "Where feature must be defined on child entity {}".format(self.child_entity.id)
            assert self.where.entity.id == self.child_entity.id, msg

        if use_previous:
            assert self.child_entity.has_time_index(), (
                "Applying function that requires time index to entity that "
                "doesn't have one")

        self.use_previous = use_previous

        super(AggregationPrimitive, self).__init__(parent_entity,
                                                   self.base_features)

    def _where_str(self):
        if self.where is not None:
            where_str = u" WHERE " + self.where.get_name()
        else:
            where_str = ''
        return where_str

    def _use_prev_str(self):
        if self.use_previous is not None:
            use_prev_str = u", Last {}".format(self.use_previous.get_name())
        else:
            use_prev_str = u''
        return use_prev_str

    def _base_feature_str(self):
        return u', ' \
            .join([bf.get_name() for bf in self.base_features])

    def _get_name(self):
        where_str = self._where_str()
        use_prev_str = self._use_prev_str()

        base_features_str = self._base_feature_str()

        return u"%s(%s.%s%s%s)" % (self.name.upper(),
                                   self.child_entity.name,
                                   base_features_str,
                                   where_str, use_prev_str)
