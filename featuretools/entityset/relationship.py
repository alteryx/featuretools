class Relationship(object):
    """Class to represent an relationship between entities

    See Also:
        :class:`.EntitySet`, :class:`.Entity`
    """

    def __init__(self, entityset, parent_dataframe_id, parent_column_id,
                 child_dataframe_id, child_column_id):
        """ Create a relationship

        Args:
            entityset (:class:`.EntitySet`): EntitySet to which the relationship belongs
            parent_dataframe_id (str): Name of the parent dataframe in the EntitySet
            parent_column_id (str): Name of the parent column
            child_dataframe_id (str): Name of the child dataframe in the EntitySet
            child_column_id (str): Name of the child column
        """

        self.entityset = entityset
        self._parent_dataframe_id = parent_dataframe_id
        self._child_dataframe_id = child_dataframe_id
        self._parent_column_id = parent_column_id
        self._child_column_id = child_column_id

        if (self.parent_dataframe.ww.index is not None and
                self._parent_column_id != self.parent_dataframe.ww.index):
            raise AttributeError(f"Parent column '{self.parent_column}' is not the index of "
                                 f"dataframe {self.parent_dataframe}")

    @classmethod
    def from_dictionary(cls, arguments, es):
        parent_dataframe = arguments['parent_dataframe_id']
        child_dataframe = arguments['child_dataframe_id']
        parent_column = arguments['parent_column_id']
        child_column = arguments['child_column_id']
        return cls(es, parent_dataframe, parent_column, child_dataframe, child_column)

    def __repr__(self):
        ret = u"<Relationship: %s.%s -> %s.%s>" % \
            (self._child_dataframe_id, self._child_column_id,
             self._parent_dataframe_id, self._parent_column_id)

        return ret

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self._parent_dataframe_id == other._parent_dataframe_id and \
            self._child_dataframe_id == other._child_dataframe_id and \
            self._parent_column_id == other._parent_column_id and \
            self._child_column_id == other._child_column_id

    def __hash__(self):
        return hash((self._parent_dataframe_id,
                     self._child_dataframe_id,
                     self._parent_column_id,
                     self._child_column_id))

    @property
    def parent_dataframe(self):
        """Parent dataframe object"""
        return self.entityset[self._parent_dataframe_id]

    @property
    def child_dataframe(self):
        """Child dataframe object"""
        return self.entityset[self._child_dataframe_id]

    @property
    def parent_column(self):
        """Column in parent dataframe"""
        return self.parent_dataframe[self._parent_column_id]

    @property
    def child_column(self):
        """Column in child dataframe"""
        return self.child_dataframe[self._child_column_id]

    @property
    def parent_name(self):
        """The name of the parent, relative to the child."""
        if self._is_unique():
            return self._parent_dataframe_id
        else:
            return '%s[%s]' % (self._parent_dataframe_id, self._child_column_id)

    @property
    def child_name(self):
        """The name of the child, relative to the parent."""
        if self._is_unique():
            return self._child_dataframe_id
        else:
            return '%s[%s]' % (self._child_dataframe_id, self._child_column_id)

    def to_dictionary(self):
        return {
            'parent_dataframe_id': self._parent_dataframe_id,
            'child_dataframe_id': self._child_dataframe_id,
            'parent_column_id': self._parent_column_id,
            'child_column_id': self._child_column_id,
        }

    def _is_unique(self):
        """Is there any other relationship with same parent and child entities?"""
        es = self.entityset
        relationships = es.get_forward_relationships(self._child_dataframe_id)
        n = len([r for r in relationships
                 if r._parent_dataframe_id == self._parent_dataframe_id])

        assert n > 0, 'This relationship is missing from the entityset'

        return n == 1


class RelationshipPath(object):
    def __init__(self, relationships_with_direction):
        self._relationships_with_direction = relationships_with_direction

    @property
    def name(self):
        relationship_names = [_direction_name(is_forward, r)
                              for is_forward, r in self._relationships_with_direction]

        return '.'.join(relationship_names)

    def entities(self):
        if self:
            # Yield first dataframe.
            is_forward, relationship = self[0]
            if is_forward:
                yield relationship.child_dataframe.ww.name
            else:
                yield relationship.parent_dataframe.ww.name

        # Yield the dataframe pointed to by each relationship.
        for is_forward, relationship in self:
            if is_forward:
                yield relationship.parent_dataframe.ww.name
            else:
                yield relationship.child_dataframe.ww.name

    def __add__(self, other):
        return RelationshipPath(self._relationships_with_direction +
                                other._relationships_with_direction)

    def __getitem__(self, index):
        return self._relationships_with_direction[index]

    def __iter__(self):
        for is_forward, relationship in self._relationships_with_direction:
            yield is_forward, relationship

    def __len__(self):
        return len(self._relationships_with_direction)

    def __eq__(self, other):
        return isinstance(other, RelationshipPath) and \
            self._relationships_with_direction == other._relationships_with_direction

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        if self._relationships_with_direction:
            path = '%s.%s' % (next(self.entities()), self.name)
        else:
            path = '[]'
        return '<RelationshipPath %s>' % path


def _direction_name(is_forward, relationship):
    if is_forward:
        return relationship.parent_name
    else:
        return relationship.child_name
