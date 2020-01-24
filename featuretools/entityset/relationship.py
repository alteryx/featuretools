class Relationship(object):
    """Class to represent an relationship between entities

    See Also:
        :class:`.EntitySet`, :class:`.Entity`, :class:`.Variable`
    """

    def __init__(self, parent_variable, child_variable):
        """ Create a relationship

        Args:
            parent_variable (:class:`.Discrete`): Instance of variable
                in parent entity.  Must be a Discrete Variable
            child_variable (:class:`.Discrete`): Instance of variable in
                child entity.  Must be a Discrete Variable

        """

        self.entityset = child_variable.entityset
        self._parent_entity_id = parent_variable.entity.id
        self._child_entity_id = child_variable.entity.id
        self._parent_variable_id = parent_variable.id
        self._child_variable_id = child_variable.id

        if (parent_variable.entity.index is not None and
                parent_variable.id != parent_variable.entity.index):
            raise AttributeError("Parent variable '%s' is not the index of entity %s" % (parent_variable, parent_variable.entity))

    @classmethod
    def from_dictionary(cls, arguments, es):
        parent_entity = es[arguments['parent_entity_id']]
        child_entity = es[arguments['child_entity_id']]
        parent_variable = parent_entity[arguments['parent_variable_id']]
        child_variable = child_entity[arguments['child_variable_id']]
        return cls(parent_variable, child_variable)

    def __repr__(self):
        ret = u"<Relationship: %s.%s -> %s.%s>" % \
            (self._child_entity_id, self._child_variable_id,
             self._parent_entity_id, self._parent_variable_id)

        return ret

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self._parent_entity_id == other._parent_entity_id and \
            self._child_entity_id == other._child_entity_id and \
            self._parent_variable_id == other._parent_variable_id and \
            self._child_variable_id == other._child_variable_id

    def __hash__(self):
        return hash((self._parent_entity_id,
                     self._child_entity_id,
                     self._parent_variable_id,
                     self._child_variable_id))

    @property
    def parent_entity(self):
        """Parent entity object"""
        return self.entityset[self._parent_entity_id]

    @property
    def child_entity(self):
        """Child entity object"""
        return self.entityset[self._child_entity_id]

    @property
    def parent_variable(self):
        """Instance of variable in parent entity"""
        return self.parent_entity[self._parent_variable_id]

    @property
    def child_variable(self):
        """Instance of variable in child entity"""
        return self.child_entity[self._child_variable_id]

    @property
    def parent_name(self):
        """The name of the parent, relative to the child."""
        if self._is_unique():
            return self._parent_entity_id
        else:
            return '%s[%s]' % (self._parent_entity_id, self._child_variable_id)

    @property
    def child_name(self):
        """The name of the child, relative to the parent."""
        if self._is_unique():
            return self._child_entity_id
        else:
            return '%s[%s]' % (self._child_entity_id, self._child_variable_id)

    def to_dictionary(self):
        return {
            'parent_entity_id': self._parent_entity_id,
            'child_entity_id': self._child_entity_id,
            'parent_variable_id': self._parent_variable_id,
            'child_variable_id': self._child_variable_id,
        }

    def _is_unique(self):
        """Is there any other relationship with same parent and child entities?"""
        es = self.child_entity.entityset
        relationships = es.get_forward_relationships(self._child_entity_id)
        n = len([r for r in relationships
                 if r._parent_entity_id == self._parent_entity_id])

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
            # Yield first entity.
            is_forward, relationship = self[0]
            if is_forward:
                yield relationship.child_entity.id
            else:
                yield relationship.parent_entity.id

        # Yield the entity pointed to by each relationship.
        for is_forward, relationship in self:
            if is_forward:
                yield relationship.parent_entity.id
            else:
                yield relationship.child_entity.id

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
