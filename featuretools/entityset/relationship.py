from featuretools.core.base import FTBase


class Relationship(FTBase):
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

    def __repr__(self):
        return "<Relationship: %s.%s -> %s.%s>" % \
            (self._child_entity_id, self._child_variable_id,
             self._parent_entity_id, self._parent_variable_id)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self._parent_entity_id == other._parent_entity_id and \
            self._child_entity_id == other._child_entity_id and \
            self._parent_variable_id == other._parent_variable_id and \
            self._child_variable_id == other._child_variable_id

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

    def get_entity_variable(self, entity_id):
        if self._child_entity_id == entity_id:
            return self._child_variable_id
        if self._parent_entity_id == entity_id:
            return self._parent_variable_id
        raise AttributeError("Entity '%s' is not part of relationship" %
                             entity_id)

    def get_other_entity(self, entity_id):
        if self._child_entity_id == entity_id:
            return self._parent_entity_id
        if self._parent_entity_id == entity_id:
            return self._child_entity_id
        raise AttributeError("Entity '%s' is not part of relationship" %
                             entity_id)

    def get_other_variable(self, variable_id):
        if self._child_variable_id == variable_id:
            return self._parent_variable_id
        if self._parent_variable_id == variable_id:
            return self._child_variable_id
        raise AttributeError("Variable '%s' is not part of relationship" %
                             variable_id)

    @classmethod
    def _get_link_variable_name(cls, path):
        r = path[0]
        child_link_name = r.child_variable.id
        for r in path[1:]:
            parent_link_name = child_link_name
            child_link_name = '%s.%s' % (r.parent_variable.entity.id,
                                         parent_link_name)
        return child_link_name
