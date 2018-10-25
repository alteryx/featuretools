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

    def __repr__(self):
        ret = u"<Relationship: %s.%s -> %s.%s>" % \
            (self._child_entity_id, self._child_variable_id,
             self._parent_entity_id, self._parent_variable_id)

        # encode for python 2
        if type(ret) != str:
            ret = ret.encode("utf-8")

        return ret

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
