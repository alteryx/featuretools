class Relationship(object):
    """Class to represent a relationship between dataframes

    See Also:
        :class:`.EntitySet`
    """

    def __init__(
        self,
        entityset,
        parent_dataframe_name,
        parent_column_name,
        child_dataframe_name,
        child_column_name,
    ):
        """Create a relationship

        Args:
            entityset (:class:`.EntitySet`): EntitySet to which the relationship belongs
            parent_dataframe_name (str): Name of the parent dataframe in the EntitySet
            parent_column_name (str): Name of the parent column
            child_dataframe_name (str): Name of the child dataframe in the EntitySet
            child_column_name (str): Name of the child column
        """

        self.entityset = entityset
        self._parent_dataframe_name = parent_dataframe_name
        self._child_dataframe_name = child_dataframe_name
        self._parent_column_name = parent_column_name
        self._child_column_name = child_column_name

        if (
            self.parent_dataframe.ww.index is not None
            and self._parent_column_name != self.parent_dataframe.ww.index
        ):
            raise AttributeError(
                f"Parent column '{self._parent_column_name}' is not the index of "
                f"dataframe {self._parent_dataframe_name}",
            )

    @classmethod
    def from_dictionary(cls, arguments, es):
        parent_dataframe = arguments["parent_dataframe_name"]
        child_dataframe = arguments["child_dataframe_name"]
        parent_column = arguments["parent_column_name"]
        child_column = arguments["child_column_name"]
        return cls(es, parent_dataframe, parent_column, child_dataframe, child_column)

    def __repr__(self):
        ret = "<Relationship: %s.%s -> %s.%s>" % (
            self._child_dataframe_name,
            self._child_column_name,
            self._parent_dataframe_name,
            self._parent_column_name,
        )

        return ret

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return (
            self._parent_dataframe_name == other._parent_dataframe_name
            and self._child_dataframe_name == other._child_dataframe_name
            and self._parent_column_name == other._parent_column_name
            and self._child_column_name == other._child_column_name
        )

    def __hash__(self):
        return hash(
            (
                self._parent_dataframe_name,
                self._child_dataframe_name,
                self._parent_column_name,
                self._child_column_name,
            ),
        )

    @property
    def parent_dataframe(self):
        """Parent dataframe object"""
        return self.entityset[self._parent_dataframe_name]

    @property
    def child_dataframe(self):
        """Child dataframe object"""
        return self.entityset[self._child_dataframe_name]

    @property
    def parent_column(self):
        """Column in parent dataframe"""
        return self.parent_dataframe.ww[self._parent_column_name]

    @property
    def child_column(self):
        """Column in child dataframe"""
        return self.child_dataframe.ww[self._child_column_name]

    @property
    def parent_name(self):
        """The name of the parent, relative to the child."""
        if self._is_unique():
            return self._parent_dataframe_name
        else:
            return "%s[%s]" % (self._parent_dataframe_name, self._child_column_name)

    @property
    def child_name(self):
        """The name of the child, relative to the parent."""
        if self._is_unique():
            return self._child_dataframe_name
        else:
            return "%s[%s]" % (self._child_dataframe_name, self._child_column_name)

    def to_dictionary(self):
        return {
            "parent_dataframe_name": self._parent_dataframe_name,
            "child_dataframe_name": self._child_dataframe_name,
            "parent_column_name": self._parent_column_name,
            "child_column_name": self._child_column_name,
        }

    def _is_unique(self):
        """Is there any other relationship with same parent and child dataframes?"""
        es = self.entityset
        relationships = es.get_forward_relationships(self._child_dataframe_name)
        n = len(
            [
                r
                for r in relationships
                if r._parent_dataframe_name == self._parent_dataframe_name
            ],
        )

        assert n > 0, "This relationship is missing from the entityset"

        return n == 1


class RelationshipPath(object):
    def __init__(self, relationships_with_direction):
        self._relationships_with_direction = relationships_with_direction

    @property
    def name(self):
        relationship_names = [
            _direction_name(is_forward, r)
            for is_forward, r in self._relationships_with_direction
        ]

        return ".".join(relationship_names)

    def dataframes(self):
        if self:
            # Yield first dataframe.
            is_forward, relationship = self[0]
            if is_forward:
                yield relationship._child_dataframe_name
            else:
                yield relationship._parent_dataframe_name

        # Yield the dataframe pointed to by each relationship.
        for is_forward, relationship in self:
            if is_forward:
                yield relationship._parent_dataframe_name
            else:
                yield relationship._child_dataframe_name

    def __add__(self, other):
        return RelationshipPath(
            self._relationships_with_direction + other._relationships_with_direction,
        )

    def __getitem__(self, index):
        return self._relationships_with_direction[index]

    def __iter__(self):
        for is_forward, relationship in self._relationships_with_direction:
            yield is_forward, relationship

    def __len__(self):
        return len(self._relationships_with_direction)

    def __eq__(self, other):
        return (
            isinstance(other, RelationshipPath)
            and self._relationships_with_direction
            == other._relationships_with_direction
        )

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        if self._relationships_with_direction:
            path = "%s.%s" % (next(self.dataframes()), self.name)
        else:
            path = "[]"
        return "<RelationshipPath %s>" % path


def _direction_name(is_forward, relationship):
    if is_forward:
        return relationship.parent_name
    else:
        return relationship.child_name
