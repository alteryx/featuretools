import logging
from builtins import object

from featuretools import variable_types as vtypes
from featuretools.core.base import FTBase

logger = logging.getLogger('featuretools.entityset')


class BFSNode(object):

    def __init__(self, entity_id, parent, relationship):
        self.entity_id = entity_id
        self.parent = parent
        self.relationship = relationship

    def build_path(self):
        path = []
        cur_node = self
        num_forward = 0
        i = 0
        last_forward = False
        while cur_node.parent is not None:
            path.append(cur_node.relationship)
            if cur_node.relationship.parent_entity.id == cur_node.entity_id:
                num_forward += 1
                if i == 0:
                    last_forward = True
            cur_node = cur_node.parent
            i += 1
        path.reverse()

        # if path ends on a forward relationship, return number of
        # forward relationships, otherwise 0
        if len(path) == 0 or not last_forward:
            num_forward = 0
        return path, num_forward


class BaseEntitySet(FTBase):
    """
    Stores all actual data for a entityset
    """
    id = None
    entities = []
    relationships = []
    name = None

    def __init__(self, id, verbose):
        self.id = id
        self.last_saved = None
        self.entity_stores = {}
        self.relationships = []
        self._verbose = verbose
        self.time_type = None

    def __eq__(self, other, deep=False):
        self_to_compare = self
        if not deep:
            if not isinstance(other, type(self)):
                return False
            if not self.is_metadata:
                self_to_compare = self.metadata
            if not other.is_metadata:
                other = other.metadata
            return EntitySet.compare_entitysets(self_to_compare, other)
        else:
            return EntitySet.compare_entitysets(self, other)

    @classmethod
    def compare_entitysets(cls, es1, es2):
        if len(es1.entity_stores) != len(es2.entity_stores):
            return False
        for eid, e in es1.entity_stores.items():
            if eid not in es2.entity_stores:
                return False
            if not Entity.compare_entities(e, other[eid]):
                return False
        for r in es2.relationships:
            if r not in es2.relationships:
                return False
        return True

    def __getitem__(self, entity_id):
        """Get entity instance from entityset

        Args:
            entity_id (str): Id of entity.

        Returns:
            :class:`.Entity` : Instance of entity. None if entity doesn't
                exist.

        Example:
            >>> my_entityset[entity_id]
            <Entity: id>
        """
        return self._get_entity(entity_id)

    @property
    def entities(self):
        return list(self.entity_stores.values())

    def _get_entity(self, entity_id):
        """Get entity instance from entityset

        Args:
            entity_id (str) : Id of entity.

        Returns:
            :class:`.Entity` : Instance of entity. None if entity doesn't exist.
        """
        if entity_id in self.entity_stores:
            return self.entity_stores[entity_id]

        raise KeyError('Entity %s does not exist in %s' % (entity_id, self.id))

    ###########################################################################
    #   Public getter/setter methods  #########################################
    ###########################################################################

    def get_name(self):
        """Returns name of entityset

        If name is None, return the id

        Returns:
            str : name of the entityset
        """
        name = self.name
        if name is None:
            name = self.id
        return name

    def get_dataframe(self, entity_id, entityset):
        """Returns dataframe of entity"""
        return entityset.get_dataframe(entity_id)

    def __repr__(self):
        fmat = self.id
        repr_out = u"Entityset: {}\n".format(fmat)
        repr_out += u"  Entities:"
        for e in self.entities:
            if e.df.shape:
                repr_out += u"\n    {} [Rows: {}, Columns: {}]".format(
                    e.id, e.df.shape[0], e.df.shape[1])
            else:
                repr_out += u"\n    {} [Rows: None, Columns: None]".format(
                    e.id)
        repr_out += "\n  Relationships:"

        if len(self.relationships) == 0:
            repr_out += "\n    No relationships"

        for r in self.relationships:
            repr_out += u"\n    %s.%s -> %s.%s" % \
                (r._child_entity_id, r._child_variable_id,
                 r._parent_entity_id, r._parent_variable_id)

        return repr_out

    def delete_entity_variables(self, entity_id, variables, **kwargs):
        entity = self._get_entity(entity_id)
        for v in variables:
            entity.delete_variable(v)

    def make_index_variable_name(self, entity_id):
        return entity_id + "_id"

    def add_relationships(self, relationships):
        """Add multiple new relationships to a entityset

        Args:
            relationships (list[Relationship]) : List of new
                relationships.
        """
        new_self = self
        for r in relationships:
            new_self = new_self.add_relationship(r)
        return new_self

    def add_relationship(self, relationship):
        """Add a new relationship between entities in the entityset

        Args:
            relationship (Relationship) : Instance of new
                relationship to be added.
        """
        if relationship in self.relationships:
            logger.warning(
                "Not adding duplicate relationship: %s", relationship)
            return self

        # _operations?

        # this is a new pair of entities
        child_e = relationship.child_entity
        child_v = relationship.child_variable.id
        parent_e = relationship.parent_entity
        parent_v = relationship.parent_variable.id
        if not isinstance(self[child_e.id][child_v], vtypes.Discrete):
            child_e.convert_variable_type(variable_id=child_v,
                                          new_type=vtypes.Id,
                                          convert_data=False)
        if not isinstance(self[parent_e.id][parent_v], vtypes.Discrete):
            parent_e.convert_variable_type(variable_id=parent_v,
                                           new_type=vtypes.Index,
                                           convert_data=False)

        self.relationships.append(relationship)
        self.index_data(relationship)
        return self

    ###########################################################################
    #   Relationship access/helper methods  ###################################
    ###########################################################################

    def find_path(self, start_entity_id, goal_entity_id,
                  include_num_forward=False):
        """Find a path in the entityset represented as a DAG
           between start_entity and goal_entity

        Args:
            start_entity_id (str) : Id of entity to start the search from.
            goal_entity_id  (str) : Id of entity to find forward path to.
            include_num_forward (bool) : If True, return number of forward
                relationships in path if the path ends on a forward
                relationship, otherwise return 0.

        Returns:
            List of relationships that go from start entity to goal
                entity. None is returned if no path exists.
            If include_forward_distance is True,
                returns a tuple of (relationship_list, forward_distance).

        See Also:
            :func:`BaseEntitySet.find_forward_path`
            :func:`BaseEntitySet.find_backward_path`
        """
        if start_entity_id == goal_entity_id:
            if include_num_forward:
                return [], 0
            else:
                return []

        # BFS so we get shortest path
        start_node = BFSNode(start_entity_id, None, None)
        queue = [start_node]
        nodes = {}

        while len(queue) > 0:
            current_node = queue.pop(0)
            if current_node.entity_id == goal_entity_id:
                path, num_forward = current_node.build_path()
                if include_num_forward:
                    return path, num_forward
                else:
                    return path

            for r in self.get_forward_relationships(current_node.entity_id):
                if r.parent_entity.id not in nodes:
                    parent_node = BFSNode(r.parent_entity.id, current_node, r)
                    nodes[r.parent_entity.id] = parent_node
                    queue.append(parent_node)

            for r in self.get_backward_relationships(current_node.entity_id):
                if r.child_entity.id not in nodes:
                    child_node = BFSNode(r.child_entity.id, current_node, r)
                    nodes[r.child_entity.id] = child_node
                    queue.append(child_node)

        raise ValueError(("No path from {} to {}! Check that all entities "
                          .format(start_entity_id, goal_entity_id)),
                         "are connected by relationships")

    def find_forward_path(self, start_entity_id, goal_entity_id):
        """Find a forward path between a start and goal entity

        Args:
            start_entity_id (str) : id of entity to start the search from
            goal_entity_id  (str) : if of entity to find forward path to

        Returns:
            List of relationships that go from start entity to goal
                entity. None is return if no path exists

        See Also:
            :func:`BaseEntitySet.find_backward_path`
            :func:`BaseEntitySet.find_path`
        """

        if start_entity_id == goal_entity_id:
            return []

        for r in self.get_forward_relationships(start_entity_id):
            new_path = self.find_forward_path(
                r.parent_entity.id, goal_entity_id)
            if new_path is not None:
                return [r] + new_path

        return None

    def find_backward_path(self, start_entity_id, goal_entity_id):
        """Find a backward path between a start and goal entity

        Args:
            start_entity_id (str) : Id of entity to start the search from.
            goal_entity_id  (str) : Id of entity to find backward path to.

        See Also:
            :func:`BaseEntitySet.find_forward_path`
            :func:`BaseEntitySet.find_path`

        Returns:
            List of relationship that go from start entity to goal entity. None
            is returned if no path exists.
        """
        forward_path = self.find_forward_path(goal_entity_id, start_entity_id)
        if forward_path is not None:
            return forward_path[::-1]
        return None

    def get_forward_entities(self, entity_id, deep=False):
        """Get entities that are in a forward relationship with entity

        Args:
            entity_id (str) - Id entity of entity to search from.
            deep (bool) - if True, recursively find forward entities.

        Returns:
            Set of entity IDs in a forward relationship with the passed in
            entity.
        """
        parents = [r.parent_entity.id for r in
                   self.get_forward_relationships(entity_id)]

        if deep:
            parents_deep = set([])
            for p in parents:
                parents_deep.add(p)

                # no loops that are typically caused by one to one relationships
                if entity_id in self.get_forward_entities(p):
                    continue

                to_add = self.get_forward_entities(p, deep=True)
                parents_deep = parents_deep.union(to_add)

            parents = parents_deep

        return set(parents)

    def get_backward_entities(self, entity_id, deep=False):
        """Get entities that are in a backward relationship with entity

        Args:
            entity_id (str) - Id entity of entity to search from.
            deep (bool) - If True, recursively find backward entities.

        Returns:
            Set of each :class:`.Entity` in a backward relationship.
        """
        children = [r.child_entity.id for r in
                    self.get_backward_relationships(entity_id)]
        if deep:
            children_deep = set([])
            for p in children:
                children_deep.add(p)
                to_add = self.get_backward_entities(p, deep=True)
                children_deep = children_deep.union(to_add)

            children = children_deep
        return set(children)

    def get_forward_relationships(self, entity_id):
        """Get relationships where entity "entity_id" is the child

        Args:
            entity_id (str): Id of entity to get relationships for.

        Returns:
            list[:class:`.Relationship`]: List of forward relationships.
        """
        return [r for r in self.relationships if r.child_entity.id == entity_id]

    def get_backward_relationships(self, entity_id):
        """
        get relationships where entity "entity_id" is the parent.

        Args:
            entity_id (str): Id of entity to get relationships for.

        Returns:
            list[:class:`.Relationship`]: list of backward relationships
        """
        return [r for r in self.relationships if r.parent_entity.id == entity_id]

    def get_relationship(self, eid_1, eid_2):
        """Get relationship, if any, between eid_1 and eid_2

        Args:
            eid_1 (str): Id of first entity to get relationships for.
            eid_2 (str): Id of second entity to get relationships for.

        Returns:
            :class:`.Relationship`: Relationship or None
        """
        for r in self.relationships:
            if r.child_entity.id == eid_1 and r.parent_entity.id == eid_2 or \
                    r.parent_entity.id == eid_1 and r.child_entity.id == eid_2:
                return r
        return None

    def _is_backward_relationship(self, rel, prev_ent):
        if prev_ent == rel.parent_entity.id:
            return True
        return False

    def path_relationships(self, path, start_entity_id):
        """
        Generate a list of the strings "forward" or "backward" corresponding to
        the direction of the relationship at each point in `path`.
        """
        prev_entity = start_entity_id
        rels = []
        for r in path:
            if self._is_backward_relationship(r, prev_entity):
                rels.append('backward')
                prev_entity = r.child_variable.entity.id
            else:
                rels.append('forward')
                prev_entity = r.parent_variable.entity.id
        return rels
