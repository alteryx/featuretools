from __future__ import print_function

import logging
from builtins import map

import pandas as pd
from past.builtins import basestring

from featuretools import variable_types as vtypes
from featuretools.core.base import FTBase
from featuretools.utils.wrangle import _dataframes_equal

logger = logging.getLogger('featuretools.entityset')


class BaseEntity(FTBase):
    """Represents an entity in a Entityset, and stores relevant metadata

    An Entity is analogous to a table in a relational database

    See Also:
        :class:`.Relationship`, :class:`.Variable`, :class:`.EntitySet`

    """
    id = None
    variables = None
    time_index = None
    index = None
    indexed_by = None

    def __init__(self, id, entityset, variable_types=None, name=None,
                 index=None, time_index=None, secondary_time_index=None,
                 relationships=None, already_sorted=False):
        """ Create Entity

        Args:
            id (str): A unique string identifying the entity.
            variable_types (dict[str -> type]): A mapping of variable IDs to
                subclasses of :class:`.Variable` that this entity will use to
                create variables.
            entityset (:class:`.base_entityset`): EntitySet this entity belongs to.
            name (str, optional): Optional human readable name.
            index (str, optional) : Id of index variable. Ignored if entityset provided.
            time_index (str, optional) : Id of time index variable. Ignored if entityset provided.
        """
        assert isinstance(id, basestring), "Entity id must be a string"

        self.id = id
        self.name = name
        self.entityset = entityset
        self.indexed_by = {}
        variable_types = variable_types or {}
        self.index = index
        self.secondary_time_index = secondary_time_index or {}
        # make sure time index is actually in the columns
        for ti, cols in self.secondary_time_index.items():
            if ti not in cols:
                cols.append(ti)

        link_vars = [v.id for rel in relationships for v in [rel.parent_variable, rel.child_variable]
                     if v.entity.id == self.id]

        inferred_variable_types = self.infer_variable_types(ignore=list(variable_types.keys()),
                                                            link_vars=link_vars)
        for var_id, desired_type in variable_types.items():
            if isinstance(desired_type, tuple):
                desired_type = desired_type[0]
            inferred_variable_types.update({var_id: desired_type})

        self.variables = []
        for v in inferred_variable_types:
            # TODO document how vtype can be tuple
            vtype = inferred_variable_types[v]
            if isinstance(vtype, tuple):
                # vtype is (ft.Variable, dict_of_kwargs)
                v = vtype[0](v, self, **vtype[1])
            else:
                v = inferred_variable_types[v](v, self)
            self.variables += [v]

        # do one last conversion of data once we've inferred
        self.convert_variable_types(inferred_variable_types)

        self.set_index(index)
        self.set_time_index(time_index, already_sorted)
        self.set_secondary_time_index(secondary_time_index)

        # todo check the logic of this. can index not be in variable types?
        if self.index is not None and self.index not in inferred_variable_types:
            self.add_variable(self.index, vtypes.Index)

        self.add_all_variable_statistics()

    def __repr__(self):
        repr_out = "Entity: {}\n".format(self.name)
        repr_out += "  Variables:"
        for v in self.variables:
            repr_out += "\n    {} (dtype: {})".format(v.id, v.dtype)

        shape = self.get_shape()
        repr_out += u"\n  Shape:\n    (Rows: {}, Columns: {})".format(
            shape[0], shape[1])
        return repr_out

    @property
    def shape(self):
        return self.get_shape()

    def __eq__(self, other, deep=False):
        if self.index != other.index:
            return False
        if self.time_index != other.time_index:
            return False
        if self.secondary_time_index != other.secondary_time_index:
            return False
        if len(self.variables) != len(other.variables):
            return False
        for v in self.variables:
            if v not in other.variables:
                return False
        if deep:
            if self.indexed_by is None and other.indexed_by is not None:
                return False
            elif self.indexed_by is not None and other.indexed_by is None:
                return False
            else:
                for v, index_map in self.indexed_by.items():
                    if v not in other.indexed_by:
                        return False
                    for i, related in index_map.items():
                        if i not in other.indexed_by[v]:
                            return False
                        # indexed_by maps instances of two entities together by lists
                        # We want to check that all the elements of the lists of instances
                        # for each relationship are the same in both entities being
                        # checked for equality, but don't care about the order.
                        if not set(related) == set(other.indexed_by[v][i]):
                            return False
            if self.last_time_index is None and other.last_time_index is not None:
                return False
            elif self.last_time_index is not None and other.last_time_index is None:
                return False
            elif self.last_time_index is not None and other.last_time_index is not None:
                if not self.last_time_index.equals(other.last_time_index):
                    return False

            if not _dataframes_equal(self.df, other.df):
                return False

        return True

    def __hash__(self):
        return id(self.id)

    def __getitem__(self, variable_id):
        return self._get_variable(variable_id)

    def _get_variable(self, variable_id):
        """Get variable instance

        Args:
            variable_id (str) : Id of variable to get.

        Returns:
            :class:`.Variable` : Instance of variable.

        Raises:
            RuntimeError : if no variable exist with provided id
        """
        for v in self.variables:
            if v.id == variable_id:
                return v

        raise KeyError("Variable: %s not found in entity" % (variable_id))

    def show_instance(self, instance_ids):
        """See row corresponding to instance id

        Args:
            instance_ids (object, list[object]) : Instance id or list of instance ids.

        Returns:
            :class:`pd.DataFrame` : Pandas DataFrame

        """
        return self.entityset.get_instance_data(self.id, instance_ids=instance_ids)

    def get_shape():
        raise NotImplementedError()

    @property
    def variable_types(self):
        return {v.id: type(v) for v in self.variables}

    def set_index(self, variable_id):
        self.index = variable_id

    def set_time_index(self, variable_id):
        self.time_index = variable_id

    def set_secondary_time_index(self, secondary_time_index):
        self.secondary_time_index = secondary_time_index or {}

    def add_variable(self, new_id, type):
        """Add variable to entity

        Args:
            new_id (str) : Id of variable to be added.
            type (Variable) : Class of variable.
        """
        if new_id in [v.id for v in self.variables]:
            logger.warning("Not adding duplicate variable: %s", new_id)
            return
        new_v = type(new_id, entity=self)
        self.variables.append(new_v)
        self.variable_types[new_id] = type
        self.add_variable_statistics(new_id)

    def get_variable_types(self):
        return self.variable_types

    def add_all_variable_statistics(self):
        for var_id in self.variable_types.keys():
            self.add_variable_statistics(var_id)

    def add_variable_statistics(self, var_id):
        vartype = self.variable_types[var_id]
        stats = vartype._setter_stats
        for stat in stats:
            try:
                value = self.get_column_stat(var_id, stat)
                setattr(self._get_variable(var_id), stat, value)
            except TypeError as e:
                print(e)

        stats = vartype._computed_stats
        for stat in stats:
            try:
                setattr(self._get_variable(var_id), stat, value)
            except TypeError as e:
                print(e)

    def get_column_stat(self, variable_id, stat):
        raise NotImplementedError()

    def _remove_variable_statistic(self, v, entityset, statistic):
        try:
            value = getattr(v, statistic)
        except AttributeError:
            pass
        else:
            if value is not None:
                setattr(v, statistic, None)

    def delete_variable(self, variable_id):
        v = self._get_variable(variable_id)
        self.variables.remove(v)

    def convert_variable_type(self, variable_id, new_type,
                              convert_data=True,
                              **kwargs):
        """Convert variable in dataframe to different type

        Args:
            variable_id (str) : Id of variable to convert.
            new_type (subclass of `Variable`) : Type of variable to convert to.
            entityset (:class:`.BaseEntitySet`) : EntitySet associated with this entity.
            convert_data (bool) : If True, convert underlying data in the EntitySet.

        Raises:
            RuntimeError : Raises if it cannot convert the underlying data

        Examples:
            >>> es["customer"].convert_variable_type("education_level", vtypes.Categorical, EntitySet)
                True
        """
        if convert_data:
            # first, convert the underlying data (or at least try to)
            self.entityset_convert_variable_type(
                variable_id, new_type, **kwargs)

        # replace the old variable with the new one, maintaining order
        variable = self._get_variable(variable_id)
        new_variable = new_type.create_from(variable)
        self.variables[self.variables.index(variable)] = new_variable

        self.add_variable_statistics(new_variable.id)

    @property
    def name(self):
        """Returns name of entity. If name is None, returns the id

        Returns:
            str : name of the entity
        """
        name = self._name
        if name is None:
            name = self.id
        return name

    @name.setter
    def name(self, name):
        self._name = name
        return True

    def has_time_index(self):
        """Returns True if there is a time_index, otherwise False"""
        return self.time_index is not None

    def is_child_of(self, entity_id):
        '''
        Returns True if self is a child of entity_id
        '''
        rels = self.entityset.get_backward_relationships(entity_id)
        return self.id in [r.child_entity.id for r in rels]

    def is_parent_of(self, entity_id):
        '''
        Returns True if self is a parent of entity_id
        '''
        rels = self.entityset.get_backward_relationships(self.id)
        return entity_id in [r.child_entity.id for r in rels]
