import copy
import logging
from collections import defaultdict

import dask.dataframe as dd
import numpy as np
import pandas as pd
from pandas.api.types import is_dtype_equal, is_numeric_dtype

import featuretools.variable_types.variable as vtypes
from featuretools.entityset import deserialize, serialize
from featuretools.entityset.entity import Entity
from featuretools.entityset.relationship import Relationship, RelationshipPath
from featuretools.utils.plot_utils import (
    check_graphviz,
    get_graphviz_format,
    save_graph
)

pd.options.mode.chained_assignment = None  # default='warn'
logger = logging.getLogger('featuretools.entityset')


class EntitySet(object):
    """
    Stores all actual data for a entityset

    Attributes:
        id
        entity_dict
        relationships
        time_type

    Properties:
        metadata

    """

    def __init__(self, id=None, entities=None, relationships=None):
        """Creates EntitySet

            Args:
                id (str) : Unique identifier to associate with this instance

                entities (dict[str -> tuple(pd.DataFrame, str, str, dict[str -> Variable])]): dictionary of
                    entities. Entries take the format
                    {entity id -> (dataframe, id column, (time_index), (variable_types), (make_index))}.
                    Note that time_index, variable_types and make_index are optional.

                relationships (list[(str, str, str, str)]): List of relationships
                    between entities. List items are a tuple with the format
                    (parent entity id, parent variable, child entity id, child variable).

            Example:

                .. code-block:: python

                    entities = {
                        "cards" : (card_df, "id"),
                        "transactions" : (transactions_df, "id", "transaction_time")
                    }

                    relationships = [("cards", "id", "transactions", "card_id")]

                    ft.EntitySet("my-entity-set", entities, relationships)
        """
        self.id = id
        self.entity_dict = {}
        self.relationships = []
        self.time_type = None

        entities = entities or {}
        relationships = relationships or []
        for entity in entities:
            df = entities[entity][0]
            index_column = entities[entity][1]
            time_index = None
            variable_types = None
            make_index = None
            if len(entities[entity]) > 2:
                time_index = entities[entity][2]
            if len(entities[entity]) > 3:
                variable_types = entities[entity][3]
            if len(entities[entity]) > 4:
                make_index = entities[entity][4]
            self.entity_from_dataframe(entity_id=entity,
                                       dataframe=df,
                                       index=index_column,
                                       time_index=time_index,
                                       variable_types=variable_types,
                                       make_index=make_index)

        for relationship in relationships:
            parent_variable = self[relationship[0]][relationship[1]]
            child_variable = self[relationship[2]][relationship[3]]
            self.add_relationship(Relationship(parent_variable,
                                               child_variable))
        self.reset_data_description()

    def __sizeof__(self):
        return sum([entity.__sizeof__() for entity in self.entities])

    def __dask_tokenize__(self):
        return (EntitySet, serialize.entityset_to_description(self.metadata))

    def __eq__(self, other, deep=False):
        if len(self.entity_dict) != len(other.entity_dict):
            return False
        for eid, e in self.entity_dict.items():
            if eid not in other.entity_dict:
                return False
            if not e.__eq__(other[eid], deep=deep):
                return False
        for r in other.relationships:
            if r not in other.relationships:
                return False
        return True

    def __ne__(self, other, deep=False):
        return not self.__eq__(other, deep=deep)

    def __getitem__(self, entity_id):
        """Get entity instance from entityset

        Args:
            entity_id (str): Id of entity.

        Returns:
            :class:`.Entity` : Instance of entity. None if entity doesn't
                exist.
        """
        if entity_id in self.entity_dict:
            return self.entity_dict[entity_id]
        name = self.id or "entity set"
        raise KeyError('Entity %s does not exist in %s' % (entity_id, name))

    @property
    def entities(self):
        return list(self.entity_dict.values())

    @property
    def metadata(self):
        '''Returns the metadata for this EntitySet. The metadata will be recomputed if it does not exist.'''
        if self._data_description is None:
            description = serialize.entityset_to_description(self)
            self._data_description = deserialize.description_to_entityset(description)

        return self._data_description

    def reset_data_description(self):
        self._data_description = None

    def to_pickle(self, path, compression=None, profile_name=None):
        '''Write entityset in the pickle format, location specified by `path`.
            Path could be a local path or a S3 path.
            If writing to S3 a tar archive of files will be written.

            Args:
                path (str): location on disk to write to (will be created as a directory)
                compression (str) : Name of the compression to use. Possible values are: {'gzip', 'bz2', 'zip', 'xz', None}.
                profile_name (str) : Name of AWS profile to use, False to use an anonymous profile, or None.
        '''
        serialize.write_data_description(self, path, format='pickle', compression=compression, profile_name=profile_name)
        return self

    def to_parquet(self, path, engine='auto', compression=None, profile_name=None):
        '''Write entityset to disk in the parquet format, location specified by `path`.
            Path could be a local path or a S3 path.
            If writing to S3 a tar archive of files will be written.

            Args:
                path (str): location on disk to write to (will be created as a directory)
                engine (str) : Name of the engine to use. Possible values are: {'auto', 'pyarrow', 'fastparquet'}.
                compression (str) : Name of the compression to use. Possible values are: {'snappy', 'gzip', 'brotli', None}.
                profile_name (str) : Name of AWS profile to use, False to use an anonymous profile, or None.
        '''
        serialize.write_data_description(self, path, format='parquet', engine=engine, compression=compression, profile_name=profile_name)
        return self

    def to_csv(self, path, sep=',', encoding='utf-8', engine='python', compression=None, profile_name=None):
        '''Write entityset to disk in the csv format, location specified by `path`.
            Path could be a local path or a S3 path.
            If writing to S3 a tar archive of files will be written.

            Args:
                path (str) : Location on disk to write to (will be created as a directory)
                sep (str) : String of length 1. Field delimiter for the output file.
                encoding (str) : A string representing the encoding to use in the output file, defaults to 'utf-8'.
                engine (str) : Name of the engine to use. Possible values are: {'c', 'python'}.
                compression (str) : Name of the compression to use. Possible values are: {'gzip', 'bz2', 'zip', 'xz', None}.
                profile_name (str) : Name of AWS profile to use, False to use an anonymous profile, or None.
        '''
        serialize.write_data_description(self, path, format='csv', index=False, sep=sep, encoding=encoding, engine=engine, compression=compression, profile_name=profile_name)
        return self

    def to_dictionary(self):
        return serialize.entityset_to_description(self)

    ###########################################################################
    #   Public getter/setter methods  #########################################
    ###########################################################################

    def __repr__(self):
        repr_out = u"Entityset: {}\n".format(self.id)
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
            repr_out += u"\n    No relationships"

        for r in self.relationships:
            repr_out += u"\n    %s.%s -> %s.%s" % \
                (r._child_entity_id, r._child_variable_id,
                 r._parent_entity_id, r._parent_variable_id)

        return repr_out

    def add_relationships(self, relationships):
        """Add multiple new relationships to a entityset

        Args:
            relationships (list[Relationship]) : List of new
                relationships.
        """
        return [self.add_relationship(r) for r in relationships][-1]

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
        if child_e.index == child_v:
            msg = "Unable to add relationship because child variable '{}' in '{}' is also its index"
            raise ValueError(msg.format(child_v, child_e.id))
        parent_e = relationship.parent_entity
        parent_v = relationship.parent_variable.id
        if not isinstance(child_e[child_v], vtypes.Id):
            child_e.convert_variable_type(variable_id=child_v,
                                          new_type=vtypes.Id,
                                          convert_data=False)

        if not isinstance(parent_e[parent_v], vtypes.Index):
            parent_e.convert_variable_type(variable_id=parent_v,
                                           new_type=vtypes.Index,
                                           convert_data=False)
        # Empty dataframes (as a result of accessing Entity.metadata)
        # default to object dtypes for discrete variables, but
        # indexes/ids default to ints. In this case, we convert
        # the empty column's type to int
        if isinstance(child_e.df, pd.DataFrame) and \
                (child_e.df.empty and child_e.df[child_v].dtype == object and
                 is_numeric_dtype(parent_e.df[parent_v])):
            child_e.df[child_v] = pd.Series(name=child_v, dtype=np.int64)

        parent_dtype = parent_e.df[parent_v].dtype
        child_dtype = child_e.df[child_v].dtype
        msg = u"Unable to add relationship because {} in {} is Pandas dtype {}"\
            u" and {} in {} is Pandas dtype {}."
        if not is_dtype_equal(parent_dtype, child_dtype):
            raise ValueError(msg.format(parent_v, parent_e.id, parent_dtype,
                                        child_v, child_e.id, child_dtype))

        self.relationships.append(relationship)
        self.reset_data_description()
        return self

    ###########################################################################
    #   Relationship access/helper methods  ###################################
    ###########################################################################

    def find_forward_paths(self, start_entity_id, goal_entity_id):
        """
        Generator which yields all forward paths between a start and goal
        entity. Does not include paths which contain cycles.

        Args:
            start_entity_id (str) : id of entity to start the search from
            goal_entity_id  (str) : if of entity to find forward path to

        See Also:
            :func:`BaseEntitySet.find_backward_paths`
        """
        for sub_entity_id, path in self._forward_entity_paths(start_entity_id):
            if sub_entity_id == goal_entity_id:
                yield path

    def find_backward_paths(self, start_entity_id, goal_entity_id):
        """
        Generator which yields all backward paths between a start and goal
        entity. Does not include paths which contain cycles.

        Args:
            start_entity_id (str) : Id of entity to start the search from.
            goal_entity_id  (str) : Id of entity to find backward path to.

        See Also:
            :func:`BaseEntitySet.find_forward_paths`
        """
        for path in self.find_forward_paths(goal_entity_id, start_entity_id):
            # Reverse path
            yield path[::-1]

    def _forward_entity_paths(self, start_entity_id, seen_entities=None):
        """
        Generator which yields the ids of all entities connected through forward
        relationships, and the path taken to each. An entity will be yielded
        multiple times if there are multiple paths to it.

        Implemented using depth first search.
        """
        if seen_entities is None:
            seen_entities = set()

        if start_entity_id in seen_entities:
            return

        seen_entities.add(start_entity_id)

        yield start_entity_id, []

        for relationship in self.get_forward_relationships(start_entity_id):
            next_entity = relationship.parent_entity.id
            # Copy seen entities for each next node to allow multiple paths (but
            # not cycles).
            descendants = self._forward_entity_paths(next_entity, seen_entities.copy())
            for sub_entity_id, sub_path in descendants:
                yield sub_entity_id, [relationship] + sub_path

    def get_forward_entities(self, entity_id, deep=False):
        """
        Get entities that are in a forward relationship with entity

        Args:
            entity_id (str): Id entity of entity to search from.
            deep (bool): if True, recursively find forward entities.

        Yields a tuple of (descendent_id, path from entity_id to descendant).
        """
        for relationship in self.get_forward_relationships(entity_id):
            parent_eid = relationship.parent_entity.id
            direct_path = RelationshipPath([(True, relationship)])
            yield parent_eid, direct_path

            if deep:
                sub_entities = self.get_forward_entities(parent_eid, deep=True)
                for sub_eid, path in sub_entities:
                    yield sub_eid, direct_path + path

    def get_backward_entities(self, entity_id, deep=False):
        """
        Get entities that are in a backward relationship with entity

        Args:
            entity_id (str): Id entity of entity to search from.
            deep (bool): if True, recursively find backward entities.

        Yields a tuple of (descendent_id, path from entity_id to descendant).
        """
        for relationship in self.get_backward_relationships(entity_id):
            child_eid = relationship.child_entity.id
            direct_path = RelationshipPath([(False, relationship)])
            yield child_eid, direct_path

            if deep:
                sub_entities = self.get_backward_entities(child_eid, deep=True)
                for sub_eid, path in sub_entities:
                    yield sub_eid, direct_path + path

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

    def has_unique_forward_path(self, start_entity_id, end_entity_id):
        """
        Is the forward path from start to end unique?

        This will raise if there is no such path.
        """
        paths = self.find_forward_paths(start_entity_id, end_entity_id)

        next(paths)
        second_path = next(paths, None)

        return not second_path

    ###########################################################################
    #  Entity creation methods  ##############################################
    ###########################################################################

    def entity_from_dataframe(self,
                              entity_id,
                              dataframe,
                              index=None,
                              variable_types=None,
                              make_index=False,
                              time_index=None,
                              secondary_time_index=None,
                              already_sorted=False):
        """
        Load the data for a specified entity from a Pandas DataFrame.

        Args:
            entity_id (str) : Unique id to associate with this entity.

            dataframe (pandas.DataFrame) : Dataframe containing the data.

            index (str, optional): Name of the variable used to index the entity.
                If None, take the first column.

            variable_types (dict[str -> Variable/str], optional):
                Keys are of variable ids and values are variable types or type_strings. Used to to
                initialize an entity's store.

            make_index (bool, optional) : If True, assume index does not
                exist as a column in dataframe, and create a new column of that name
                using integers. Otherwise, assume index exists.

            time_index (str, optional): Name of the variable containing
                time data. Type must be in :class:`variables.DateTime` or be
                able to be cast to datetime (e.g. str, float, or numeric.)

            secondary_time_index (dict[str -> Variable]): Name of variable
                containing time data to use a second time index for the entity.

            already_sorted (bool, optional) : If True, assumes that input dataframe
                is already sorted by time. Defaults to False.

        Notes:

            Will infer variable types from Pandas dtype

        Example:
            .. ipython:: python

                import featuretools as ft
                import pandas as pd
                transactions_df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                                                "session_id": [1, 2, 1, 3, 4, 5],
                                                "amount": [100.40, 20.63, 33.32, 13.12, 67.22, 1.00],
                                                "transaction_time": pd.date_range(start="10:00", periods=6, freq="10s"),
                                                "fraud": [True, False, True, False, True, True]})
                es = ft.EntitySet("example")
                es.entity_from_dataframe(entity_id="transactions",
                                         index="id",
                                         time_index="transaction_time",
                                         dataframe=transactions_df)

                es["transactions"]
                es["transactions"].df

        """
        variable_types = variable_types or {}

        if time_index is not None and time_index == index:
            raise ValueError("time_index and index cannot be the same value, %s" % (time_index))

        if time_index is None:
            for variable, variable_type in variable_types.items():
                if variable_type == vtypes.DatetimeTimeIndex:
                    raise ValueError("DatetimeTimeIndex variable %s must be set using time_index parameter" % (variable))

        if len(self.entities) > 0:
            if not isinstance(dataframe, type(self.entities[0].df)):
                raise ValueError("All entity dataframes must be of the same type. "
                                 "Cannot add entity of type {} to an entityset with existing entities "
                                 "of type {}".format(type(dataframe), type(self.entities[0].df)))

        entity = Entity(
            entity_id,
            dataframe,
            self,
            variable_types=variable_types,
            index=index,
            time_index=time_index,
            secondary_time_index=secondary_time_index,
            already_sorted=already_sorted,
            make_index=make_index)
        self.entity_dict[entity.id] = entity
        self.reset_data_description()
        return self

    def normalize_entity(self, base_entity_id, new_entity_id, index,
                         additional_variables=None, copy_variables=None,
                         make_time_index=None,
                         make_secondary_time_index=None,
                         new_entity_time_index=None,
                         new_entity_secondary_time_index=None):
        """Create a new entity and relationship from unique values of an existing variable.

        Args:
            base_entity_id (str) : Entity id from which to split.

            new_entity_id (str): Id of the new entity.

            index (str): Variable in old entity
                that will become index of new entity. Relationship
                will be created across this variable.

            additional_variables (list[str]):
                List of variable ids to remove from
                base_entity and move to new entity.

            copy_variables (list[str]): List of
                variable ids to copy from old entity
                and move to new entity.

            make_time_index (bool or str, optional): Create time index for new entity based
                on time index in base_entity, optionally specifying which variable in base_entity
                to use for time_index. If specified as True without a specific variable,
                uses the primary time index. Defaults to True if base entity has a time index.

            make_secondary_time_index (dict[str -> list[str]], optional): Create a secondary time index
                from key. Values of dictionary
                are the variables to associate with the secondary time index. Only one
                secondary time index is allowed. If None, only associate the time index.

            new_entity_time_index (str, optional): Rename new entity time index.

            new_entity_secondary_time_index (str, optional): Rename new entity secondary time index.

        """
        base_entity = self.entity_dict[base_entity_id]
        additional_variables = additional_variables or []
        copy_variables = copy_variables or []

        # Check base entity to make sure time index is valid
        if base_entity.time_index is not None:
            t_index = base_entity[base_entity.time_index]
            if not isinstance(t_index, (vtypes.NumericTimeIndex, vtypes.DatetimeTimeIndex)):
                base_error = "Time index '{0}' is not a NumericTimeIndex or DatetimeTimeIndex, but type {1}. Use set_time_index on entity '{2}' to set the time_index."
                raise TypeError(base_error.format(base_entity.time_index, type(t_index), str(base_entity.id)))

        if not isinstance(additional_variables, list):
            raise TypeError("'additional_variables' must be a list, but received type {}"
                            .format(type(additional_variables)))

        if len(additional_variables) != len(set(additional_variables)):
            raise ValueError("'additional_variables' contains duplicate variables. All variables must be unique.")

        if not isinstance(copy_variables, list):
            raise TypeError("'copy_variables' must be a list, but received type {}"
                            .format(type(copy_variables)))

        if len(copy_variables) != len(set(copy_variables)):
            raise ValueError("'copy_variables' contains duplicate variables. All variables must be unique.")

        for v in additional_variables + copy_variables:
            if v == index:
                raise ValueError("Not copying {} as both index and variable".format(v))

        for v in additional_variables:
            if v == base_entity.time_index:
                raise ValueError("Not moving {} as it is the base time index variable. Perhaps, move the variable to the copy_variables.".format(v))

        if isinstance(make_time_index, str):
            if make_time_index not in base_entity.df.columns:
                raise ValueError("'make_time_index' must be a variable in the base entity")
            elif make_time_index not in additional_variables + copy_variables:
                raise ValueError("'make_time_index' must be specified in 'additional_variables' or 'copy_variables'")
        if index == base_entity.index:
            raise ValueError("'index' must be different from the index column of the base entity")

        transfer_types = {}
        transfer_types[index] = type(base_entity[index])
        for v in additional_variables + copy_variables:
            if type(base_entity[v]) == vtypes.DatetimeTimeIndex:
                transfer_types[v] = vtypes.Datetime
            elif type(base_entity[v]) == vtypes.NumericTimeIndex:
                transfer_types[v] = vtypes.Numeric
            else:
                transfer_types[v] = type(base_entity[v])

        # create and add new entity
        new_entity_df = self[base_entity_id].df.copy()

        if make_time_index is None and base_entity.time_index is not None:
            make_time_index = True

        if isinstance(make_time_index, str):
            # Set the new time index to make_time_index.
            base_time_index = make_time_index
            new_entity_time_index = make_time_index
            already_sorted = (new_entity_time_index == base_entity.time_index)
        elif make_time_index:
            # Create a new time index based on the base entity time index.
            base_time_index = base_entity.time_index
            if new_entity_time_index is None:
                new_entity_time_index = "first_%s_time" % (base_entity.id)

            already_sorted = True

            assert base_entity.time_index is not None, \
                "Base entity doesn't have time_index defined"

            if base_time_index not in [v for v in additional_variables]:
                copy_variables.append(base_time_index)

            transfer_types[new_entity_time_index] = type(base_entity[base_entity.time_index])
        else:
            new_entity_time_index = None
            already_sorted = False

        if new_entity_time_index is not None and new_entity_time_index == index:
            raise ValueError("time_index and index cannot be the same value, %s" % (new_entity_time_index))

        selected_variables = [index] +\
            [v for v in additional_variables] +\
            [v for v in copy_variables]

        new_entity_df2 = new_entity_df. \
            drop_duplicates(index, keep='first')[selected_variables]

        if make_time_index:
            new_entity_df2 = new_entity_df2.rename(columns={base_time_index: new_entity_time_index})
        if make_secondary_time_index:
            assert len(make_secondary_time_index) == 1, "Can only provide 1 secondary time index"
            secondary_time_index = list(make_secondary_time_index.keys())[0]

            secondary_variables = [index, secondary_time_index] + list(make_secondary_time_index.values())[0]
            secondary_df = new_entity_df. \
                drop_duplicates(index, keep='last')[secondary_variables]
            if new_entity_secondary_time_index:
                secondary_df = secondary_df.rename(columns={secondary_time_index: new_entity_secondary_time_index})
                secondary_time_index = new_entity_secondary_time_index
            else:
                new_entity_secondary_time_index = secondary_time_index
            secondary_df = secondary_df.set_index(index)
            new_entity_df = new_entity_df2.join(secondary_df, on=index)
        else:
            new_entity_df = new_entity_df2

        base_entity_index = index

        transfer_types[index] = vtypes.Categorical
        if make_secondary_time_index:
            old_ti_name = list(make_secondary_time_index.keys())[0]
            ti_cols = list(make_secondary_time_index.values())[0]
            ti_cols = [c if c != old_ti_name else secondary_time_index for c in ti_cols]
            make_secondary_time_index = {secondary_time_index: ti_cols}

        self.entity_from_dataframe(
            new_entity_id,
            new_entity_df,
            index,
            already_sorted=already_sorted,
            time_index=new_entity_time_index,
            secondary_time_index=make_secondary_time_index,
            variable_types=transfer_types)

        self.entity_dict[base_entity_id].delete_variables(additional_variables)

        new_entity = self.entity_dict[new_entity_id]
        base_entity.convert_variable_type(base_entity_index, vtypes.Id, convert_data=False)
        self.add_relationship(Relationship(new_entity[index], base_entity[base_entity_index]))
        self.reset_data_description()
        return self

    ###########################################################################
    #  Data wrangling methods  ###############################################
    ###########################################################################

    def concat(self, other, inplace=False):
        '''Combine entityset with another to create a new entityset with the
        combined data of both entitysets.
        '''
        assert_string = "Entitysets must have the same entities, relationships"\
            ", and variable_ids"
        assert (self.__eq__(other) and
                self.relationships == other.relationships), assert_string

        for entity in self.entities:
            assert entity.id in other.entity_dict, assert_string
            assert (len(self[entity.id].variables) ==
                    len(other[entity.id].variables)), assert_string
            other_variable_ids = [o_variable.id for o_variable in
                                  other[entity.id].variables]
            assert (all([variable.id in other_variable_ids
                         for variable in self[entity.id].variables])), assert_string

        if inplace:
            combined_es = self
        else:
            combined_es = copy.deepcopy(self)

        has_last_time_index = []
        for entity in self.entities:
            self_df = entity.df
            other_df = other[entity.id].df
            combined_df = pd.concat([self_df, other_df])
            if entity.created_index == entity.index:
                columns = [col for col in combined_df.columns if
                           col != entity.index or col != entity.time_index]
            else:
                columns = [entity.index]
            combined_df.drop_duplicates(columns, inplace=True)

            if entity.time_index:
                combined_df.sort_values([entity.time_index, entity.index], inplace=True)
            else:
                combined_df.sort_index(inplace=True)
            if (entity.last_time_index is not None or
                    other[entity.id].last_time_index is not None):
                has_last_time_index.append(entity.id)
            combined_es[entity.id].update_data(df=combined_df,
                                               recalculate_last_time_indexes=False)

        combined_es.add_last_time_indexes(updated_entities=has_last_time_index)
        self.reset_data_description()
        return combined_es

    ###########################################################################
    #  Indexing methods  ###############################################
    ###########################################################################
    def add_last_time_indexes(self, updated_entities=None):
        """
        Calculates the last time index values for each entity (the last time
        an instance or children of that instance were observed).  Used when
        calculating features using training windows
        Args:
            updated_entities (list[str]): List of entity ids to update last_time_index for
                (will update all parents of those entities as well)
        """
        # Generate graph of entities to find leaf entities
        children = defaultdict(list)  # parent --> child mapping
        child_vars = defaultdict(dict)
        for r in self.relationships:
            children[r.parent_entity.id].append(r.child_entity)
            child_vars[r.parent_entity.id][r.child_entity.id] = r.child_variable

        updated_entities = updated_entities or []
        if updated_entities:
            # find parents of updated_entities
            parent_queue = updated_entities[:]
            parents = set()
            while len(parent_queue):
                e = parent_queue.pop(0)
                if e in parents:
                    continue
                parents.add(e)

                for parent_id, _ in self.get_forward_entities(e):
                    parent_queue.append(parent_id)

            queue = [self[p] for p in parents]
            to_explore = parents
        else:
            to_explore = set([e.id for e in self.entities[:]])
            queue = self.entities[:]

        explored = set()

        for e in queue:
            e.last_time_index = None

        # We will explore children of entities on the queue,
        # which may not be in the to_explore set. Therefore,
        # we check whether all elements of to_explore are in
        # explored, rather than just comparing length
        while not to_explore.issubset(explored):
            entity = queue.pop(0)

            if entity.last_time_index is None:
                if entity.time_index is not None:
                    lti = entity.df[entity.time_index].copy()
                    if isinstance(entity.df, dd.DataFrame):
                        # The current Dask implementation doesn't set the index of the dataframe
                        # to the entity's index, so we have to do it manually here
                        lti.index = entity.df[entity.index].copy()
                else:
                    lti = entity.df[entity.index].copy()
                    if isinstance(entity.df, dd.DataFrame):
                        lti.index = entity.df[entity.index].copy()
                        lti = lti.apply(lambda x: None)
                    else:
                        lti[:] = None
                entity.last_time_index = lti

            if entity.id in children:
                child_entities = children[entity.id]

                # if all children not explored, skip for now
                if not set([e.id for e in child_entities]).issubset(explored):
                    # Now there is a possibility that a child entity
                    # was not explicitly provided in updated_entities,
                    # and never made it onto the queue. If updated_entities
                    # is None then we just load all entities onto the queue
                    # so we didn't need this logic
                    for e in child_entities:
                        if e.id not in explored and e.id not in [q.id for q in queue]:
                            queue.append(e)
                    queue.append(entity)
                    continue

                # updated last time from all children
                for child_e in child_entities:
                    if child_e.last_time_index is None:
                        continue
                    link_var = child_vars[entity.id][child_e.id].id

                    if isinstance(child_e.last_time_index, dd.Series):
                        to_join = child_e.df[link_var]
                        to_join.index = child_e.df[child_e.index]

                        lti_df = child_e.last_time_index.to_frame(name='last_time').join(
                            to_join.to_frame(name=entity.index)
                        )
                        new_index = lti_df.index.copy()
                        new_index.name = None
                        lti_df.index = new_index
                        lti_df = lti_df.groupby(lti_df[entity.index]).agg('max')

                        lti_df = entity.last_time_index.to_frame(name='last_time_old').join(lti_df)

                    else:
                        lti_df = pd.DataFrame({'last_time': child_e.last_time_index,
                                               entity.index: child_e.df[link_var]})

                        # sort by time and keep only the most recent
                        lti_df.sort_values(['last_time', entity.index],
                                           kind="mergesort", inplace=True)

                        lti_df.drop_duplicates(entity.index,
                                               keep='last',
                                               inplace=True)

                        lti_df.set_index(entity.index, inplace=True)
                        lti_df = lti_df.reindex(entity.last_time_index.index)
                        lti_df['last_time_old'] = entity.last_time_index
                    if not isinstance(lti_df, dd.DataFrame) and lti_df.empty:
                        # Pandas errors out if it tries to do fillna and then max on an empty dataframe
                        lti_df = pd.Series()
                    else:
                        lti_df['last_time'] = lti_df['last_time'].astype('datetime64[ns]')
                        lti_df['last_time_old'] = lti_df['last_time_old'].astype('datetime64[ns]')
                        lti_df = lti_df.fillna(pd.to_datetime('1800-01-01 00:00')).max(axis=1)
                        lti_df = lti_df.replace(pd.to_datetime('1800-01-01 00:00'), pd.NaT)
                    # lti_df = lti_df.apply(lambda x: x.dropna().max(), axis=1)

                    entity.last_time_index = lti_df
                    entity.last_time_index.name = 'last_time'

            explored.add(entity.id)
        self.reset_data_description()

    ###########################################################################
    #  Other ###############################################
    ###########################################################################

    def add_interesting_values(self, max_values=5, verbose=False):
        """Find interesting values for categorical variables, to be used to generate "where" clauses

        Args:
            max_values (int) : Maximum number of values per variable to add.
            verbose (bool) : If True, print summary of interesting values found.

        Returns:
            None

        """
        for entity in self.entities:
            entity.add_interesting_values(max_values=max_values, verbose=verbose)
        self.reset_data_description()

    def plot(self, to_file=None):
        """
        Create a UML diagram-ish graph of the EntitySet.

        Args:
            to_file (str, optional) : Path to where the plot should be saved.
                If set to None (as by default), the plot will not be saved.

        Returns:
            graphviz.Digraph : Graph object that can directly be displayed in
                Jupyter notebooks.

        """
        graphviz = check_graphviz()
        format_ = get_graphviz_format(graphviz=graphviz,
                                      to_file=to_file)

        # Initialize a new directed graph
        graph = graphviz.Digraph(self.id, format=format_,
                                 graph_attr={'splines': 'ortho'})

        # Draw entities
        for entity in self.entities:
            variables_string = '\l'.join([var.id + ' : ' + var.type_string  # noqa: W605
                                          for var in entity.variables])
            nrows = entity.shape[0]
            label = '{%s (%d row%s)|%s\l}' % (entity.id, nrows, 's' * (nrows > 1), variables_string)  # noqa: W605
            graph.node(entity.id, shape='record', label=label)

        # Draw relationships
        for rel in self.relationships:
            # Display the key only once if is the same for both related entities
            if rel._parent_variable_id == rel._child_variable_id:
                label = rel._parent_variable_id
            else:
                label = '%s -> %s' % (rel._parent_variable_id,
                                      rel._child_variable_id)

            graph.edge(rel._child_entity_id, rel._parent_entity_id, xlabel=label)

        if to_file:
            save_graph(graph, to_file, format_)
        return graph
