from __future__ import division, print_function

import copy
import logging
import time
from builtins import range
from datetime import datetime

import numpy as np
import pandas as pd
from past.builtins import basestring

from .timedelta import Timedelta

from featuretools import variable_types as vtypes
from featuretools.utils.wrangle import (
    _check_time_type,
    _check_timedelta,
    _dataframes_equal
)

logger = logging.getLogger('featuretools.entityset')

_numeric_types = vtypes.PandasTypes._pandas_numerics
_categorical_types = [vtypes.PandasTypes._categorical]
_datetime_types = vtypes.PandasTypes._pandas_datetimes


class Entity(object):
    """Represents an entity in a Entityset, and stores relevant metadata and data

    An Entity is analogous to a table in a relational database

    See Also:
        :class:`.Relationship`, :class:`.Variable`, :class:`.EntitySet`

    """
    id = None
    variables = None
    time_index = None
    index = None
    indexed_by = None

    def __init__(self, id, df, entityset, variable_types=None,
                 index=None, time_index=None, secondary_time_index=None,
                 last_time_index=None, encoding=None, relationships=None,
                 already_sorted=False, created_index=None, verbose=False):
        """ Create Entity

        Args:
            id (str): Id of Entity.
            df (pd.DataFrame): Dataframe providing the data for the
                entity.
            entityset (EntitySet): Entityset for this Entity.
            variable_types (dict[str -> dict[str -> type]]) : Optional mapping of
                entity_id to variable_types dict with which to initialize an
                entity's store.
                An entity's variable_types dict maps string variable ids to types (:class:`.Variable`).
            index (str): Name of id column in the dataframe.
            time_index (str): Name of time column in the dataframe.
            secondary_time_index (dict[str -> str]): Dictionary mapping columns
                in the dataframe to the time index column they are associated with.
            last_time_index (pd.Series): Time index of the last event for each
                instance across all child entities.
            encoding (str, optional)) : If None, will use 'ascii'. Another option is 'utf-8',
                or any encoding supported by pandas.
            relationships (list): List of known relationships to other entities,
                used for inferring variable types.

        """
        assert isinstance(id, basestring), "Entity id must be a string"
        assert len(df.columns) == len(set(df.columns)), "Duplicate column names"
        self.data = {"df": df,
                     "last_time_index": last_time_index,
                     "indexed_by": {}
                     }
        self.encoding = encoding
        self._verbose = verbose
        self.created_index = created_index
        self.convert_all_variable_data(variable_types)
        self.id = id
        self.entityset = entityset
        self.indexed_by = {}
        variable_types = variable_types or {}
        self.index = index
        self.time_index = time_index
        self.secondary_time_index = secondary_time_index or {}
        # make sure time index is actually in the columns
        for ti, cols in self.secondary_time_index.items():
            if ti not in cols:
                cols.append(ti)

        relationships = relationships or []
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
                _v = vtype[0](v, self, **vtype[1])
            else:
                _v = inferred_variable_types[v](v, self)
            self.variables += [_v]

        # do one last conversion of data once we've inferred
        self.convert_all_variable_data(inferred_variable_types)

        # todo check the logic of this. can index not be in variable types?
        if self.index is not None and self.index not in inferred_variable_types:
            self.add_variable(self.index, vtypes.Index)

        # make sure index is at the beginning
        index_variable = [v for v in self.variables
                          if v.id == self.index][0]
        self.variables = [index_variable] + [v for v in self.variables
                                             if v.id != self.index]
        self.update_data(df=self.df,
                         already_sorted=already_sorted,
                         recalculate_last_time_indexes=False,
                         reindex=False)

    def __repr__(self):
        repr_out = u"Entity: {}\n".format(self.id)
        repr_out += u"  Variables:"
        for v in self.variables:
            repr_out += u"\n    {} (dtype: {})".format(v.id, v.dtype)

        shape = self.shape
        repr_out += u"\n  Shape:\n    (Rows: {}, Columns: {})".format(
            shape[0], shape[1])

        # encode for python 2
        if type(repr_out) != str:
            repr_out = repr_out.encode("utf-8")

        return repr_out

    @property
    def shape(self):
        return self.df.shape

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

    def __sizeof__(self):
        return sum([value.__sizeof__() for value in self.data.values()])

    @property
    def is_metadata(self):
        return self.entityset.is_metadata

    @property
    def df(self):
        return self.data["df"]

    @df.setter
    def df(self, _df):
        self.data["df"] = _df

    @property
    def last_time_index(self):
        return self.data["last_time_index"]

    @last_time_index.setter
    def last_time_index(self, lti):
        self.data["last_time_index"] = lti

    @property
    def indexed_by(self):
        return self.data["indexed_by"]

    @indexed_by.setter
    def indexed_by(self, idx):
        self.data["indexed_by"] = idx

    @property
    def parents(self):
        return [p.parent_entity.id for p in self.entityset.get_forward_relationships(self.id)]

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

    @property
    def variable_types(self):
        return {v.id: type(v) for v in self.variables}

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
            self.convert_variable_data(
                variable_id, new_type, **kwargs)

        # replace the old variable with the new one, maintaining order
        variable = self._get_variable(variable_id)
        new_variable = new_type.create_from(variable)
        self.variables[self.variables.index(variable)] = new_variable

    def convert_all_variable_data(self, variable_types):
        for var_id, desired_type in variable_types.items():
            type_args = {}
            if isinstance(desired_type, tuple):
                # grab args before assigning type
                type_args = desired_type[1]
                desired_type = desired_type[0]

            if var_id not in self.df.columns:
                raise LookupError("Variable ID %s not in DataFrame" % (var_id))
            current_type = self.df[var_id].dtype.name

            if issubclass(desired_type, vtypes.Numeric) and \
                    current_type not in _numeric_types:
                self.convert_variable_data(var_id, desired_type, **type_args)

            if issubclass(desired_type, vtypes.Discrete) and \
                    current_type not in _categorical_types:
                self.convert_variable_data(var_id, desired_type, **type_args)

            if issubclass(desired_type, vtypes.Datetime) and \
                    current_type not in _datetime_types:
                self.convert_variable_data(var_id, desired_type, **type_args)

    def convert_variable_data(self, column_id, new_type, **kwargs):
        """
        Convert variable in data set to different type
        """
        df = self.df
        if df[column_id].empty:
            return
        if new_type == vtypes.Numeric:
            orig_nonnull = df[column_id].dropna().shape[0]
            df[column_id] = pd.to_numeric(df[column_id], errors='coerce')
            # This will convert strings to nans
            # If column contained all strings, then we should
            # just raise an error, because that shouldn't have
            # been converted to numeric
            nonnull = df[column_id].dropna().shape[0]
            if nonnull == 0 and orig_nonnull != 0:
                raise TypeError("Attempted to convert all string column {} to numeric".format(column_id))
        elif new_type == vtypes.Datetime:
            format = kwargs.get("format", None)
            # TODO: if float convert to int?
            df[column_id] = pd.to_datetime(df[column_id], format=format,
                                           infer_datetime_format=True)
        elif new_type == vtypes.Boolean:
            map_dict = {kwargs.get("true_val", True): True,
                        kwargs.get("false_val", False): False,
                        True: True,
                        False: False}
            # TODO: what happens to nans?
            df[column_id] = df[column_id].map(map_dict).astype(np.bool)
        elif not issubclass(new_type, vtypes.Discrete):
            raise Exception("Cannot convert column %s to %s" %
                            (column_id, new_type))

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

    def query_by_values(self, instance_vals, variable_id=None, columns=None,
                        time_last=None, training_window=None,
                        return_sorted=False, start=None, end=None,
                        random_seed=None, shuffle=False):
        """Query instances that have variable with given value

        Args:
            instance_vals (pd.Dataframe, pd.Series, list[str] or str) :
                Instance(s) to match.
            variable_id (str) : Variable to query on. If None, query on index.
            columns (list[str]) : Columns to return. Return all columns if None.
            time_last (pd.TimeStamp) : Query data up to and including this
                time. Only applies if entity has a time index.
            training_window (Timedelta, optional):
                Data older than time_last by more than this will be ignored
            return_sorted (bool) : Return instances in the same order as
                the instance_vals are passed.
            start (int) : If provided, only return instances equal to or after this index.
            end (int) : If provided, only return instances before this index.
            random_seed (int) : Provided to the shuffling procedure.
            shuffle (bool) : If True, values will be shuffled before returning.

        Returns:
            pd.DataFrame : instances that match constraints
        """
        instance_vals = self._vals_to_series(instance_vals, variable_id)

        training_window = _check_timedelta(training_window)
        if training_window is not None:
            assert (isinstance(training_window, Timedelta) and
                    training_window.is_absolute()),\
                "training window must be an absolute Timedelta"

        if instance_vals is None:
            df = self.df

        elif variable_id is None or variable_id == self.index:
            df = self.df.reindex(instance_vals)
            df.dropna(subset=[self.index], inplace=True)

        elif variable_id in self.indexed_by:
            # some variables are indexed ahead of time
            index = self.indexed_by[variable_id]

            # generate pd.Series of all values from the index. Indexing
            # is much faster on this type.
            to_append = [pd.Series(index[v]) for v in instance_vals
                         if v in index]
            my_id_vals = pd.Series([]).append(to_append)
            df = self.df.loc[my_id_vals]

        else:
            # filter by "row.variable_id IN instance_vals"
            mask = self.df[variable_id].isin(instance_vals)
            df = self.df[mask]

        sortby = variable_id if (return_sorted and not shuffle) else None
        return self._filter_and_sort(df=df,
                                     time_last=time_last,
                                     training_window=training_window,
                                     columns=columns,
                                     sortby=sortby,
                                     start=start,
                                     end=end,
                                     shuffle=shuffle,
                                     random_seed=random_seed)

    def index_data(self):
        for p in self.parents:
            self.index_by_parent(self.entityset[p])

    def index_by_parent(self, parent_entity):
        """
        Cache the instances of this entity grouped by the parent entity.
        """
        r = parent_entity.entityset.get_relationship(self.id,
                                                     parent_entity.id)
        relation_var_id = r.child_variable.id
        if relation_var_id not in self.indexed_by:
            self.indexed_by[relation_var_id] = {}
        else:
            if self._verbose:
                print('Re-indexing %s by %s' % (self.id, parent_entity.id))

        self._index_by_variable(relation_var_id)

    def _index_by_variable(self, variable_id):
        """
        Cache the instances of this entity grouped by a variable.
        This allows filtering to happen much more quickly later.
        """
        ts = time.time()
        gb = self.df.groupby(self.df[variable_id])
        index = self.indexed_by[variable_id]

        if self._verbose:
            print("Indexing '%s' in %d groups by variable '%s'" %
                  (self.id, len(gb.groups), variable_id))

        # index by each parent instance separately
        for i in gb.groups:
            index[i] = np.array(gb.groups[i])

        if self._verbose:
            print("...%d child instances took %.2f seconds" %
                  (len(self.df.index), time.time() - ts))

    def infer_variable_types(self, ignore=None, link_vars=None):
        """Extracts the variables from a dataframe

        Args:
            ignore (list[str]): Names of variables (columns) for which to skip
                inference.
            link_vars (list[str]): Name of linked variables to other entities.
        Returns:
            list[Variable]: A list of variables describing the
                contents of the dataframe.
        """
        # TODO: set pk and pk types here
        ignore = ignore or []
        link_vars = link_vars or []
        inferred_types = {}
        df = self.df
        vids_to_assume_datetime = [self.time_index]
        if len(list(self.secondary_time_index.keys())):
            vids_to_assume_datetime.append(list(self.secondary_time_index.keys())[0])
        inferred_type = vtypes.Unknown
        for variable in df.columns:
            if variable in ignore:
                continue
            elif variable == self.index or variable.lower() == 'id':
                inferred_type = vtypes.Index

            elif variable in vids_to_assume_datetime:
                if col_is_datetime(df[variable]):
                    inferred_type = vtypes.Datetime
                else:
                    inferred_type = vtypes.Numeric

            elif df[variable].dtype == "object":
                if variable in link_vars:
                    inferred_type = vtypes.Categorical
                elif len(df[variable]):
                    if col_is_datetime(df[variable]):
                        inferred_type = vtypes.Datetime
                    else:
                        # heuristics to predict this some other than categorical
                        sample = df[variable].sample(min(10000, df[variable].nunique()))
                        avg_length = sample.str.len().mean()
                        if avg_length > 50:
                            inferred_type = vtypes.Text
                        else:
                            inferred_type = vtypes.Categorical

            elif df[variable].dtype == "bool":
                inferred_type = vtypes.Boolean

            elif df[variable].dtype.name == "category":
                inferred_type = vtypes.Categorical

            elif col_is_datetime(df[variable]):
                inferred_type = vtypes.Datetime

            elif variable in link_vars:
                inferred_type = vtypes.Ordinal

            elif len(df[variable]):
                sample = df[variable] \
                    .sample(min(10000, df[variable].nunique(dropna=False)))

                unique = sample.unique()
                percent_unique = sample.size / len(unique)

                if percent_unique < .05:
                    inferred_type = vtypes.Categorical
                else:
                    inferred_type = vtypes.Numeric

            inferred_types[variable] = inferred_type

        return inferred_types

    def update_data(self, df, already_sorted=False,
                    reindex=True, recalculate_last_time_indexes=True):
        '''Update entity's internal dataframe, optionaly making sure data is sorted,
        reference indexes to other entities are consistent, and last_time_indexes
        are consistent.
        '''
        if len(df.columns) != len(self.variables):
            raise ValueError("Updated dataframe contains {} columns, expecting {}".format(len(df.columns),
                                                                                          len(self.variables)))
        for v in self.variables:
            if v.id not in df.columns:
                raise ValueError("Updated dataframe is missing new {} column".format(v.id))

        # Make sure column ordering matches variable ordering
        self.df = df[[v.id for v in self.variables]]
        self.set_index(self.index)
        self.set_time_index(self.time_index, already_sorted=already_sorted)
        self.set_secondary_time_index(self.secondary_time_index)
        if reindex:
            self.index_data()
        if recalculate_last_time_indexes and self.last_time_index is not None:
            self.entityset.add_last_time_indexes(updated_entities=[self.id])

    def add_interesting_values(self, max_values=5, verbose=False):
        """
        Find interesting values for categorical variables, to be used to
            generate "where" clauses

        Args:
            max_values (int) : Maximum number of values per variable to add.
            verbose (bool) : If True, print summary of interesting values found.

        Returns:
            None
        """
        for variable in self.variables:
            # some heuristics to find basic 'where'-able variables
            if isinstance(variable, vtypes.Discrete):
                variable.interesting_values = []

                # TODO - consider removing this constraints
                # don't add interesting values for entities in relationships
                skip = False
                for r in self.entityset.relationships:
                    if variable in [r.child_variable, r.parent_variable]:
                        skip = True
                        break
                if skip:
                    continue

                counts = self.df[variable.id].value_counts()

                # find how many of each unique value there are; sort by count,
                # and add interesting values to each variable
                total_count = np.sum(counts)
                counts[:] = counts.sort_values()[::-1]
                for i in range(min(max_values, len(counts.index))):
                    idx = counts.index[i]

                    # add the value to interesting_values if it represents more than
                    # 25% of the values we have not seen so far
                    if len(counts.index) < 25:
                        if verbose:
                            msg = "Variable {}: Marking {} as an "
                            msg += "interesting value"
                            logger.info(msg.format(variable.id, idx))
                        variable.interesting_values += [idx]
                    else:
                        fraction = counts[idx] / total_count
                        if fraction > 0.05 and fraction < 0.95:
                            if verbose:
                                msg = "Variable {}: Marking {} as an "
                                msg += "interesting value"
                                logger.info(msg.format(variable.id, idx))
                            variable.interesting_values += [idx]
                            # total_count -= counts[idx]
                        else:
                            break

    def add_variable(self, new_id, type=None, data=None):
        """Add variable to entity

        Args:
            new_id (str) : Id of variable to be added.
            type (Variable) : Class of variable.
            data (pd.Series) : Variable's data to be placed in entity's dataframe
        """
        if new_id in [v.id for v in self.variables]:
            logger.warning("Not adding duplicate variable: %s", new_id)
            return
        if data is not None:
            self.df[new_id] = data

        if type is None:
            assert data in self.df.columns, "Must provide data to infer type"
            existing_columns = [c for c in self.df.columns if c.id != new_id]
            type = self.infer_variable_types(ignore=existing_columns)[new_id]

        new_v = type(new_id, entity=self)
        self.variables.append(new_v)

    def delete_variable(self, variable_id):
        """
        Remove variable from entity's dataframe and from
        self.variables
        """
        self.df.drop(variable_id, axis=1, inplace=True)
        v = self._get_variable(variable_id)
        self.variables.remove(v)

    def set_time_index(self, variable_id, already_sorted=False):
        if variable_id is not None:
            # check time type
            if self.df.empty:
                time_to_check = vtypes.DEFAULT_DTYPE_VALUES[self[variable_id]._default_pandas_dtype]
            else:
                time_to_check = self.df[variable_id].iloc[0]

            time_type = _check_time_type(time_to_check)
            if time_type is None:
                raise TypeError("%s time index not recognized as numeric or"
                                " datetime" % (self.id))

            if self.entityset.time_type is None:
                self.entityset.time_type = time_type
            elif self.entityset.time_type != time_type:
                raise TypeError("%s time index is %s type which differs from"
                                " other entityset time indexes" %
                                (self.id, time_type))

            # use stable sort
            if not already_sorted:
                # sort by time variable, then by index
                self.df.sort_values([variable_id, self.index], inplace=True)

            t = vtypes.NumericTimeIndex
            if col_is_datetime(self.df[variable_id]):
                t = vtypes.DatetimeTimeIndex
            self.convert_variable_type(variable_id, t, convert_data=False)
        else:
            # todo add test for this
            if not already_sorted:
                # sort by index
                self.df.sort_index(kind="mergesort",
                                   inplace=True)

        self.time_index = variable_id

    def set_index(self, variable_id, unique=True):
        """
        Args:
            variable_id (string) : Name of an existing variable to set as index.
            unique (bool) : Whether to assert that the index is unique.
        """
        self.df = self.df.set_index(self.df[variable_id], drop=False)
        self.df.index.name = None
        if unique:
            assert self.df.index.is_unique, "Index is not unique on dataframe (Entity {})".format(self.id)

        self.convert_variable_type(variable_id, vtypes.Index, convert_data=False)
        self.index = variable_id

    def set_secondary_time_index(self, secondary_time_index):
        if secondary_time_index is not None:
            for time_index in secondary_time_index:
                if self.df.empty:
                    time_to_check = vtypes.DEFAULT_DTYPE_VALUES[self[time_index]._default_pandas_dtype]
                else:
                    time_to_check = self.df[time_index].iloc[0]
                time_type = _check_time_type(time_to_check)
                if time_type is None:
                    raise TypeError("%s time index not recognized as numeric or"
                                    " datetime" % (self.id))
                if self.entityset.time_type != time_type:
                    raise TypeError("%s time index is %s type which differs from"
                                    " other entityset time indexes" %
                                    (self.id, time_type))
        self.secondary_time_index = secondary_time_index or {}

    def _vals_to_series(self, instance_vals, variable_id):
        """
        instance_vals may be a pd.Dataframe, a pd.Series, a list, a single
        value, or None. This function always returns a Series or None.
        """
        if instance_vals is None:
            return None

        # If this is a single value, make it a list
        if not hasattr(instance_vals, '__iter__'):
            instance_vals = [instance_vals]

        # convert iterable to pd.Series
        if type(instance_vals) == pd.DataFrame:
            out_vals = instance_vals[variable_id]
        elif type(instance_vals) == pd.Series:
            out_vals = instance_vals.rename(variable_id)
        else:
            out_vals = pd.Series(instance_vals, name=variable_id)

        # we've had weird problem with pandas read-only errors
        out_vals = copy.deepcopy(out_vals)
        # no duplicates or NaN values
        return pd.Series(out_vals).drop_duplicates().dropna()

    def _filter_and_sort(self, df, time_last=None,
                         training_window=None,
                         columns=None, sortby=None,
                         start=None, end=None,
                         shuffle=False, random_seed=None):
        """
        Filter a dataframe for all instances before time_last.
        If this entity does not have a time index, return the original
        dataframe.
        """
        if self.time_index:
            if time_last is not None and not df.empty:
                df = df[df[self.time_index] <= time_last]
                if training_window is not None:
                    mask = df[self.time_index] >= time_last - training_window
                    if self.last_time_index is not None:
                        lti_slice = self.last_time_index.reindex(df.index)
                        lti_mask = lti_slice >= time_last - training_window
                        mask = mask | lti_mask
                    else:
                        logger.warning(
                            "Using training_window but last_time_index is "
                            "not set on entity %s" % (self.id)
                        )
                    df = df[mask]

        for secondary_time_index in self.secondary_time_index:
                # should we use ignore time last here?
            if time_last is not None and not df.empty:
                mask = df[secondary_time_index] >= time_last
                second_time_index_columns = self.secondary_time_index[secondary_time_index]
                df.loc[mask, second_time_index_columns] = np.nan

        if columns is not None:
            df = df[columns]

        if sortby is not None:
            cat_vals = df[sortby]
            df_sort = df[[sortby]].copy()
            df_sort[sortby] = df_sort[sortby].astype("category")
            df_sort[sortby].cat.set_categories(cat_vals, inplace=True)
            df_sort.sort_values(sortby, inplace=True)

            # TODO: consider also using .loc[df_sort.index]
            df = df.reindex(df_sort.index, copy=False)

        if shuffle:
            df = df.sample(frac=1, random_state=random_seed)

        if start is not None and end is None:
            df = df.iloc[start:df.shape[0]]
        elif start is not None:
            df = df.iloc[start:end]
        elif end is not None:
            df = df.iloc[0:end]

        return df.copy()


def col_is_datetime(col):
    if (col.dtype.name.find('datetime') > -1 or
            (len(col) and isinstance(col.iloc[0], datetime))):
        return True

    # TODO: not sure this is ideal behavior.
    # it converts int columns that have dtype=object to datetimes starting from 1970
    elif col.dtype.name.find('str') > -1 or col.dtype.name.find('object') > -1:
        try:
            pd.to_datetime(col.dropna().iloc[:10], errors='raise')
        except Exception:
            return False
        else:
            return True
    return False
