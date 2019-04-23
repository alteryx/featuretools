# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import logging
from builtins import range

import numpy as np
import pandas as pd
import pandas.api.types as pdtypes

from .timedelta import Timedelta

from featuretools import variable_types as vtypes
from featuretools.utils import is_string
from featuretools.utils.entity_utils import (
    col_is_datetime,
    convert_all_variable_data,
    convert_variable_data,
    get_linked_vars,
    infer_variable_types
)
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

    def __init__(self, id, df, entityset, variable_types=None,
                 index=None, time_index=None, secondary_time_index=None,
                 last_time_index=None, already_sorted=False, make_index=False,
                 verbose=False):
        """ Create Entity

        Args:
            id (str): Id of Entity.
            df (pd.DataFrame): Dataframe providing the data for the
                entity.
            entityset (EntitySet): Entityset for this Entity.
            variable_types (dict[str -> dict[str -> type]]) : An entity's
                variable_types dict maps string variable ids to types (:class:`.Variable`)
                or (type, kwargs) to pass keyword arguments to the Variable.
            index (str): Name of id column in the dataframe.
            time_index (str): Name of time column in the dataframe.
            secondary_time_index (dict[str -> str]): Dictionary mapping columns
                in the dataframe to the time index column they are associated with.
            last_time_index (pd.Series): Time index of the last event for each
                instance across all child entities.
            make_index (bool, optional) : If True, assume index does not exist as a column in
                dataframe, and create a new column of that name using integers the (0, len(dataframe)).
                Otherwise, assume index exists in dataframe.
        """
        _validate_entity_params(id, df, time_index)
        created_index, index, df = _create_index(index, make_index, df)

        self.id = id
        self.entityset = entityset
        self.data = {'df': df, 'last_time_index': last_time_index}
        self.created_index = created_index
        self._verbose = verbose

        secondary_time_index = secondary_time_index or {}
        self._create_variables(variable_types, index, time_index, secondary_time_index)

        self.df = df[[v.id for v in self.variables]]
        self.set_index(index)

        self.time_index = None
        if time_index:
            self.set_time_index(time_index, already_sorted=already_sorted)
        elif not already_sorted:
            self.df.sort_index(kind="mergesort", inplace=True)
        self.set_secondary_time_index(secondary_time_index)

    def __repr__(self):
        repr_out = u"Entity: {}\n".format(self.id)
        repr_out += u"  Variables:"
        for v in self.variables:
            repr_out += u"\n    {} (dtype: {})".format(v.id, v.type_string)

        shape = self.shape
        repr_out += u"\n  Shape:\n    (Rows: {}, Columns: {})".format(
            shape[0], shape[1])

        # encode for python 2
        if type(repr_out) != str:
            repr_out = repr_out.encode("utf-8")

        return repr_out

    @property
    def shape(self):
        '''Shape of the entity's dataframe'''
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
    def df(self):
        '''Dataframe providing the data for the entity.'''
        return self.data["df"]

    @df.setter
    def df(self, _df):
        self.data["df"] = _df

    @property
    def last_time_index(self):
        '''
        Time index of the last event for each instance across all child entities.
        '''
        return self.data["last_time_index"]

    @last_time_index.setter
    def last_time_index(self, lti):
        self.data["last_time_index"] = lti

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
        '''Dictionary mapping variable id's to variable types'''
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
            >>> from featuretools.tests.testing_utils import make_ecommerce_entityset
            >>> es = make_ecommerce_entityset()
            >>> es["customers"].convert_variable_type("engagement_level", vtypes.Categorical)
        """
        if convert_data:
            # first, convert the underlying data (or at least try to)
            self.df = convert_variable_data(df=self.df,
                                            column_id=variable_id,
                                            new_type=new_type,
                                            **kwargs)

        # replace the old variable with the new one, maintaining order
        variable = self._get_variable(variable_id)
        new_variable = new_type.create_from(variable)
        self.variables[self.variables.index(variable)] = new_variable

    def query_by_values(self, instance_vals, variable_id=None, columns=None,
                        time_last=None, training_window=None):
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

        Returns:
            pd.DataFrame : instances that match constraints with ids in order of underlying dataframe
        """
        instance_vals = self._vals_to_series(instance_vals, variable_id)

        training_window = _check_timedelta(training_window)
        if training_window is not None:
            assert (isinstance(training_window, Timedelta) and
                    training_window.is_absolute()),\
                "training window must be an absolute Timedelta"

        if instance_vals is None:
            df = self.df.copy()

        elif instance_vals.shape[0] == 0:
            df = self.df.head(0)

        elif variable_id is None or variable_id == self.index:
            df = self.df.reindex(instance_vals)
            df.dropna(subset=[self.index], inplace=True)

        else:
            df = self.df[self.df[variable_id].isin(instance_vals)]

            df = df.set_index(self.index, drop=False)

            # ensure filtered df has same categories as original
            # workaround for issue below
            # github.com/pandas-dev/pandas/issues/22501#issuecomment-415982538
            if pdtypes.is_categorical_dtype(self.df[variable_id]):
                categories = pd.api.types.CategoricalDtype(categories=self.df[variable_id].cat.categories)
                df[variable_id] = df[variable_id].astype(categories)

        df = self._handle_time(df=df,
                               time_last=time_last,
                               training_window=training_window)

        if columns is not None:
            df = df[columns]

        return df

    def _create_variables(self, variable_types, index, time_index, secondary_time_index):
        """Extracts the variables from a dataframe

        Args:
            variable_types (dict[str -> dict[str -> type]]) : An entity's
                variable_types dict maps string variable ids to types (:class:`.Variable`)
                or (type, kwargs) to pass keyword arguments to the Variable.
            index (str): Name of index column
            time_index (str or None): Name of time_index column
            secondary_time_index (dict[str: [str]]): Dictionary of secondary time columns
                that each map to a list of columns that depend on that secondary time
        """
        variables = []
        variable_types = variable_types or {}
        if index not in variable_types:
            variable_types[index] = vtypes.Index

        link_vars = get_linked_vars(self)
        inferred_variable_types = infer_variable_types(self.df,
                                                       link_vars,
                                                       variable_types,
                                                       time_index,
                                                       secondary_time_index)
        inferred_variable_types.update(variable_types)

        for v in inferred_variable_types:
            # TODO document how vtype can be tuple
            vtype = inferred_variable_types[v]
            if isinstance(vtype, tuple):
                # vtype is (ft.Variable, dict_of_kwargs)
                _v = vtype[0](v, self, **vtype[1])
            else:
                _v = inferred_variable_types[v](v, self)
            variables += [_v]
        # convert data once we've inferred
        self.df = convert_all_variable_data(df=self.df,
                                            variable_types=inferred_variable_types)
        # make sure index is at the beginning
        index_variable = [v for v in variables
                          if v.id == index][0]
        self.variables = [index_variable] + [v for v in variables
                                             if v.id != index]

    def update_data(self, df, already_sorted=False,
                    recalculate_last_time_indexes=True):
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
        if self.time_index is not None:
            self.set_time_index(self.time_index, already_sorted=already_sorted)
        elif not already_sorted:
            self.df.sort_index(kind="mergesort", inplace=True)
        self.set_secondary_time_index(self.secondary_time_index)
        if recalculate_last_time_indexes and self.last_time_index is not None:
            self.entityset.add_last_time_indexes(updated_entities=[self.id])
        self.entityset.reset_data_description()

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

        self.entityset.reset_data_description()

    def delete_variable(self, variable_id):
        """
        Remove variable from entity's dataframe and from
        self.variables
        """
        self.df.drop(variable_id, axis=1, inplace=True)
        v = self._get_variable(variable_id)
        self.variables.remove(v)

    def set_time_index(self, variable_id, already_sorted=False):
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
        for time_index, columns in secondary_time_index.items():
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
            if time_index not in columns:
                columns.append(time_index)

        self.secondary_time_index = secondary_time_index

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
            out_vals = pd.Series(instance_vals)

        # no duplicates or NaN values
        out_vals = out_vals.drop_duplicates().dropna()

        # want index to have no name for the merge in query_by_values
        out_vals.index.name = None

        return out_vals

    def _handle_time(self, df, time_last=None,
                     training_window=None,
                     columns=None):
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

        for secondary_time_index, columns in self.secondary_time_index.items():
            # should we use ignore time last here?
            if time_last is not None and not df.empty:
                mask = df[secondary_time_index] >= time_last
                df.loc[mask, columns] = np.nan

        return df


def _create_index(index, make_index, df):
    '''Handles index creation logic base on user input'''
    created_index = None

    if index is None:
        # Case 1: user wanted to make index but did not specify column name
        assert not make_index, "Must specify an index name if make_index is True"
        # Case 2: make_index not specified but no index supplied, use first column
        logger.warning(("Using first column as index. ",
                        "To change this, specify the index parameter"))
        index = df.columns[0]
    elif make_index and index in df.columns:
        # Case 3: user wanted to make index but column already exists
        raise RuntimeError("Cannot make index: index variable already present")
    elif index not in df.columns:
        if not make_index:
            # Case 4: user names index, it is not in df. does not specify
            # make_index.  Make new index column and warn
            logger.warning("index %s not found in dataframe, creating new "
                           "integer column", index)
        # Case 5: make_index with no errors or warnings
        # (Case 4 also uses this code path)
        df.insert(0, index, range(0, len(df)))
        created_index = index
    # Case 6: user specified index, which is already in df. No action needed.
    return created_index, index, df


def _validate_entity_params(id, df, time_index):
    '''Validation checks for Entity inputs'''
    assert is_string(id), "Entity id must be a string"
    assert len(df.columns) == len(set(df.columns)), "Duplicate column names"
    for c in df.columns:
        if not is_string(c):
            raise ValueError("All column names must be strings (Column {} "
                             "is not a string)".format(c))
    if time_index is not None and time_index not in df.columns:
        raise LookupError('Time index not found in dataframe')
