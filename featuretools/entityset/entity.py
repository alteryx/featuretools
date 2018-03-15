from __future__ import division, print_function

import copy
import logging
import time
from builtins import range
from datetime import datetime

import numpy as np
import pandas as pd

from .base_entity import BaseEntity
from .timedelta import Timedelta

from featuretools import variable_types as vtypes
from featuretools.utils.wrangle import _check_timedelta

logger = logging.getLogger('featuretools.entityset')

_numeric_types = vtypes.PandasTypes._pandas_numerics
_categorical_types = [vtypes.PandasTypes._categorical]
_datetime_types = vtypes.PandasTypes._pandas_datetimes


class Entity(BaseEntity):
    """
    Stores all actual data for an entity
    """
    indexed_by = None

    def __init__(self, id, df, entityset, variable_types=None, name=None,
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
            name (str): Name of entity.
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
        assert len(df.columns) == len(set(df.columns)), "Duplicate column names"
        self.df = df
        self.encoding = encoding
        self.indexed_by = {}
        self._verbose = verbose
        self.created_index = created_index
        self.convert_variable_types(variable_types)
        self.attempt_cast_index_to_int(index)
        self.last_time_index = last_time_index
        super(Entity, self).__init__(id, entityset, variable_types, name, index,
                                     time_index, secondary_time_index, relationships, already_sorted)

    def attempt_cast_index_to_int(self, index_var):
        dtype_name = self.df[index_var].dtype.name
        if (dtype_name.find('int') == -1 and
                dtype_name.find('object') > -1 or dtype_name.find('categ') > -1):
            if isinstance(self.df[index_var].iloc[0], (int, np.int32, np.int64)):
                try:
                    self.df[index_var] = self.df[index_var].astype(int)
                except ValueError:
                    pass

    def convert_variable_types(self, variable_types):
        for var_id, desired_type in variable_types.items():
            type_args = {}
            if isinstance(desired_type, tuple):
                # grab args before assigning type
                type_args = desired_type[1]
                desired_type = desired_type[0]

            if var_id not in self.df.columns:
                raise LookupError("Variable ID %s not in DataFrame" % (var_id))
            current_type = self.get_column_type(var_id)

            if issubclass(desired_type, vtypes.Numeric) and \
                    current_type not in _numeric_types:
                self.entityset_convert_variable_type(var_id, desired_type, **type_args)

            if issubclass(desired_type, vtypes.Discrete) and \
                    current_type not in _categorical_types:
                self.entityset_convert_variable_type(var_id, desired_type, **type_args)

            if issubclass(desired_type, vtypes.Datetime) and \
                    current_type not in _datetime_types:
                self.entityset_convert_variable_type(var_id, desired_type, **type_args)

    def normalize(self, normalizer):
        d = {k: v for k, v in self.__dict__.items()
             if k not in ['df', 'indexed_by', 'entityset']}
        return normalizer(d)

    @property
    def num_instances(self):
        return self.df.shape[0]

    def get_shape(self):
        return self.df.shape

    def is_index_column(self, varname):
        if varname == self.index:
            return True

        elif varname.lower() == 'id':
            return True

        return False

    def head(self, n=10, cutoff_time=None):
        """See first n instance in entity

        Args:
            n (int) : Number of instances to return.
            cutoff_time (pd.Timestamp,pd.DataFrame) : Timestamp(s) to restrict rows.

        Returns:
            :class:`pd.DataFrame` : A Pandas DataFrame.

        """

        if cutoff_time is None:
            valid_data = self.df

        elif isinstance(cutoff_time, pd.Timestamp) or \
                isinstance(cutoff_time, datetime):
            valid_data = self.df[self.df[self.time_index] < cutoff_time]

        elif isinstance(cutoff_time, pd.DataFrame):

            instance_ids, time = list(cutoff_time)

            # TODO filtering the top n during "isin" would be more efficient
            valid_data = self.df[
                self.df[self.index].isin(cutoff_time[instance_ids])]
            valid_data = valid_data[
                valid_data[self.time_index] < cutoff_time[time]]

        else:
            raise ValueError(
                'cutoff_time must be None, a Datetime, a pd.Timestamp, or a pd.DataFrame')

        return valid_data.head(n)

    def get_column_type(self, column_id):
        """ get type of column in underlying data structure """
        return self.df[column_id].dtype.name

    def get_column_stat(self, column_id, stat):
        """ maximum value """
        if column_id not in self.df.columns:
            raise AttributeError(u"%s not in entity" % (column_id))
        s = getattr(self.df[column_id], stat)()
        return s

    def get_column_max(self, column_id):
        return self.get_column_stat(column_id, 'max')

    def get_column_min(self, column_id):
        return self.get_column_stat(column_id, 'min')

    def get_column_std(self, column_id):
        return self.get_column_stat(column_id, 'std')

    def get_column_count(self, column_id):
        return self.get_column_stat(column_id, 'count')

    def get_column_mean(self, column_id):
        return self.get_column_stat(column_id, 'mean')

    def get_column_nunique(self, column_id):
        return self.get_column_stat(column_id, 'nunique')

    def get_all_instances(self):
        instance_df = self.query_by_values(None,
                                           columns=[self.index])

        return instance_df[self.index].values

    def get_top_n_instances(self, top_n):
        instance_df = self.query_by_values(None,
                                           columns=[self.index])

        return instance_df[self.index].values[:top_n]

    def sample_instances(self, n, random_seed=None):
        instance_df = self.query_by_values(None,
                                           columns=[self.index])

        n = min(instance_df.shape[0], n)
        return instance_df[self.index].sample(n, random_state=random_seed).values

    def get_sliced_instance_ids(self, start, end, random_seed=None, shuffle=False):
        instance_df = self.query_by_values(None,
                                           columns=[self.index],
                                           start=start,
                                           end=end,
                                           random_seed=random_seed,
                                           shuffle=shuffle)

        return instance_df[self.index].values

    def get_column_data(self, column_id):
        """ get data from column in specified form """
        return self.df[column_id]

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
            training_window (dict[str -> Timedelta] or Timedelta, optional):
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
            df = self.df.loc[instance_vals]
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

        self.index_by_variable(relation_var_id)

    def index_by_variable(self, variable_id):
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
            elif self.is_index_column(variable):
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

    def update_data(self, df):
        self.df = df
        self.add_all_variable_statistics()

    def get_sample(self, n):
        df = self.df
        n = min(n, len(df))
        sampled = df.sample(n)
        self.df = sampled
        indexed_by = self.indexed_by
        self.indexed_by = {}
        copied = copy.copy(self)
        self.df = df
        self.indexed_by = indexed_by
        for variable in copied.variables:
            variable.entity = copied
        return copied

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

    def add_column(self, column_id, column_data, type=None):
        """
        Add variable to entity's dataframe
        """
        existing_columns = list(self.df.columns)
        self.df[column_id] = column_data
        if type is None:
            type = self.infer_variable_types(ignore=existing_columns)[column_id]
        self.variable_types[column_id] = type

    def delete_column(self, column_id):
        """
        Remove variable from entity's dataframe
        """
        self.df.drop(column_id, axis=1, inplace=True)
        del self.variable_types[column_id]

    def entityset_convert_variable_type(self, column_id, new_type, **kwargs):
        """
        Convert variable in data set to different type
        """
        df = self.df
        if df[column_id].empty:
            return
        if new_type == vtypes.Numeric:
            df[column_id] = pd.to_numeric(df[column_id], errors='coerce')
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

    def set_time_index(self, variable_id, already_sorted=False):
        if variable_id is not None:
            # use stable sort
            if not already_sorted:
                # sort by time variable, then by index
                self.df.sort_values([variable_id, self.index],
                                    kind="mergesort",
                                    inplace=True)

            t = vtypes.TimeIndex
            if col_is_datetime(self.df[variable_id]):
                t = vtypes.DatetimeTimeIndex
            self.convert_variable_type(variable_id, t, convert_data=False)
        else:
            # todo add test for this
            if not already_sorted:
                # sort by time variable, then by index
                self.df.sort_values([self.index],
                                    kind="mergesort",
                                    inplace=True)

        super(Entity, self).set_time_index(variable_id)

    def set_index(self, variable_id, unique=True):
        """
        Args:
            variable_id (string) : Name of an existing variable to set as index.
            unique (bool) : Whether to assert that the index is unique.
        """
        self.df = self.df.set_index(self.df[variable_id], drop=False)
        if unique:
            assert self.df.index.is_unique, "Index is not unique on dataframe (Entity {})".format(self.id)

        self.convert_variable_type(variable_id, vtypes.Index, convert_data=False)

        super(Entity, self).set_index(variable_id)

    def set_last_time_index(self, last_time_index):
        self.last_time_index = last_time_index

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
                # TODO: make sure this try/except is a good idea
                try:
                    df.iloc[0][self.time_index] <= time_last
                except TypeError:
                    pass
                else:
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
