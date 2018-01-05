import copy
import itertools
import logging
from builtins import range, zip
from collections import defaultdict

import dask.dataframe as dd
import numpy as np
import pandas as pd

from .base_entityset import BaseEntitySet
from .entity import Entity
from .relationship import Relationship
from .serialization import read_pickle, to_pickle

import featuretools.variable_types.variable as vtypes
from featuretools.utils.gen_utils import make_tqdm_iterator
from featuretools.utils.wrangle import _check_variable_list

pd.options.mode.chained_assignment = None  # default='warn'
logger = logging.getLogger('featuretools.entityset')


class EntitySet(BaseEntitySet):
    """
    Stores all actual data for a entityset

    Attributes:
        entity_stores
    """

    def __init__(self, id, entities=None, relationships=None, verbose=False):
        """Creates EntitySet

            Args:
                id (str) : unique identifier to associate with this instance
                verbose (boolean)

                entities (dict[str: tuple(pd.DataFrame, str, str)]): dictionary of
                    entities. Entries take the format
                    {entity id: (dataframe, id column, (time_column), (variable_types))}
                    Note that time_column and variable_types are optional

                relationships (list[(str, str, str, str)]): list of relationships
                    between entities. List items are a tuple with the format
                    (parent entity id, parent variable, child entity id, child variable)

            Example:

                .. code-block:: python

                    entities = {
                        "cards" : (card_df, "id"),
                        "transactions" : (transactions_df, "id", "transaction_time")
                    }

                    relationships = [("cards", "id", "transactions", "card_id")]

                    ft.EntitySet("my-entity-set", entities, relationships)
        """
        super(EntitySet, self).__init__(id, verbose)

        entities = entities or {}
        relationships = relationships or []
        for entity in entities:
            df = entities[entity][0]
            index_column = entities[entity][1]
            time_column = None
            variable_types = None
            if len(entities[entity]) > 2:
                time_column = entities[entity][2]
            if len(entities[entity]) > 3:
                variable_types = entities[entity][3]
            self.entity_from_dataframe(entity_id=entity,
                                       dataframe=df,
                                       index=index_column,
                                       time_index=time_column,
                                       variable_types=variable_types)

        for relationship in relationships:
            parent_variable = self[relationship[0]][relationship[1]]
            child_variable = self[relationship[2]][relationship[3]]
            self.add_relationship(Relationship(parent_variable,
                                               child_variable))

    def normalize(self, normalizer):
        return super(EntitySet, self).normalize(normalizer=normalizer, remove_entityset=False)

    @property
    def entity_names(self):
        """
        Return list of each entity's id
        """
        return [e.id for e in self.entities]

    def to_pickle(self, path):
        to_pickle(self, path)
        return self

    @classmethod
    def read_pickle(cls, path):
        return read_pickle(path)

    ###########################################################################
    #  Public API methods  ###################################################
    ###########################################################################

    # Read-only entityset-level methods

    def get_sample(self, n):
        full_entities = {}
        for eid, entity in self.entity_stores.items():
            full_entities[eid] = self.entity_stores[eid]
            self.entity_stores[eid] = entity.get_sample(n)
        sampled = copy.copy(self)
        for entity in sampled.entity_stores.values():
            entity.entityset = sampled
        self.entity_stores = full_entities
        return sampled

    def head(self, entity_id, n=10, variable_id=None, cutoff_time=None):
        if variable_id is None:
            return self.entity_stores[entity_id].head(
                n, cutoff_time=cutoff_time)
        else:
            return self.entity_stores[entity_id].head(
                n, cutoff_time=cutoff_time)[variable_id]

    def get_instance_data(self, entity_id, instance_ids):
        return self.entity_stores[entity_id].query_by_values(instance_ids)

    def num_instances(self, entity_id):
        entity = self.entity_stores[entity_id]
        return entity.num_instances

    def get_all_instances(self, entity_id):
        entity = self.entity_stores[entity_id]
        return entity.get_all_instances()

    def get_top_n_instances(self, entity_id, top_n=10):
        entity = self.entity_stores[entity_id]
        return entity.get_top_n_instances(top_n)

    def sample_instances(self, entity_id, n=10):
        entity = self.entity_stores[entity_id]
        return entity.sample_instances(n)

    def get_sliced_instance_ids(self, entity_id, start, end, random_seed=None, shuffle=False):
        entity = self.entity_stores[entity_id]
        return entity.get_sliced_instance_ids(start, end, random_seed=random_seed, shuffle=shuffle)

    def get_pandas_data_slice(self, filter_entity_ids, index_eid,
                              instances, time_last=None, training_window=None,
                              verbose=False):
        """
        Get the slice of data related to the supplied instances of the index
        entity.
        """
        eframes_by_filter = {}

        if verbose:
            iterator = make_tqdm_iterator(iterable=filter_entity_ids,
                                          desc="Gathering relevant data",
                                          unit="entity")
        else:
            iterator = filter_entity_ids
        # gather frames for each child, for each parent
        for filter_eid in iterator:
            # get the instances of the top-level entity linked by our instances
            toplevel_slice = self._related_instances(start_entity_id=index_eid,
                                                     final_entity_id=filter_eid,
                                                     instance_ids=instances,
                                                     time_last=time_last,
                                                     training_window=training_window)

            eframes = {filter_eid: toplevel_slice}

            # Do a bredth-first search of the relationship tree rooted at this
            # entity, filling out eframes for each entity we hit on the way.
            r_queue = self.get_backward_relationships(filter_eid)
            while r_queue:
                r = r_queue.pop(0)
                child_eid = r.child_variable.entity.id
                parent_eid = r.parent_variable.entity.id

                # If we've already seen this child, this is a diamond graph and
                # we don't know what to do
                if child_eid in eframes:
                    raise RuntimeError('Diamond graph detected!')

                # Add this child's children to the queue
                r_queue += self.get_backward_relationships(child_eid)

                # Query the child of the current backwards relationship for the
                # instances we want
                instance_vals = eframes[parent_eid][r.parent_variable.id]
                eframes[child_eid] =\
                    self.entity_stores[child_eid].query_by_values(
                        instance_vals, variable_id=r.child_variable.id,
                        time_last=time_last, training_window=training_window)

                # add link variables to this dataframe in order to link it to its
                # (grand)parents
                self._add_multigenerational_link_vars(frames=eframes,
                                                      start_entity_id=filter_eid,
                                                      end_entity_id=child_eid)

            eframes_by_filter[filter_eid] = eframes

        # If there are no instances of *this* entity in the index, return None
        if eframes_by_filter[index_eid][index_eid].shape[0] == 0:
            return None

        return eframes_by_filter

    # Read-only entity-level methods

    def get_dataframe(self, entity_id):
        """
        Get the data for a specified entity as a pandas dataframe.
        """
        return self.entity_stores[entity_id].df

    def get_column_names(self, entity_id):
        """
        Return a list of the columns on the underlying data store
        """
        return self.entity_stores[entity_id].df.columns

    def get_index(self, entity_id):
        """
        Get name of the primary key ID column for this entity
        """
        return self.entity_stores[entity_id].index

    def get_time_index(self, entity_id):
        """
        Get name of the time index column for this entity
        """
        return self.entity_stores[entity_id].time_index

    def get_secondary_time_index(self, entity_id):
        """
        Get names and associated variables of the secondary time index columns for this entity
        """
        return self.entity_stores[entity_id].secondary_time_index

    def query_entity_by_values(self, entity_id, instance_vals, variable_id=None,
                               columns=None, time_last=None,
                               return_sorted=False):
        """
        Query entity for all rows which have one of instance_vals in the
        variable_id column.
        """
        estore = self.entity_stores[entity_id]
        return estore.query_by_values(instance_vals,
                                      variable_id=variable_id,
                                      columns=columns,
                                      time_last=time_last,
                                      return_sorted=return_sorted)

    # Read-only variable-level methods

    def get_column_type(self, entity_id, column_id):
        """ get type of column in underlying data structure """
        return self.entity_stores[entity_id].get_column_type(column_id)

    def get_column_stat(self, eid, column_id, stat):
        return self.entity_stores[eid].get_column_stat(column_id, stat)

    def get_column_max(self, eid, column_id):
        return self.get_column_stat(eid, column_id, 'max')

    def get_column_min(self, eid, column_id):
        return self.get_column_stat(eid, column_id, 'min')

    def get_column_std(self, eid, column_id):
        return self.get_column_stat(eid, column_id, 'std')

    def get_column_count(self, eid, column_id):
        return self.get_column_stat(eid, column_id, 'count')

    def get_column_mean(self, eid, column_id):
        return self.get_column_stat(eid, column_id, 'mean')

    def get_column_nunique(self, eid, column_id):
        return self.get_column_stat(eid, column_id, 'nunique')

    def get_column_data(self, entity_id, column_id):
        """ get data from column in specified form """
        return self.entity_stores[entity_id].get_column_data(column_id)

    def get_variable_types(self, entity_id):
        return self.entity_stores[entity_id].get_variable_types()

    # Read-write variable-level methods

    def add_column(self, entity_id, column_id, column_data, type=None):
        """
        Add variable to entity's dataframe
        """
        self.entity_stores[entity_id].add_column(column_id, column_data, type=type)

    def delete_column(self, entity_id, column_id):
        """
        Remove variable from entity's dataframe
        """
        self.entity_stores[entity_id].delete_column(column_id)

    def store_convert_variable_type(self, entity_id, column_id, new_type, **kwargs):
        """
        Convert variable in data set to different type
        """
        # _operations?
        self.entity_stores[entity_id].convert_variable_type(column_id, new_type, **kwargs)

    # Read-write entity-level methods

    ###########################################################################
    #  Entity creation methods  ##############################################
    ###########################################################################
    def entity_from_csv(self, entity_id,
                        csv_path,
                        index=None,
                        variable_types=None,
                        use_variables=None,
                        make_index=False,
                        time_index=None,
                        secondary_time_index=None,
                        time_index_components=None,
                        parse_date_cols=None,
                        encoding=None,
                        **kwargs):
        """
        Load the data for a specified entity from a CSV file.

        Args:
            entity_id (str) : unique id to associate with this entity

            csv_path (str) : path to the file containing the data

            index (str, optional): Name of the variable used to index the entity.
                If None, take the first column

            variable_types (dict[str->dict[str->type]]) : Optional mapping of
                entity_id -> variable_types dict with which to initialize an
                entity's store.
                An entity's variable_types dict maps string variable ids to types (:class:`.Variable`)

            use_variables (Optional(list[str])) : List of column names to pull from csv

            make_index (Optional(boolean)) : If True, assume index does not exist as a column in
                csv, and create a new column of that name using integers the (0, len(dataframe)).
                Otherwise, assume index exists in csv

            time_index (Optional[str]): Name of the variable containing
                time data. Type must be in Variables.datetime or be able to be
                cast to datetime (e.g. str, float), or numeric.

            secondary_time_index (Optional[str]): Name of variable containing
                time data to use a second time index for the entity

            time_index_components (Optional((list[str])) : Names of columns to combine (separated by spaces)
                and allow Pandas to parse as a single Datetime time_index column. Columns are
                combined in the order provided. Useful if there are separate date and time
                columns (e.g. col1[0] = '8/8/2016' and col2[0] = '4:30')

            parse_date_cols (Optional(list[str])) : list of column names to parse as datetimes

            encoding (Optional(str)) : If None, will use 'ascii'. Another option is 'utf-8',
                or any encoding supported by pandas. Passed into underlying
                pandas.read_csv() and pandas.to_csv() calls, so see Pandas documentation
                for more information

            **kwargs : Extra arguments will be passed to :func:`pd.read_csv`
        """

        # If time index components are passed, combine them into a single column
        # TODO look into handling secondary_time_index here

        # _operations?
        if parse_date_cols:
            parse_date_cols = parse_date_cols or []

        def load_df(e, csv_path):
            ext = csv_path.split('.')[-1]
            compression = None
            compression_formats = ['bz2', 'gzip']
            if ext in compression_formats:
                compression = ext
            elif ext != 'csv':
                raise ValueError("Unknown extension: %s", ext)

            read_csv = pd.read_csv
            glob = False

            if "*" in csv_path:
                glob = True
                # read dask dataframe from multiple files
                read_csv = dd.read_csv
                kwargs['blocksize'] = None

            if time_index_components:
                parse_dates = {time_index: time_index_components}
                df = read_csv(csv_path, low_memory=False,
                              parse_dates=parse_dates,
                              compression=compression,
                              usecols=use_variables,
                              encoding=encoding,
                              **kwargs)
            else:
                df = read_csv(csv_path, low_memory=False,
                              parse_dates=parse_date_cols,
                              compression=compression,
                              usecols=use_variables,
                              encoding=encoding,
                              **kwargs)

            if glob:
                # convert dask dataframe back to pandas
                if e:
                    df = e.compute(df)
                else:
                    df = df.compute()
            return df

        # handle if we are working list of csv_path
        # TODO: handle multiple csvs using dask so that they will be downloaded in parallel
        dfs = []
        if not isinstance(csv_path, list):
            csv_path = [csv_path]

        # DFS todo: what about
        # # try:
        #         with worker_client() as e:
        #             df = load_df(e, csv)
        #     except AttributeError:
        for csv in csv_path:
            df = load_df(None, csv)
            dfs.append(df)

        df = pd.concat(dfs)

        return self._import_from_dataframe(entity_id, df, index=index,
                                           make_index=make_index,
                                           time_index=time_index,
                                           secondary_time_index=secondary_time_index,
                                           variable_types=variable_types,
                                           parse_date_cols=parse_date_cols,
                                           encoding=encoding)

    def entity_from_dataframe(self,
                              entity_id,
                              dataframe,
                              index=None,
                              variable_types=None,
                              make_index=False,
                              time_index=None,
                              secondary_time_index=None,
                              encoding=None,
                              already_sorted=False):
        """
        Load the data for a specified entity from a Pandas DataFrame.

        Args:
            entity_id (str) : unique id to associate with this entity

            dataframe (pandas.DataFrame) : dataframe containing the data

            index (str, optional): Name of the variable used to index the entity.
                If None, take the first column

            variable_types (dict[str->dict[str->type]]) : Optional mapping of
                entity_id -> variable_types dict with which to initialize an
                entity's store.
                An entity's variable_types dict maps string variable ids to types (:class:`.Variable`)

            make_index (Optional(boolean)) : If True, assume index does not exist as a column in
                csv, and create a new column of that name using integers the (0, len(dataframe)).
                Otherwise, assume index exists in csv

            time_index (Optional[str]): Name of the variable containing
                time data. Type must be in Variables.datetime or be able to be
                cast to datetime (e.g. str, float), or numeric.

            secondary_time_index (Optional[str]): Name of variable containing
                time data to use a second time index for the entity

            encoding (Optional[str]) : If None, will use 'ascii'. Another option is 'utf-8',
                or any encoding supported by pandas. Passed into underlying pandas.to_csv() calls,
                so see Pandas documentation for more information.

            already_sorted (Optional[boolean]) : If True, assumes that input dataframe is already sorted by time.
                Defaults to False.

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

        # If time index components are passed, combine them into a single column
        # TODO look into handling secondary_time_index here
        # _operations?
        return self._import_from_dataframe(entity_id, dataframe.copy(), index=index,
                                           make_index=make_index,
                                           time_index=time_index,
                                           secondary_time_index=secondary_time_index,
                                           variable_types=variable_types,
                                           encoding=encoding,
                                           already_sorted=already_sorted)

    def _import_from_dataframe(self,
                               entity_id,
                               dataframe,
                               index=None,
                               variable_types=None,
                               make_index=False,
                               time_index=None,
                               secondary_time_index=None,
                               last_time_index=None,
                               parse_date_cols=None,
                               encoding=None,
                               already_sorted=False):
        """
        Load the data for a specified entity from a pandas dataframe.

        Args:
            entity_id (str) : unique id to associate with this entity
            dataframe (:class:`.pd.DataFrame`) : Pandas dataframe containing the data
            index (str, optional): Name of the variable used to index the entity.
                If None, take the first column
            variable_types (dict[str->dict[str->type]]) : Optional mapping of
                entity_id -> variable_types dict with which to initialize an
                entity's store.
            make_index (Optional(boolean)) : If True, assume index does not exist as a column in
                dataframe, and create a new column of that name using integers the (0, len(dataframe)).
                Otherwise, assume index exists in dataframe
                An entity's variable_types dict maps string variable ids to types (:class:`.Variable`)
            time_index (Optional(str)) : Name of column to use as a time index for this entity. Must be
                a Datetime or Numeric dtype
            secondary_time_index (Optional[str]): Name of variable containing
                time data to use a second time index for the entity
            encoding (Optional[str]) : If None, will use 'ascii'. Another option is 'utf-8',
                or any encoding supported by pandas. Passed into underlying pandas.to_csv() calls,
                so see Pandas documentation for more information.
            already_sorted (Optional[boolean]) : If True, assumes that input dataframe is already sorted by time.
                Defaults to False.
        """
        variable_types = variable_types or {}

        # DFS TODO: confirm we want this if else block
        if index is None:
            assert not make_index, "Must specify an index name if make_index is True"
            logger.warning(("Using first column as index. ",
                            "To change this, specify the index parameter"))
        else:
            if index not in variable_types:
                variable_types[index] = vtypes.Index

        created_index = None
        if make_index or index not in dataframe.columns:
            if not make_index:
                logger.warning("index %s not found in dataframe, creating new integer column",
                               index)
            if index in dataframe.columns:
                raise RuntimeError("Cannot make index: index variable already present")
            dataframe.insert(0, index, range(0, len(dataframe)))
            created_index = index
        elif index is None:
            index = dataframe.columns[0]

        elif time_index is not None and time_index not in dataframe.columns:
            raise LookupError('Time index not found in dataframe')
        if parse_date_cols is not None:
            for c in parse_date_cols:
                variable_types[c] = vtypes.Datetime

        current_relationships = [r for r in self.relationships
                                 if r.parent_entity.id == entity_id or
                                 r.child_entity.id == entity_id]
        # "category" dtype generates a lot of errors because pandas treats it
        # differently in merges. In particular, if there are no matching
        # instances on a dataframe to join, the new joined dataframe turns out
        # to be empty if the column we're joining on is a category
        # When its any other dtype, we get the same number of rows as the
        # original dataframe but filled in with nans

        df = dataframe
        for c in df.columns:
            if df[c].dtype.name.find('category') > -1:
                df[c] = df[c].astype(object)
                if c not in variable_types:
                    variable_types[c] = vtypes.Categorical
        if df.index.dtype.name.find('category') > -1:
            df.index = df.index.astype(object)

        self.add_entity(entity_id,
                        df,
                        variable_types=variable_types,
                        index=index,
                        time_index=time_index,
                        secondary_time_index=secondary_time_index,
                        last_time_index=last_time_index,
                        encoding=encoding,
                        relationships=current_relationships,
                        already_sorted=already_sorted,
                        created_index=created_index)
        return self

    def add_entity(self,
                   entity_id,
                   df,
                   **kwargs):
        entity = Entity(entity_id,
                        df,
                        self,
                        verbose=self._verbose,
                        **kwargs)

        # TODO DFS: think about if we need both list and dictionary
        self.entity_stores[entity.id] = entity
        return entity

    def normalize_entity(self, base_entity_id, new_entity_id, index,
                         additional_variables=None, copy_variables=None,
                         convert_links_to_integers=False,
                         make_time_index=None,
                         make_secondary_time_index=None,
                         new_entity_time_index=None,
                         new_entity_secondary_time_index=None,
                         time_index_reduce='first', variable_types=None):
        """Utility to normalize an entity_store

        Args:
            base_entity_id (str) : entity id to split from

            new_entity_id (str): id of the new entity

            index (str): variable in old entity
                that will become index of new entity. Relationship
                will be across this variable.

            additional_variables (list[str]):
                list of variable ids to remove from
                base_entity and move to new entity

            copy_variables (list[str]): list of
                variable ids to copy from old entity
                and move to new enentity

            convert_links_to_integers (bool) : If True,
                convert the linking variable between the two
                entities to an integer. Old variable will be kept only
                in the new normalized entity, and the new variable will have
                the old variable's name plus "_id"

            make_time_index (bool or str, optional): create time index for new entity based
                on time index in base_entity, optionally specifying which variable in base_entity
                to use for time_index. If specified as True without a specific variable,
                uses the primary time index. Defaults to True is base entity has time index

            make_secondary_time_index (dict[str=>list[str]], optional): create secondary time index(es)
                for new entity based on secondary time indexes in base entity. Values of dictionary
                are the variables to associate with the secondary time index. Only one
                secondary time index is allowed. If values left blank, only associate the time index.


            new_entity_time_index (Optional[str]): rename new entity time index

            new_entity_secondary_time_index (Optional[str]): rename new entity secondary time index

            time_index_reduce (str): If making a time_index, choose either
                the 'first' time or the 'last' time from the associated children instances.
                If creating a secondary time index, then the primary time index always reduces
                using 'first', and secondary using 'last'

        """
        base_entity = self.entity_stores[base_entity_id]
        # variable_types = base_entity.variable_types
        additional_variables = additional_variables or []
        copy_variables = copy_variables or []
        for v in additional_variables + copy_variables:
            if v == index:
                raise ValueError("Not copying {} as both index and variable".format(v))
                break
        new_index = index

        if convert_links_to_integers:
            new_index = self.make_index_variable_name(new_entity_id)

        transfer_types = {}
        transfer_types[new_index] = type(base_entity[index])
        for v in additional_variables + copy_variables:
            transfer_types[v] = type(base_entity[v])

        # create and add new entity
        new_entity_df = self.get_dataframe(base_entity_id)

        if make_time_index is None and base_entity.has_time_index():
            make_time_index = True

        if isinstance(make_time_index, str):
            base_time_index = make_time_index
            new_entity_time_index = base_entity[make_time_index].id
        elif make_time_index:
            base_time_index = base_entity.time_index
            if new_entity_time_index is None:
                new_entity_time_index = "%s_%s_time" % (time_index_reduce, base_entity.id)

            assert base_entity.has_time_index(), \
                "Base entity doesn't have time_index defined"

            if base_time_index not in [v for v in additional_variables]:
                copy_variables.append(base_time_index)

            transfer_types[new_entity_time_index] = type(base_entity[base_entity.time_index])

            new_entity_df.sort_values([base_time_index, base_entity.index], kind="mergesort", inplace=True)
        else:
            new_entity_time_index = None

        selected_variables = [index] +\
            [v for v in additional_variables] +\
            [v for v in copy_variables]

        new_entity_df2 = new_entity_df. \
            drop_duplicates(index, keep=time_index_reduce)[selected_variables]

        if make_time_index:
            new_entity_df2.rename(columns={base_time_index: new_entity_time_index}, inplace=True)
        if make_secondary_time_index:
            time_index_reduce = 'first'

            assert len(make_secondary_time_index) == 1, "Can only provide 1 secondary time index"
            secondary_time_index = list(make_secondary_time_index.keys())[0]

            secondary_variables = [index, secondary_time_index] + list(make_secondary_time_index.values())[0]
            secondary_df = new_entity_df. \
                drop_duplicates(index, keep='last')[secondary_variables]
            if new_entity_secondary_time_index:
                secondary_df.rename(columns={secondary_time_index: new_entity_secondary_time_index},
                                    inplace=True)
                secondary_time_index = new_entity_secondary_time_index
            else:
                new_entity_secondary_time_index = secondary_time_index
            secondary_df.set_index(index, inplace=True)
            new_entity_df = new_entity_df2.join(secondary_df, on=index)
        else:
            new_entity_df = new_entity_df2

        base_entity_index = index
        if convert_links_to_integers:
            old_entity_df = self.get_dataframe(base_entity_id)
            link_variable_id = self.make_index_variable_name(new_entity_id)
            new_entity_df[link_variable_id] = np.arange(0, new_entity_df.shape[0])
            just_index = old_entity_df[[index]]
            id_as_int = just_index.merge(new_entity_df,
                                         left_on=index,
                                         right_on=index,
                                         how='left')[link_variable_id]

            old_entity_df.loc[:, index] = id_as_int.values

            base_entity.update_data(old_entity_df)
            index = link_variable_id

        # TODO dfs: do i need this?
        # for v_id in selected_variables:
        #     if v_id in variable_types and v_id not in transfer_types:
        #         transfer_types[v_id] = variable_types[v_id]

        transfer_types[index] = vtypes.Categorical
        self._import_from_dataframe(new_entity_id, new_entity_df,
                                    index,
                                    time_index=new_entity_time_index,
                                    secondary_time_index=make_secondary_time_index,
                                    last_time_index=None,
                                    variable_types=transfer_types,
                                    encoding=base_entity.encoding)

        for v in additional_variables:
            self.delete_column(base_entity_id, v)
        self.delete_entity_variables(base_entity_id, additional_variables)

        new_entity = self.entity_stores[new_entity_id]
        if make_secondary_time_index:
            values = make_secondary_time_index.values()[0]
            values.remove(make_secondary_time_index.keys()[0])
            new_dict = {secondary_time_index: values}

            new_entity.secondary_time_index = new_dict
            for ti, cols in new_entity.secondary_time_index.items():
                if ti not in cols:
                    cols.append(ti)

        base_entity.convert_variable_type(base_entity_index, vtypes.Id, convert_data=False)

        self.add_relationship(Relationship(new_entity[index], base_entity[base_entity_index]))

        return self

    ###########################################################################
    #  Data wrangling methods  ###############################################
    ###########################################################################

    # TODO dfs: where is this used? it doesn't seem tested either
    def add_parent_time_index(self, entity_id, parent_entity_id,
                              parent_time_index_variable=None,
                              child_time_index_variable=None,
                              include_secondary_time_index=False,
                              secondary_time_index_variables=[]):
        entity = self.entity_stores[entity_id]

        parent_entity = self.entity_stores[parent_entity_id]
        if parent_time_index_variable is None:
            parent_time_index_variable = parent_entity.time_index
            assert parent_time_index_variable is not None, ("If parent does not have a time index, ",
                                                            "you must specify which variable to use")
        msg = ("parent time index variable must be ",
               "a Datetime, Numeric, or Ordinal")
        assert isinstance(parent_entity[parent_time_index_variable], (vtypes.Numeric, vtypes.Ordinal, vtypes.Datetime)), msg

        self._add_parent_variable_to_df(entity_id, parent_entity_id,
                                        parent_time_index_variable,
                                        child_time_index_variable)
        entity.set_time_index(child_time_index_variable)
        if include_secondary_time_index:
            msg = "Parent entity has no secondary time index"
            assert len(parent_entity.secondary_time_index), msg
            parent_sec_ti_id = list(parent_entity.secondary_time_index.keys())[0]
            parent_sec_ti_vars = list(parent_entity.secondary_time_index.values())[0]
            if isinstance(secondary_time_index_variables, list):
                parent_sec_ti_vars = [v for v in parent_sec_ti_vars
                                      if v in secondary_time_index_variables]
            # TODO: arg to name child vars
            for v in [parent_sec_ti_id] + parent_sec_ti_vars:
                self._add_parent_variable_to_df(entity_id, parent_entity_id,
                                                v)
            new_secondary_time_index = {parent_sec_ti_id: parent_sec_ti_vars}
            entity.set_secondary_time_index(new_secondary_time_index)

    def _add_parent_variable_to_df(self, child_entity_id, parent_entity_id,
                                   parent_variable_id, child_variable_id=None):
        entity = self.entity_stores[child_entity_id]
        parent_entity = self.entity_stores[parent_entity_id]

        if child_variable_id is None:
            child_variable_id = parent_variable_id

        path = self.find_forward_path(child_entity_id, parent_entity_id)
        assert len(path) > 0, "must be a parent entity"
        if len(path) > 1:
            raise NotImplementedError("adding time index from a grandparent not yet supported")
        rel = path[0]

        child_data = entity.df
        # get columns of parent that we need and rename in prep for merge
        parent_data = self.entity_stores[parent_entity_id].df[[rel.parent_variable.id, parent_variable_id]]
        col_map = {parent_variable_id: child_variable_id}
        parent_data.rename(columns=col_map, inplace=True)

        # add parent time
        # use right index to avoid parent join key in merged dataframe
        parent_data.set_index(rel.parent_variable.id, inplace=True)
        new_child_data = child_data.merge(parent_data,
                                          left_on=rel.child_variable.id,
                                          right_index=True,
                                          how='left')

        # TODO: look in to using update_data method of Entity
        entity.df = new_child_data

        parent_type = type(parent_entity[parent_variable_id])
        entity.add_variable(child_variable_id, parent_type)
        entity.add_variable_statistics(child_variable_id)

    # todo dfs: this isn't united tested, and hasn't be manually verified to work
    def combine_variables(self, entity_id, new_id, to_combine,
                          drop=False, hashed=False, **kwargs):
        """Combines two variable into variable new_id

        Args:
            entity_id (str): ID of Entity to be modified
            new_id (str): Id of new variable being created
            to_combine (list[:class:`.Variable`] or list[str]): list of
                variables to combine
            drop (Optional[bool]): if True, variables that are combined are
                dropped from the entity
            hashed (Optional[bool]): if True, combination variables values are
                hashed, resulting in an integer column dtype. Otherwise, values
                are just concatenated.

        Note:
            underlying data for variable must be of type str

        """
        # _operations?
        entity = self._get_entity(entity_id)
        to_combine = _check_variable_list(to_combine, entity)

        df = self.get_dataframe(entity.id)

        new_data = None
        for v in to_combine:
            if new_data is None:
                new_data = df[v.id].map(lambda x: (str(x) if isinstance(x, (int, float)) else x).encode('utf-8'))
                continue
            new_data += "_".encode('utf-8')
            new_data += df[v.id].map(lambda x: (str(x) if isinstance(x, (int, float)) else x).encode('utf-8'))

        if hashed:
            new_data = new_data.map(hash)

        # first add to entityset
        self.add_column(entity.id, new_id, new_data, type=vtypes.Categorical)

        # TODO dfs: add column vs add variable?
        entity.add_variable(new_id, vtypes.Categorical)

        entity.add_variable_statistics(new_id)

        if drop:
            [self.delete_column(entity.id, v.id) for v in to_combine]
            [entity.delete_variable(v.id) for v in to_combine]

    def concat(self, other, inplace=False):
        '''Combine entityset with another to create a new entityset with the
        combined data of both entitysets.
        '''
        assert_string = "Entitysets must have the same entities, relationships"\
            ", and variable_ids"
        assert (self.__eq__(other) and
                self.relationships == other.relationships), assert_string

        for entity in self.entities:
            assert entity.id in other.entity_stores, assert_string
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
            combined_es[entity.id].update_data(combined_df)

        return combined_es

    # TODO DFS: is this used anywhere?
    # def filter_entityset_by_entity(self, entity_id):
    #     """ Filter out instances of all entities that aren't connected to
    #     entity_id """
    #     for e in self.entities:
    #         if e.id == entity_id:
    #             continue

    #         df = self._related_instances(entity_id, e.id, self)
    #         self.entity_stores[e.id].update_data(df)
    #         e.update_variable_statistics_all(self)

    ###########################################################################
    #  Indexing methods  ###############################################
    ###########################################################################

    def index_data(self, r):
        """
        If necessary, generate an index on the data which links instances of
        parent entities to collections of child instances which link to them.
        """
        parent_entity = self.entity_stores[r.parent_variable.entity.id]
        child_entity = self.entity_stores[r.child_variable.entity.id]
        child_entity.index_by_parent(parent_entity=parent_entity)

    def add_last_time_indexes(self):
        """
        Calculates the last time index values for each entity (the last time
        an instance or children of that instance were observed).  Used when
        calculating features using training windows
        """
        # Generate graph of entities to find leaf entities
        children = defaultdict(list)
        child_vars = defaultdict(dict)
        for r in self.relationships:
            children[r.parent_entity.id].append(r.child_entity)
            child_vars[r.parent_entity.id][r.child_entity.id] = r.child_variable

        explored = set([])
        queue = self.entities[:]

        for entity in self.entities:
            entity.set_last_time_index(None)

        while len(explored) < len(self.entities):
            entity = queue.pop(0)
            if entity.id not in children:
                if entity.has_time_index():
                    entity.set_last_time_index(entity.df[entity.time_index])
            else:
                child_entities = children[entity.id]
                if not set([e.id for e in child_entities]).issubset(explored):
                    queue.append(entity)
                    continue
                for child_e in child_entities:
                    link_var = child_vars[entity.id][child_e.id].id
                    if child_e.last_time_index is None:
                        continue
                    lti_df = pd.DataFrame({'last_time': child_e.last_time_index,
                                           entity.index: child_e.df[link_var]})
                    lti_df.drop_duplicates(entity.index,
                                           keep='last',
                                           inplace=True)
                    lti_df.set_index(entity.index, inplace=True)
                    if entity.last_time_index is None:
                        entity.last_time_index = lti_df['last_time']
                    else:
                        lti_df['last_time_old'] = entity.last_time_index
                        entity.last_time_index = lti_df.max(axis=1).dropna()
                        entity.last_time_index.name = 'last_time'
            explored.add(entity.id)

    ###########################################################################
    #  Other ###############################################
    ###########################################################################

    def add_interesting_values(self, max_values=5, verbose=False):
        """Find interesting values for categorical variables, to be used to generate "where" clauses

        Args:
            max_values (int) : maximum number of values per variable to add
            verbose (bool) : If True, print summary of interesting values found

        Returns:
            None

        """
        # _operations?
        for entity in self.entities:
            entity.add_interesting_values(max_values=max_values, verbose=verbose)

    ###########################################################################
    #  Private methods  ######################################################
    ###########################################################################

    # TODO: public?
    def _related_instances(self, start_entity_id, final_entity_id,
                           instance_ids=None, time_last=None, add_link=False,
                           training_window=None):
        """
        Filter out all but relevant information from dataframes along path
        from start_entity_id to final_entity_id,
        exclude data if it does not lie within  and time_last

        Args:
            start_entity_id (str) : id of start entity
            final_entity_id (str) : id of final entity
            instance_ids (list[str]) : list of start entity instance ids from
                which to find related instances in final entity
            time_last (pd.TimeStamp) :  latest allowed time
            add_link (bool) : if True, add a link variable from the first
                entity in the path to the last. Assumes the path is made up of
                only backwards relationships.

        Returns:
            pd.DataFrame : Dataframe of related instances on the final_entity_id
        """
        # Load the filtered dataframe for the first entity
        training_window_is_dict = isinstance(training_window, dict)
        window = training_window
        start_estore = self.entity_stores[start_entity_id]
        if instance_ids is None:
            df = start_estore.df
        else:   # instance_ids was passed in
            # This check might be brittle
            if not hasattr(instance_ids, '__iter__'):
                instance_ids = [instance_ids]

            if training_window_is_dict:
                window = training_window.get(start_estore.id)
            df = start_estore.query_by_values(instance_ids,
                                              time_last=time_last,
                                              training_window=window)

        # if we're querying on a path that's not actually a path, just return
        # the relevant slice of the entityset
        if start_entity_id == final_entity_id:
            return df

        # get relationship path from start to end entity
        path = self.find_path(start_entity_id, final_entity_id)
        if path is None or len(path) == 0:
            return pd.DataFrame()

        if add_link:
            assert 'forward' not in self.path_relationships(path,
                                                            start_entity_id)
            rvar = path[0].get_entity_variable(start_entity_id)
            link_map = {i: i for i in df[rvar]}

        prev_entity_id = start_entity_id

        # Walk down the path of entities and take related instances at each step
        for i, r in enumerate(path):
            new_entity_id = r.get_other_entity(prev_entity_id)
            rvar_old = r.get_entity_variable(prev_entity_id)
            rvar_new = r.get_entity_variable(new_entity_id)
            all_ids = df[rvar_old]

            # filter the next entity by the values found in the previous
            # entity's relationship column
            entity_store = self.entity_stores[new_entity_id]
            if training_window_is_dict:
                window = training_window.get(entity_store.id)
            df = entity_store.query_by_values(all_ids,
                                              variable_id=rvar_new,
                                              time_last=time_last,
                                              training_window=window)

            # group the rows in the new dataframe by the instances of the first
            # dataframe, and add a new column linking the two.
            if add_link:
                new_link_map = {}
                for parent_ix, gdf in df.groupby(rvar_new):
                    for child_ix in gdf.index:
                        new_link_map[child_ix] = link_map[parent_ix]
                link_map = new_link_map

                child_link_var = \
                    Relationship._get_link_variable_name(path[:i + 1])
                if child_link_var not in df.columns:
                    df = df.join(pd.Series(link_map, name=child_link_var))

            prev_entity_id = new_entity_id

        return df

    def _add_multigenerational_link_vars(self, frames, start_entity_id,
                                         end_entity_id=None, path=None):
        """
        Add multi-generational link variables to entity dataframes in order to
        keep track of deep relationships.

        For example: if entity 'grandparent' has_many of entity 'parent' which
        has_many of entity 'child', and parent is related to grandparent by
        variable 'grandparent_id', add a column to child called
        'parent.grandparent_id' so that child instances can be grouped by
        grandparent_id as well.

        This function adds link variables to all relationships along the
        provided path.
        """

        # caller can pass either a path or a start/end entity pair

        assert start_entity_id is not None
        if path is None:
            assert end_entity_id is not None
            path = self.find_path(start_entity_id, end_entity_id)

        directions = self.path_relationships(path, start_entity_id)
        relationship_directions = list(zip(directions, path))
        groups = itertools.groupby(relationship_directions, key=lambda k: k[0])

        # each group is a contiguous series of backward relationships on `path`
        for key, group in groups:
            if key != 'backward':
                continue

            # extract the path again
            chain = [g[1] for g in group]

            # generate a list of all sub-paths which have at least 2
            # relationships
            rel_chains = [chain[i:] for i in range(len(chain) - 1)]

            # loop over all subpaths
            for chain in rel_chains:
                # pop off the first relationship: this one already has a
                # direct variable link, but we'll need to remember its link
                # variable name for later.
                r = chain.pop(0)
                child_link_name = r.child_variable.id

                # step through each deep relationship of the subpath
                for r in chain:
                    parent_entity = r.parent_entity
                    child_entity = r.child_entity
                    parent_df = frames[parent_entity.id]
                    child_df = frames[child_entity.id]

                    # generate the link variable name
                    parent_link_name = child_link_name
                    child_link_name = '%s.%s' % (parent_entity.id,
                                                 parent_link_name)
                    if child_link_name in child_df.columns:
                        continue

                    # print 'adding link var %s to entity %s' % (child_link_name,
                    #                                            child_entity.id)

                    # create an intermeidate dataframe which shares a column
                    # with the child dataframe and has a column with the
                    # original parent's id.
                    col_map = {r.parent_variable.id: r.child_variable.id,
                               parent_link_name: child_link_name}
                    merge_df = parent_df[list(col_map.keys())].rename(columns=col_map)

                    # merge the dataframe, adding the link variable to the child
                    frames[child_entity.id] = pd.merge(left=merge_df,
                                                       right=child_df,
                                                       on=r.child_variable.id)

    def gen_relationship_var(self, child_eid, parent_eid):
        path = self.find_path(parent_eid, child_eid)
        r = path.pop(0)
        child_link_name = r.child_variable.id
        for r in path:
            parent_entity = r.parent_entity
            parent_link_name = child_link_name
            child_link_name = '%s.%s' % (parent_entity.id,
                                         parent_link_name)
        return child_link_name
