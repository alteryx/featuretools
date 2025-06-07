import copy
import logging
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from woodwork import init_series
from woodwork.logical_types import Datetime, LatLong

from featuretools.entityset import deserialize, serialize
from featuretools.entityset.relationship import Relationship, RelationshipPath
from featuretools.feature_base.feature_base import _ES_REF
from featuretools.utils.plot_utils import (
    check_graphviz,
    get_graphviz_format,
    save_graph,
)
from featuretools.utils.wrangle import _check_timedelta

pd.options.mode.chained_assignment = None  # default='warn'
logger = logging.getLogger("featuretools.entityset")

LTI_COLUMN_NAME = "_ft_last_time"
WW_SCHEMA_KEY = "_ww__getstate__schemas"


class EntitySet(object):
    """
    Stores all actual data and typing information for an entityset

    Attributes:
        id
        dataframe_dict
        relationships
        time_type

    Properties:
        metadata

    """

    def __init__(self, id=None, dataframes=None, relationships=None):
        """Creates EntitySet

        Args:
            id (str) : Unique identifier to associate with this instance
            dataframes (dict[str -> tuple(DataFrame, str, str, dict[str -> str/Woodwork.LogicalType], dict[str->str/set], boolean)]):
                Dictionary of DataFrames. Entries take the format
                {dataframe name -> (dataframe, index column, time_index, logical_types, semantic_tags, make_index)}.
                Note that only the dataframe is required. If a Woodwork DataFrame is supplied, any other parameters
                will be ignored.
            relationships (list[(str, str, str, str)]): List of relationships
                between dataframes. List items are a tuple with the format
                (parent dataframe name, parent column, child dataframe name, child column).

        Example:

            .. code-block:: python

                dataframes = {
                    "cards" : (card_df, "id"),
                    "transactions" : (transactions_df, "id", "transaction_time")
                }

                relationships = [("cards", "id", "transactions", "card_id")]

                ft.EntitySet("my-entity-set", dataframes, relationships)
        """
        self.id = id
        self.dataframe_dict = {}
        self.relationships = []
        self.time_type = None

        dataframes = dataframes or {}
        relationships = relationships or []
        for df_name in dataframes:
            df = dataframes[df_name][0]
            if df.ww.schema is not None and df.ww.name != df_name:
                raise ValueError(
                    f"Naming conflict in dataframes dictionary: dictionary key '{df_name}' does not match dataframe name '{df.ww.name}'",
                )

            index_column = None
            time_index = None
            make_index = False
            semantic_tags = None
            logical_types = None
            if len(dataframes[df_name]) > 1:
                index_column = dataframes[df_name][1]
            if len(dataframes[df_name]) > 2:
                time_index = dataframes[df_name][2]
            if len(dataframes[df_name]) > 3:
                logical_types = dataframes[df_name][3]
            if len(dataframes[df_name]) > 4:
                semantic_tags = dataframes[df_name][4]
            if len(dataframes[df_name]) > 5:
                make_index = dataframes[df_name][5]
            self.add_dataframe(
                dataframe_name=df_name,
                dataframe=df,
                index=index_column,
                time_index=time_index,
                logical_types=logical_types,
                semantic_tags=semantic_tags,
                make_index=make_index,
            )

        for relationship in relationships:
            parent_df, parent_column, child_df, child_column = relationship
            self.add_relationship(parent_df, parent_column, child_df, child_column)

        self.reset_data_description()
        _ES_REF[self.id] = self

    def __sizeof__(self):
        return sum([df.__sizeof__() for df in self.dataframes])

    def __dask_tokenize__(self):
        return (EntitySet, serialize.entityset_to_description(self.metadata))

    def __eq__(self, other, deep=False):
        if self.id != other.id:
            return False
        if self.time_type != other.time_type:
            return False
        if len(self.dataframe_dict) != len(other.dataframe_dict):
            return False
        for df_name, df in self.dataframe_dict.items():
            if df_name not in other.dataframe_dict:
                return False
            if not df.ww.__eq__(other[df_name].ww, deep=deep):
                return False
        if not len(self.relationships) == len(other.relationships):
            return False
        for r in self.relationships:
            if r not in other.relationships:
                return False
        return True

    def __ne__(self, other, deep=False):
        return not self.__eq__(other, deep=deep)

    def __getitem__(self, dataframe_name):
        """Get dataframe instance from entityset

        Args:
            dataframe_name (str): Name of dataframe.

        Returns:
            :class:`.DataFrame` : Instance of dataframe with Woodwork typing information. None if dataframe doesn't
                exist on the entityset.
        """
        if dataframe_name in self.dataframe_dict:
            return self.dataframe_dict[dataframe_name]
        name = self.id or "entity set"
        raise KeyError("DataFrame %s does not exist in %s" % (dataframe_name, name))

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "dataframe_dict":
                # Copy the DataFrames, retaining Woodwork typing information
                copied_attr = copy.copy(v)
                for df_name, df in copied_attr.items():
                    copied_attr[df_name] = df.ww.copy()
            else:
                copied_attr = copy.deepcopy(v, memo)

            setattr(result, k, copied_attr)

        for df in result.dataframe_dict.values():
            result._add_references_to_metadata(df)
        return result

    @property
    def dataframes(self):
        return list(self.dataframe_dict.values())

    @property
    def metadata(self):
        """Returns the metadata for this EntitySet. The metadata will be recomputed if it does not exist."""
        if self._data_description is None:
            description = serialize.entityset_to_description(self)
            self._data_description = deserialize.description_to_entityset(description)

        return self._data_description

    def reset_data_description(self):
        self._data_description = None

    def to_pickle(self, path, compression=None, profile_name=None):
        """Write entityset in the pickle format, location specified by `path`.
        Path could be a local path or a S3 path.
        If writing to S3 a tar archive of files will be written.

        Args:
            path (str): location on disk to write to (will be created as a directory)
            compression (str) : Name of the compression to use. Possible values are: {'gzip', 'bz2', 'zip', 'xz', None}.
            profile_name (str) : Name of AWS profile to use, False to use an anonymous profile, or None.
        """
        serialize.write_data_description(
            self,
            path,
            format="pickle",
            compression=compression,
            profile_name=profile_name,
        )
        return self

    def to_parquet(self, path, engine="auto", compression=None, profile_name=None):
        """Write entityset to disk in the parquet format, location specified by `path`.
        Path could be a local path or a S3 path.
        If writing to S3 a tar archive of files will be written.

        Args:
            path (str): location on disk to write to (will be created as a directory)
            engine (str) : Name of the engine to use. Possible values are: {'auto', 'pyarrow'}.
            compression (str) : Name of the compression to use. Possible values are: {'snappy', 'gzip', 'brotli', None}.
            profile_name (str) : Name of AWS profile to use, False to use an anonymous profile, or None.
        """
        serialize.write_data_description(
            self,
            path,
            format="parquet",
            engine=engine,
            compression=compression,
            profile_name=profile_name,
        )
        return self

    def to_csv(
        self,
        path,
        sep=",",
        encoding="utf-8",
        engine="python",
        compression=None,
        profile_name=None,
    ):
        """Write entityset to disk in the csv format, location specified by `path`.
        Path could be a local path or a S3 path.
        If writing to S3 a tar archive of files will be written.

        Args:
            path (str) : Location on disk to write to (will be created as a directory)
            sep (str) : String of length 1. Field delimiter for the output file.
            encoding (str) : A string representing the encoding to use in the output file, defaults to 'utf-8'.
            engine (str) : Name of the engine to use. Possible values are: {'python', 'c', 'pyarrow'}.
            compression (str) : Name of the compression to use. Possible values are: {'gzip', 'bz2', 'zip', 'xz', None}.
            profile_name (str) : Name of AWS profile to use, False to use an anonymous profile, or None.
        """
        serialize.write_data_description(
            self,
            path,
            format="csv",
            sep=sep,
            encoding=encoding,
            engine=engine,
            compression=compression,
            profile_name=profile_name,
        )
        return self

    def to_dictionary(self):
        return serialize.entityset_to_description(self)

    def __repr__(self):
        """Print string representation of EntitySet"""
        return serialize.entityset_to_json(self.metadata)

    def add_relationships(self, relationships):
        for rel in relationships:
            self.add_relationship(relationship=rel)

        return self

    def add_relationship(
        self,
        parent_dataframe_name=None,
        parent_column_name=None,
        child_dataframe_name=None,
        child_column_name=None,
        relationship=None,
    ):
        if relationship is None:
            relationship = Relationship(
                self,
                parent_dataframe_name,
                parent_column_name,
                child_dataframe_name,
                child_column_name,
            )

        if relationship not in self.relationships:
            self.relationships.append(relationship)

        self.reset_data_description()
        return self

    def set_secondary_time_index(self, dataframe_name, secondary_time_index):
        self._set_secondary_time_index(self[dataframe_name], secondary_time_index)
        self.reset_data_description()
        return self

    def _set_secondary_time_index(self, dataframe, secondary_time_index):
        if secondary_time_index:
            dataframe.ww.set_secondary_time_index(secondary_time_index)
        self._add_references_to_metadata(dataframe)

    def find_forward_paths(self, start_dataframe_name, goal_dataframe_name):
        # check all relationships in dataframe for a path to other dataframe
        return self.metadata.find_forward_paths(
            start_dataframe_name,
            goal_dataframe_name,
        )

    def find_backward_paths(self, start_dataframe_name, goal_dataframe_name):
        return self.metadata.find_backward_paths(
            start_dataframe_name,
            goal_dataframe_name,
        )

    def _forward_dataframe_paths(self, start_dataframe_name, seen_dataframes=None):
        """
        Helper function which returns a list of dataframes which can be accessed from
        the start dataframe by forward relationships
        """
        if seen_dataframes is None:
            seen_dataframes = set()
        if start_dataframe_name in seen_dataframes:
            return []
        seen_dataframes.add(start_dataframe_name)

        all_dataframes = []
        for relationship in self.get_forward_relationships(start_dataframe_name):
            new_dataframe = relationship.child_dataframe.ww.name
            all_dataframes.append(new_dataframe)
            all_dataframes.extend(
                self._forward_dataframe_paths(new_dataframe, seen_dataframes),
            )
        return all_dataframes

    def get_forward_dataframes(self, dataframe_name, deep=False):
        if deep:
            return self._forward_dataframe_paths(dataframe_name)
        return [r.child_dataframe.ww.name for r in self.get_forward_relationships(dataframe_name)]

    def get_backward_dataframes(self, dataframe_name, deep=False):
        if deep:
            return self._backward_dataframe_paths(dataframe_name)
        return [
            r.parent_dataframe.ww.name for r in self.get_backward_relationships(dataframe_name)
        ]

    def _backward_dataframe_paths(self, start_dataframe_name, seen_dataframes=None):
        """
        Helper function which returns a list of dataframes which can be accessed from
        the start dataframe by backward relationships
        """
        if seen_dataframes is None:
            seen_dataframes = set()
        if start_dataframe_name in seen_dataframes:
            return []
        seen_dataframes.add(start_dataframe_name)

        all_dataframes = []
        for relationship in self.get_backward_relationships(start_dataframe_name):
            new_dataframe = relationship.parent_dataframe.ww.name
            all_dataframes.append(new_dataframe)
            all_dataframes.extend(
                self._backward_dataframe_paths(new_dataframe, seen_dataframes),
            )
        return all_dataframes

    def get_forward_relationships(self, dataframe_name):
        return [r for r in self.relationships if r.parent_dataframe.ww.name == dataframe_name]

    def get_backward_relationships(self, dataframe_name):
        return [r for r in self.relationships if r.child_dataframe.ww.name == dataframe_name]

    def has_unique_forward_path(self, start_dataframe_name, end_dataframe_name):
        # check if there is one unique forward path between start and end dataframes
        return self.metadata.has_unique_forward_path(
            start_dataframe_name,
            end_dataframe_name,
        )

    def add_dataframe(
        self,
        dataframe,
        dataframe_name=None,
        index=None,
        logical_types=None,
        semantic_tags=None,
        make_index=False,
        time_index=None,
        secondary_time_index=None,
        already_sorted=False,
    ):
        """Adds a dataframe to the EntitySet.

        Args:
            dataframe (DataFrame): DataFrame to add.
            dataframe_name (str): Name of the dataframe to add. If Woodwork DataFrame is supplied, this parameter
                will be ignored.
            index (str): Column in the DataFrame to use as the index.
                If no index is supplied, one is created automatically as the
                dataframe's Woodwork index. If Woodwork DataFrame is supplied, this parameter
                will be ignored.
            logical_types (dict[str -> str/Woodwork.LogicalType], optional):
                Dictionary mapping column names to the LogicalType that their
                data should be interpreted as. If Woodwork DataFrame is supplied, this parameter
                will be ignored.
            semantic_tags (dict[str -> str/set], optional):
                Dictionary mapping column names to the semantic tags that their
                data should be tagged with. If Woodwork DataFrame is supplied, this parameter
                will be ignored.
            make_index (bool): Whether to add a new column with unique integer values
                to be used as the index. If Woodwork DataFrame is supplied, this parameter
                will be ignored.
            time_index (str, optional): Column in the DataFrame to use as the time
                index. If Woodwork DataFrame is supplied, this parameter will be ignored.
            secondary_time_index (dict[str->[str]], optional): Dictionary of secondary time index
                columns, mapping a column name to a list of columns that must be empty
                for that time index to be valid. If Woodwork DataFrame is supplied, this parameter
                will be ignored.
            already_sorted (bool): If true, dataframe is assumed to be sorted by time_index
                and no additional sorting will be performed.
                If Woodwork DataFrame is supplied, this parameter will be ignored.

        Returns:
            :class:`.EntitySet` : Instance of the calling EntitySet

        Example:
            .. code-block:: python

                es = ft.EntitySet(id="my_entity_set")
                es.add_dataframe(dataframe_name="transactions",
                                 dataframe=transactions_df,
                                 index="transaction_id",
                                 time_index="transaction_time")

                es["transactions"].ww
        """
        # If Woodwork init was done outside of add_dataframe, ensure that no parameters conflict with Woodwork
        # metadata
        if dataframe.ww.schema is not None:
            if dataframe_name is not None and dataframe_name != dataframe.ww.name:
                raise ValueError(
                    f"Naming conflict: dataframe_name '{dataframe_name}' does not match Woodwork name '{dataframe.ww.name}'",
                )
            if index is not None and index != dataframe.ww.index:
                raise ValueError(
                    f"Index conflict: index '{index}' does not match Woodwork index '{dataframe.ww.index}'",
                )
            if time_index is not None and time_index != dataframe.ww.time_index:
                raise ValueError(
                    f"Time index conflict: time_index '{time_index}' does not match Woodwork time index '{dataframe.ww.time_index}'",
                )

        self.dataframe_dict[dataframe.ww.name] = dataframe

        self._add_references_to_metadata(dataframe)
        self.reset_data_description()
        return self

    def __setitem__(self, key, value):
        self.add_dataframe(key, value)

    def normalize_dataframe(
        self,
        base_dataframe_name,
        new_dataframe_name,
        index,
        additional_columns=None,
        copy_columns=None,
        make_time_index=None,
        make_secondary_time_index=None,
        new_dataframe_time_index=None,
        new_dataframe_secondary_time_index=None,
    ):
        """Normalizes an existing dataframe by creating a new dataframe and
        a relationship between them.

        Args:
            base_dataframe_name (str): The name of the dataframe to normalize.
            new_dataframe_name (str): The name of the new dataframe.
            index (str): The column in the base dataframe that will become the
                index of the new dataframe.
            additional_columns (list[str], optional): List of columns to move
                from the base dataframe to the new dataframe. Can be a string
                or a list of strings.
            copy_columns (list[str], optional): List of columns to copy from the base
                dataframe to the new dataframe. Can be a string or a list of
                strings.
            make_time_index (bool): Whether to use the time_index of the base_dataframe as the
                time_index of the new_dataframe. If no time_index is set on the base_dataframe,
                this parameter is ignored. Defaults to true.
            make_secondary_time_index (dict[str->[str]], optional): List of secondary time index
                columns to move from the base dataframe to the new dataframe and use as secondary
                time indexes in the new dataframe.
            new_dataframe_time_index (str, optional): The column in the new dataframe to be used as its time_index.
                If this is set, make_time_index is ignored.
            new_dataframe_secondary_time_index (dict[str->[str]], optional):
                Dictionary of secondary time index columns to use as secondary time indexes
                in the new dataframe. If this is set, make_secondary_time_index is ignored.

        Returns:
            :class:`.EntitySet` : Instance of the calling EntitySet

        Example:
            .. code-block:: python

                es = ft.EntitySet(id="my_entity_set")
                es.add_dataframe(dataframe_name="transactions",
                                 dataframe=transactions_df,
                                 index="transaction_id",
                                 time_index="transaction_time")
                es.normalize_dataframe(new_dataframe_name="products",
                                      base_dataframe_name="transactions",
                                      index="product_id",
                                      additional_columns=["price", "rating"])
        """

        # TODO: This method has a lot of side effects, need to reduce them
        base_df = self[base_dataframe_name]

        if additional_columns is None:
            additional_columns = []
        if not isinstance(additional_columns, list):
            additional_columns = [additional_columns]

        if copy_columns is None:
            copy_columns = []
        if not isinstance(copy_columns, list):
            copy_columns = [copy_columns]

        # 1. Create new dataframe
        # make sure new_df does not include the base_dataframe's index
        # if it is one of the additional_columns
        new_df = base_df.ww.pop(index)
        for col in additional_columns:
            new_df[col] = base_df.ww.pop(col)
        new_df = new_df.ww.copy_dataframe(include_index=False, include_time_index=False)
        new_df.index = new_df.index.astype(base_df.ww.index_type)
        new_df.index.name = index
        new_df = new_df.ww.init(logical_types={index: "index"})

        time_index = None
        if new_dataframe_time_index:
            time_index = new_dataframe_time_index
        elif make_time_index is None or make_time_index:
            time_index = base_df.ww.time_index

        # handle secondary time indexes
        secondary_time_index = None
        if new_dataframe_secondary_time_index:
            secondary_time_index = new_dataframe_secondary_time_index
        elif make_secondary_time_index:
            secondary_time_index = {}
            for col in make_secondary_time_index:
                base_df.ww.pop(col)
                secondary_time_index[col] = base_df.ww.schema.secondary_time_indexes.get(
                    col,
                    None,
                )

        # Add new dataframe to entityset
        self.add_dataframe(
            dataframe=new_df,
            dataframe_name=new_dataframe_name,
            index=index,
            time_index=time_index,
            secondary_time_index=secondary_time_index,
        )

        # 2. Update existing dataframe
        # remove index column from base_dataframe, if it wasn't removed already
        if index != base_df.ww.index and index in base_df.columns:
            base_df.ww.pop(index)
        # add relationship column if it was not already present in the original dataframe
        if index not in base_df.columns:
            new_col = base_df.ww.add.relationship(index, new_dataframe_name)
            new_col.ww.add.semantic_tags("foreign_key")
            base_df[index] = new_col
            # Move the old column from base dataframe to new dataframe
            # to allow Woodwork to use its own initialization to infer logical types
            # of the new dataframe.
            base_df.ww.set_logical_type(index, "id")
        for col in copy_columns:
            base_df.ww.add.relationship(col, new_dataframe_name)
            base_df.ww.set_logical_type(col, "id")

        # 3. Create relationship
        self.add_relationship(new_dataframe_name, index, base_dataframe_name, index)

        self.reset_data_description()

        return self

    def concat(self, other, inplace=False):
        """
        Combines two entitysets together by concatenating all dataframes with the same name.

        Args:
            other (EntitySet): EntitySet to concat with.
            inplace (bool): If True, update the calling EntitySet in place. Otherwise, return a new EntitySet.

        Returns:
            :class:`.EntitySet` : A new EntitySet with concatenated dataframes.
        """
        if not isinstance(other, type(self)):
            raise TypeError("Cannot concat %s with %s" % (type(self), type(other)))

        if self.id == other.id:
            raise ValueError("Cannot concat an EntitySet with itself")

        if inplace:
            new_es = self
        else:
            new_es = self.copy()

        for df_name in new_es.dataframe_dict.keys():
            if df_name in other.dataframe_dict:
                df = new_es[df_name]
                other_df = other[df_name]

                if not df.ww.schema.is_compatible(other_df.ww.schema, close_match=True):
                    raise ValueError(
                        "Schemas for dataframe '%s' are not compatible. Incompatible schemas: %s and %s" % (df_name, df.ww.schema, other_df.ww.schema)
                    )

                df = pd.concat([df, other_df], ignore_index=True, sort=True)
                df.ww.init(
                    schema=new_es[df_name].ww.schema,
                    name=df_name,
                    index=new_es[df_name].ww.index,
                    time_index=new_es[df_name].ww.time_index,
                    log_schema=new_es[df_name].ww.schema.log_schema,
                )
                new_es[df_name] = df

        return new_es

    def add_last_time_indexes(self, updated_dataframes=None):
        """Adds a last time index to any dataframe that has a time index.

        Args:
            updated_dataframes (list[str], optional): List of dataframes names to update.
                If None, all dataframes with a time index are updated.
        """
        dataframes_to_update = updated_dataframes
        if updated_dataframes is None:
            dataframes_to_update = list(self.dataframe_dict.keys())

        for df_name in dataframes_to_update:
            dataframe = self[df_name]

            if dataframe.ww.time_index is None:
                continue

            # use a list of columns including the index and all secondary time indexes
            groupby_cols = [dataframe.ww.index]
            if dataframe.ww.secondary_time_indexes:
                for secondary_index_col in dataframe.ww.secondary_time_indexes:
                    groupby_cols.append(secondary_index_col)

            # if a dataframe has a natural time index, we can just use the last time
            # an instance appears in the data as its last_time_index
            last_time_index = (
                dataframe.groupby(groupby_cols)[dataframe.ww.time_index]
                .apply(lambda x: x.sort_values().iloc[-1])
                .reset_index()
            )
            last_time_index.rename(
                columns={dataframe.ww.time_index: LTI_COLUMN_NAME},
                inplace=True,
            )
            # Merge the new last_time_index dataframe into the Woodwork metadata
            dataframe.ww.metadata["last_time_index"] = last_time_index

        self.reset_data_description()

    def __getstate__(self):
        # Clear _data_description before pickling, to avoid recursion depth issues.
        # Metadata is computed when accessed via property, so it will be recreated on load.
        state = self.__dict__.copy()
        state["_data_description"] = None
        if WW_SCHEMA_KEY not in state:
            schemas = {name: df.ww.schema for name, df in state["dataframe_dict"].items()}
            state[WW_SCHEMA_KEY] = schemas
        return state

    def __setstate__(self, state):
        # For older entitysets, manually deserialize Woodwork schemas.
        if WW_SCHEMA_KEY in state:
            schemas = state.pop(WW_SCHEMA_KEY)
            for name, df in state["dataframe_dict"].items():
                df.ww.init(schema=schemas[name], name=name)

        self.__dict__.update(state)
        _ES_REF[self.id] = self

    def add_interesting_values(
        self,
        max_values=5,
        verbose=False,
        dataframe_name=None,
        values=None,
    ):
        """Finds and adds interesting values to Woodwork for categorical features.

        Args:
            max_values (int, optional): The maximum number of unique values to include.
                If there are more unique values than `max_values`, the most frequent
                values will be included.
            verbose (bool, optional): Whether to print out the interesting values.
            dataframe_name (str, optional): Name of the dataframe to add interesting values to.
                If None, interesting values are added to all dataframes.
            values (list, optional): A list of interesting values to add. If this is
                set, `max_values` is ignored.
        """

        def add_value(df, col, val, verbose):
            df.ww[col].ww.add_interesting_values(val, verbose=verbose)

        if dataframe_name is None:
            for df in self.dataframes:
                for col in df.columns:
                    if df.ww[col].ww.logical_type.type_string == "categorical":
                        add_value(df, col, values, verbose)
        else:
            df = self[dataframe_name]
            for col in df.columns:
                if df.ww[col].ww.logical_type.type_string == "categorical":
                    add_value(df, col, values, verbose)
        self.reset_data_description()

    def plot(self, to_file=None):
        """Plots the EntitySet and returns a graphviz Digraph object.

        Args:
            to_file (str, optional): Path to save the plot. If this is None,
                the plot will not be saved. If the path does not include
                an extension, it will default to .png. If the path is a directory,
                a filename will be generated automatically. If `graphviz` is not
                installed, the plot will not be rendered.

        Returns:
            graphviz.Digraph : Graphviz Digraph object of the EntitySet.
        """
        check_graphviz()

        return self.metadata.plot(to_file=to_file)

    def _handle_time(
        self,
        dataframe_name,
        df,
        time_last=None,
        training_window=None,
        include_cutoff_time=True,
    ):
        """
        Filter a dataframe for all instances before time_last.
        If the dataframe does not have a time index, return the original
        dataframe.
        Modified: Retain rows with missing time indices (NaN/NaT) so that non-time-based features can still be computed.
        """

        schema = self[dataframe_name].ww.schema
        if schema.time_index:
            df_empty = df.empty
            if time_last is not None and not df_empty:
                # Identify rows with missing time index
                missing_time_mask = df[schema.time_index].isna()
                if missing_time_mask.any():
                    warnings.warn(
                        f"DataFrame '{dataframe_name}' contains rows with missing time indices. Time-based features will be NaN for these rows.",
                    )
                # Filter by time_last for non-missing time indices
                filtered_df = df[~missing_time_mask].copy()
                if include_cutoff_time:
                    filtered_df = filtered_df[filtered_df[schema.time_index] <= time_last]
                else:
                    filtered_df = filtered_df[filtered_df[schema.time_index] < time_last]

                # Recombine filtered rows with rows that had missing time indices
                # This ensures rows with NaT are retained but won't affect time-based calculations
                df = pd.concat([filtered_df, df[missing_time_mask]], axis=0, ignore_index=True)
                # Handle training window for non-missing time indices
                if training_window is not None:
                    # For now, training_window only supports datetime time_index
                    if not isinstance(schema.logical_types[schema.time_index], Datetime):
                        raise TypeError(
                            "training_window is only supported with datetime time_index",
                        )
                    time_first = time_last - _check_timedelta(training_window)
                    df = df[(df[schema.time_index] >= time_first) | (missing_time_mask)]

            # If there's no time_last, but a training window is specified, filter by training window
            elif training_window is not None and not df_empty:
                if not isinstance(schema.logical_types[schema.time_index], Datetime):
                    raise TypeError(
                        "training_window is only supported with datetime time_index",
                    )
                time_first = time_last - _check_timedelta(training_window)
                df = df[df[schema.time_index] >= time_first]
        return df

    def query_by_values(
        self,
        dataframe_name,
        instance_vals,
        column_name=None,
        columns=None,
        time_last=None,
        training_window=None,
        include_cutoff_time=True,
    ):
        """
        Get the dataframe for a given dataframe and filter by instance values
        and time.

        Args:
            dataframe_name (str): Name of dataframe to query.

            instance_vals (np.ndarray or pd.Series): Instances to filter for.

            column_name (str, optional): Name of column to filter on.
                If None, the dataframe's index is used.

            columns (list[str], optional): List of column names to select.
                If None, all columns are selected.

            time_last (pd.Timestamp, optional): Last allowed time. Data from exactly this
                time not allowed.

            training_window (Timedelta, optional): Window defining how much time before the cutoff time data
                can be used when calculating features. If None, all data before cutoff time is used.

            include_cutoff_time (bool): If True, data at cutoff time are included
                in calculating features.

        Returns:
            pd.DataFrame: dataframe for this dataframe, filtered by the given
                instance ids and time.

        """
        if columns is None:
            columns = self[dataframe_name].ww.columns.keys()

        if column_name is None:
            column_name = self[dataframe_name].ww.index

        if not isinstance(instance_vals, pd.Series):
            instance_vals = _vals_to_series(instance_vals, column_name)

        # Filter by the values.
        df = self[dataframe_name].ww.loc[instance_vals.index][columns]

        # Filter by time if a time_last was given.
        df = self._handle_time(
            dataframe_name,
            df,
            time_last,
            training_window,
            include_cutoff_time=include_cutoff_time,
        )

        return df

    def replace_dataframe(
        self,
        dataframe_name,
        df,
        already_sorted=False,
        recalculate_last_time_indexes=True,
    ):
        """
        Replace a dataframe in the EntitySet.

        Args:
            dataframe_name (str): The name of the dataframe to replace.
            df (pd.DataFrame): The new dataframe to replace the old one.
            already_sorted (bool): If true, dataframe is assumed to be sorted by time_index
                and no additional sorting will be performed.
            recalculate_last_time_indexes (bool): If True, the last_time_indexes for all
                dataframes with a time_index will be re-calculated. Defaults to True.
        """
        if dataframe_name not in self.dataframe_dict:
            raise KeyError(f"DataFrame '{dataframe_name}' not found in entityset.")

        new_dataframe = df.ww.init(
            schema=self.dataframe_dict[dataframe_name].ww.schema,
            name=dataframe_name,
            index=self.dataframe_dict[dataframe_name].ww.index,
            time_index=self.dataframe_dict[dataframe_name].ww.time_index,
            log_schema=self.dataframe_dict[dataframe_name].ww.schema.log_schema,
            already_sorted=already_sorted,
        )

        if not self.dataframe_dict[dataframe_name].ww.schema.is_compatible(
            new_dataframe.ww.schema,
            close_match=True,
        ):
            raise ValueError(
                "New dataframe is not compatible with old dataframe. Incompatible schemas: %s and %s" % (self.dataframe_dict[dataframe_name].ww.schema, new_dataframe.ww.schema)
            )

        self.dataframe_dict[dataframe_name] = new_dataframe

        if recalculate_last_time_indexes:
            self.add_last_time_indexes([dataframe_name])

        self.reset_data_description()

    def _check_time_indexes(self):
        """
        Check that all time_indexes are datetime or Datetime logical types.
        """
        for dataframe in self.dataframes:
            if dataframe.ww.time_index:
                self._check_uniform_time_index(dataframe)

    def _check_secondary_time_index(self, dataframe, secondary_time_index=None):
        # For older entitysets, sometimes the secondary_time_indexes are not an array of dictionaries
        # in the schema, so this check will make them into an array of dictionaries if that is the case
        dataframe.ww.set_secondary_time_index(secondary_time_index)
        for s in dataframe.ww.secondary_time_indexes:
            self._check_uniform_time_index(dataframe, s)

    def _check_uniform_time_index(self, dataframe, column_name=None):
        dataframe.ww._check_uniform_time_index(column_name)

    def _get_time_type(self, dataframe, column_name=None):
        return dataframe.ww._get_time_type(column_name)

    def _add_references_to_metadata(self, dataframe):
        dataframe.ww.metadata["entityset"] = self

    def _normalize_values(self, dataframe):
        # normalize any values Woodwork added that were outside the schema
        def replace(x):
            try:
                x.ww.init(schema=dataframe.ww.schema[x.name], logical_types=dataframe.ww.logical_types[x.name])
            except TypeError:
                pass
            return x

        return dataframe.apply(replace)


def _vals_to_series(instance_vals, column_id):
    if isinstance(instance_vals, list):
        instance_vals = pd.Series(instance_vals, name=column_id)
    return instance_vals


def _get_or_create_index(index, make_index, df):
    if make_index:
        # TODO: Handle case where index exists as a regular column.
        # If Woodwork is already initialized on df, then index will be the existing
        # Woodwork index. If it is also passed as a column and not the index
        # then the index will be duplicated.
        if df.ww.index is None:
            df.ww.init(index=index, make_index=make_index)
        return df.ww.index

    if index not in df.columns:
        raise LookupError("Specified index column '%s' not found in dataframe." % index)
    return df.ww.index


def _create_index(df, index):
    if index is None:
        index = df.name
        df = df.reset_index()
        df.rename(columns={'index': index}, inplace=True)
    return index
