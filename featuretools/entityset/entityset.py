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
from featuretools.utils.gen_utils import Library, import_or_none, is_instance
from featuretools.utils.plot_utils import (
    check_graphviz,
    get_graphviz_format,
    save_graph,
)
from featuretools.utils.wrangle import _check_timedelta

dd = import_or_none("dask.dataframe")
ps = import_or_none("pyspark.pandas")

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
    def dataframe_type(self):
        """String specifying the library used for the dataframes. Null if no dataframes"""
        df_type = None

        if self.dataframes:
            if isinstance(self.dataframes[0], pd.DataFrame):
                df_type = Library.PANDAS
            elif is_instance(self.dataframes[0], dd, "DataFrame"):
                df_type = Library.DASK
            elif is_instance(self.dataframes[0], ps, "DataFrame"):
                df_type = Library.SPARK

        return df_type

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
            engine (str) : Name of the engine to use. Possible values are: {'c', 'python'}.
            compression (str) : Name of the compression to use. Possible values are: {'gzip', 'bz2', 'zip', 'xz', None}.
            profile_name (str) : Name of AWS profile to use, False to use an anonymous profile, or None.
        """
        if self.dataframe_type == Library.SPARK:
            compression = str(compression)
        serialize.write_data_description(
            self,
            path,
            format="csv",
            index=False,
            sep=sep,
            encoding=encoding,
            engine=engine,
            compression=compression,
            profile_name=profile_name,
        )
        return self

    def to_dictionary(self):
        return serialize.entityset_to_description(self)

    ###########################################################################
    #   Public getter/setter methods  #########################################
    ###########################################################################

    def __repr__(self):
        repr_out = "Entityset: {}\n".format(self.id)
        repr_out += "  DataFrames:"
        for df in self.dataframes:
            if df.shape:
                repr_out += "\n    {} [Rows: {}, Columns: {}]".format(
                    df.ww.name,
                    df.shape[0],
                    df.shape[1],
                )
            else:
                repr_out += "\n    {} [Rows: None, Columns: None]".format(df.ww.name)
        repr_out += "\n  Relationships:"

        if len(self.relationships) == 0:
            repr_out += "\n    No relationships"

        for r in self.relationships:
            repr_out += "\n    %s.%s -> %s.%s" % (
                r._child_dataframe_name,
                r._child_column_name,
                r._parent_dataframe_name,
                r._parent_column_name,
            )

        return repr_out

    def add_relationships(self, relationships):
        """Add multiple new relationships to a entityset

        Args:
            relationships (list[tuple(str, str, str, str)] or list[Relationship]) : List of
                new relationships to add. Relationships are specified either as a :class:`.Relationship`
                object or a four element tuple identifying the parent and child columns:
                (parent_dataframe_name, parent_column_name, child_dataframe_name, child_column_name)
        """
        for rel in relationships:
            if isinstance(rel, Relationship):
                self.add_relationship(relationship=rel)
            else:
                self.add_relationship(*rel)
        return self

    def add_relationship(
        self,
        parent_dataframe_name=None,
        parent_column_name=None,
        child_dataframe_name=None,
        child_column_name=None,
        relationship=None,
    ):
        """Add a new relationship between dataframes in the entityset. Relationships can be specified
        by passing dataframe and columns names or by passing a :class:`.Relationship` object.

        Args:
            parent_dataframe_name (str): Name of the parent dataframe in the EntitySet. Must be specified
                if relationship is not.
            parent_column_name (str): Name of the parent column. Must be specified if relationship is not.
            child_dataframe_name (str): Name of the child dataframe in the EntitySet. Must be specified
                if relationship is not.
            child_column_name (str): Name of the child column. Must be specified if relationship is not.
            relationship (Relationship): Instance of new relationship to be added. Must be specified
                if dataframe and column names are not supplied.
        """
        if relationship and (
            parent_dataframe_name
            or parent_column_name
            or child_dataframe_name
            or child_column_name
        ):
            raise ValueError(
                "Cannot specify dataframe and column name values and also supply a Relationship",
            )

        if not relationship:
            relationship = Relationship(
                self,
                parent_dataframe_name,
                parent_column_name,
                child_dataframe_name,
                child_column_name,
            )
        if relationship in self.relationships:
            warnings.warn("Not adding duplicate relationship: " + str(relationship))
            return self

        # _operations?

        # this is a new pair of dataframes
        child_df = relationship.child_dataframe
        child_column = relationship._child_column_name
        if child_df.ww.index == child_column:
            msg = "Unable to add relationship because child column '{}' in '{}' is also its index"
            raise ValueError(msg.format(child_column, child_df.ww.name))
        parent_df = relationship.parent_dataframe
        parent_column = relationship._parent_column_name

        if parent_df.ww.index != parent_column:
            parent_df.ww.set_index(parent_column)

        # Empty dataframes (as a result of accessing metadata)
        # default to object dtypes for categorical columns, but
        # indexes/foreign keys default to ints. In this case, we convert
        # the empty column's type to int
        if isinstance(child_df, pd.DataFrame) and (
            child_df.empty
            and child_df[child_column].dtype == object
            and parent_df.ww.columns[parent_column].is_numeric
        ):
            child_df.ww[child_column] = pd.Series(name=child_column, dtype=np.int64)

        parent_ltype = parent_df.ww.logical_types[parent_column]
        child_ltype = child_df.ww.logical_types[child_column]
        if parent_ltype != child_ltype:
            difference_msg = ""
            if str(parent_ltype) == str(child_ltype):
                difference_msg = "There is a conflict between the parameters. "

            warnings.warn(
                f"Logical type {child_ltype} for child column {child_column} does not match "
                f"parent column {parent_column} logical type {parent_ltype}. {difference_msg}"
                "Changing child logical type to match parent.",
            )
            child_df.ww.set_types(logical_types={child_column: parent_ltype})

        if "foreign_key" not in child_df.ww.semantic_tags[child_column]:
            child_df.ww.add_semantic_tags({child_column: "foreign_key"})

        self.relationships.append(relationship)
        self.reset_data_description()
        return self

    def set_secondary_time_index(self, dataframe_name, secondary_time_index):
        """
        Set the secondary time index for a dataframe in the EntitySet using its dataframe name.

        Args:
            dataframe_name (str) : name of the dataframe for which to set the secondary time index.
            secondary_time_index (dict[str-> list[str]]): Name of column containing time data to
                be used as a secondary time index mapped to a list of the columns in the dataframe
                associated with that secondary time index.
        """
        dataframe = self[dataframe_name]
        self._set_secondary_time_index(dataframe, secondary_time_index)

    def _set_secondary_time_index(self, dataframe, secondary_time_index):
        """Sets the secondary time index for a Woodwork dataframe passed in"""
        assert (
            dataframe.ww.schema is not None
        ), "Cannot set secondary time index if Woodwork is not initialized"
        self._check_secondary_time_index(dataframe, secondary_time_index)
        if secondary_time_index is not None:
            dataframe.ww.metadata["secondary_time_index"] = secondary_time_index

    ###########################################################################
    #   Relationship access/helper methods  ###################################
    ###########################################################################

    def find_forward_paths(self, start_dataframe_name, goal_dataframe_name):
        """
        Generator which yields all forward paths between a start and goal
        dataframe. Does not include paths which contain cycles.

        Args:
            start_dataframe_name (str) : name of dataframe to start the search from
            goal_dataframe_name  (str) : name of dataframe to find forward path to

        See Also:
            :func:`BaseEntitySet.find_backward_paths`
        """
        for sub_dataframe_name, path in self._forward_dataframe_paths(
            start_dataframe_name,
        ):
            if sub_dataframe_name == goal_dataframe_name:
                yield path

    def find_backward_paths(self, start_dataframe_name, goal_dataframe_name):
        """
        Generator which yields all backward paths between a start and goal
        dataframe. Does not include paths which contain cycles.

        Args:
            start_dataframe_name (str) : Name of dataframe to start the search from.
            goal_dataframe_name  (str) : Name of dataframe to find backward path to.

        See Also:
            :func:`BaseEntitySet.find_forward_paths`
        """
        for path in self.find_forward_paths(goal_dataframe_name, start_dataframe_name):
            # Reverse path
            yield path[::-1]

    def _forward_dataframe_paths(self, start_dataframe_name, seen_dataframes=None):
        """
        Generator which yields the names of all dataframes connected through forward
        relationships, and the path taken to each. A dataframe will be yielded
        multiple times if there are multiple paths to it.

        Implemented using depth first search.
        """
        if seen_dataframes is None:
            seen_dataframes = set()

        if start_dataframe_name in seen_dataframes:
            return

        seen_dataframes.add(start_dataframe_name)

        yield start_dataframe_name, []

        for relationship in self.get_forward_relationships(start_dataframe_name):
            next_dataframe = relationship._parent_dataframe_name
            # Copy seen dataframes for each next node to allow multiple paths (but
            # not cycles).
            descendants = self._forward_dataframe_paths(
                next_dataframe,
                seen_dataframes.copy(),
            )
            for sub_dataframe_name, sub_path in descendants:
                yield sub_dataframe_name, [relationship] + sub_path

    def get_forward_dataframes(self, dataframe_name, deep=False):
        """
        Get dataframes that are in a forward relationship with dataframe

        Args:
            dataframe_name (str): Name of dataframe to search from.
            deep (bool): if True, recursively find forward dataframes.

        Yields a tuple of (descendent_name, path from dataframe_name to descendant).
        """
        for relationship in self.get_forward_relationships(dataframe_name):
            parent_dataframe_name = relationship._parent_dataframe_name
            direct_path = RelationshipPath([(True, relationship)])
            yield parent_dataframe_name, direct_path

            if deep:
                sub_dataframes = self.get_forward_dataframes(
                    parent_dataframe_name,
                    deep=True,
                )
                for sub_dataframe_name, path in sub_dataframes:
                    yield sub_dataframe_name, direct_path + path

    def get_backward_dataframes(self, dataframe_name, deep=False):
        """
        Get dataframes that are in a backward relationship with dataframe

        Args:
            dataframe_name (str): Name of dataframe to search from.
            deep (bool): if True, recursively find backward dataframes.

        Yields a tuple of (descendent_name, path from dataframe_name to descendant).
        """
        for relationship in self.get_backward_relationships(dataframe_name):
            child_dataframe_name = relationship._child_dataframe_name
            direct_path = RelationshipPath([(False, relationship)])
            yield child_dataframe_name, direct_path

            if deep:
                sub_dataframes = self.get_backward_dataframes(
                    child_dataframe_name,
                    deep=True,
                )
                for sub_dataframe_name, path in sub_dataframes:
                    yield sub_dataframe_name, direct_path + path

    def get_forward_relationships(self, dataframe_name):
        """Get relationships where dataframe "dataframe_name" is the child

        Args:
            dataframe_name (str): Name of dataframe to get relationships for.

        Returns:
            list[:class:`.Relationship`]: List of forward relationships.
        """
        return [
            r for r in self.relationships if r._child_dataframe_name == dataframe_name
        ]

    def get_backward_relationships(self, dataframe_name):
        """
        get relationships where dataframe "dataframe_name" is the parent.

        Args:
            dataframe_name (str): Name of dataframe to get relationships for.

        Returns:
            list[:class:`.Relationship`]: list of backward relationships
        """
        return [
            r for r in self.relationships if r._parent_dataframe_name == dataframe_name
        ]

    def has_unique_forward_path(self, start_dataframe_name, end_dataframe_name):
        """
        Is the forward path from start to end unique?

        This will raise if there is no such path.
        """
        paths = self.find_forward_paths(start_dataframe_name, end_dataframe_name)

        next(paths)
        second_path = next(paths, None)

        return not second_path

    ###########################################################################
    #  DataFrame creation methods  ##############################################
    ###########################################################################

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
        """
        Add a DataFrame to the EntitySet with Woodwork typing information.

        Args:
            dataframe (pandas.DataFrame) : Dataframe containing the data.

            dataframe_name (str, optional) : Unique name to associate with this dataframe. Must be
                provided if Woodwork is not initialized on the input DataFrame.

            index (str, optional): Name of the column used to index the dataframe.
                Must be unique. If None, take the first column.

            logical_types (dict[str -> Woodwork.LogicalTypes/str, optional]):
                Keys are column names and values are logical types. Will be inferred if not specified.

            semantic_tags (dict[str -> str/set], optional):
                Keys are column names and values are semantic tags.

            make_index (bool, optional) : If True, assume index does not
                exist as a column in dataframe, and create a new column of that name
                using integers. Otherwise, assume index exists.

            time_index (str, optional): Name of the column containing
                time data. Type must be numeric or datetime in nature.

            secondary_time_index (dict[str -> list[str]]): Name of column containing time data to
                be used as a secondary time index mapped to a list of the columns in the dataframe
                associated with that secondary time index.

            already_sorted (bool, optional) : If True, assumes that input dataframe
                is already sorted by time. Defaults to False.

        Notes:

            Will infer logical types from the data.

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
                es.add_dataframe(dataframe_name="transactions",
                                 index="id",
                                 time_index="transaction_time",
                                 dataframe=transactions_df)

                es["transactions"]

        """
        logical_types = logical_types or {}
        semantic_tags = semantic_tags or {}

        if len(self.dataframes) > 0:
            if not isinstance(dataframe, type(self.dataframes[0])):
                raise ValueError(
                    "All dataframes must be of the same type. "
                    "Cannot add dataframe of type {} to an entityset with existing dataframes "
                    "of type {}".format(type(dataframe), type(self.dataframes[0])),
                )

        # Only allow string column names
        non_string_names = [
            name for name in dataframe.columns if not isinstance(name, str)
        ]
        if non_string_names:
            raise ValueError(
                "All column names must be strings (Columns {} "
                "are not strings)".format(non_string_names),
            )

        if dataframe.ww.schema is None:
            if dataframe_name is None:
                raise ValueError(
                    "Cannot add dataframe to EntitySet without a name. "
                    "Please provide a value for the dataframe_name parameter.",
                )
            # Warn when performing inference on Dask or Spark DataFrames
            if not set(dataframe.columns).issubset(set(logical_types.keys())) and (
                is_instance(dataframe, dd, "DataFrame")
                or is_instance(dataframe, ps, "DataFrame")
            ):
                warnings.warn(
                    "Performing type inference on Dask or Spark DataFrames may be computationally intensive. "
                    "Specify logical types for each column to speed up EntitySet initialization.",
                )

            index_was_created, index, dataframe = _get_or_create_index(
                index,
                make_index,
                dataframe,
            )

            dataframe.ww.init(
                name=dataframe_name,
                index=index,
                time_index=time_index,
                logical_types=logical_types,
                semantic_tags=semantic_tags,
                already_sorted=already_sorted,
            )
            if index_was_created:
                dataframe.ww.metadata["created_index"] = index

        else:
            if dataframe.ww.name is None:
                raise ValueError(
                    "Cannot add a Woodwork DataFrame to EntitySet without a name",
                )
            if dataframe.ww.index is None:
                raise ValueError(
                    "Cannot add Woodwork DataFrame to EntitySet without index",
                )

            extra_params = []
            if index is not None:
                extra_params.append("index")
            if time_index is not None:
                extra_params.append("time_index")
            if logical_types:
                extra_params.append("logical_types")
            if make_index:
                extra_params.append("make_index")
            if semantic_tags:
                extra_params.append("semantic_tags")
            if already_sorted:
                extra_params.append("already_sorted")
            if dataframe_name is not None and dataframe_name != dataframe.ww.name:
                extra_params.append("dataframe_name")
            if extra_params:
                warnings.warn(
                    "A Woodwork-initialized DataFrame was provided, so the following parameters were ignored: "
                    + ", ".join(extra_params),
                )

        if dataframe.ww.time_index is not None:
            self._check_uniform_time_index(dataframe)
            self._check_secondary_time_index(dataframe)

        if secondary_time_index:
            self._set_secondary_time_index(
                dataframe,
                secondary_time_index=secondary_time_index,
            )

        dataframe = self._normalize_values(dataframe)

        self.dataframe_dict[dataframe.ww.name] = dataframe
        self.reset_data_description()
        self._add_references_to_metadata(dataframe)

        return self

    def __setitem__(self, key, value):
        self.add_dataframe(dataframe=value, dataframe_name=key)

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
        """Create a new dataframe and relationship from unique values of an existing column.

        Args:
            base_dataframe_name (str) : Dataframe name from which to split.

            new_dataframe_name (str): Name of the new dataframe.

            index (str): Column in old dataframe
                that will become index of new dataframe. Relationship
                will be created across this column.

            additional_columns (list[str]):
                List of column names to remove from
                base_dataframe and move to new dataframe.

            copy_columns (list[str]): List of
                column names to copy from old dataframe
                and move to new dataframe.

            make_time_index (bool or str, optional): Create time index for new dataframe based
                on time index in base_dataframe, optionally specifying which column in base_dataframe
                to use for time_index. If specified as True without a specific column name,
                uses the primary time index. Defaults to True if base dataframe has a time index.

            make_secondary_time_index (dict[str -> list[str]], optional): Create a secondary time index
                from key. Values of dictionary are the columns to associate with a secondary time index.
                Only one secondary time index is allowed. If None, only associate the time index.

            new_dataframe_time_index (str, optional): Rename new dataframe time index.

            new_dataframe_secondary_time_index (str, optional): Rename new dataframe secondary time index.

        """
        base_dataframe = self.dataframe_dict[base_dataframe_name]
        additional_columns = additional_columns or []
        copy_columns = copy_columns or []

        for list_name, col_list in {
            "copy_columns": copy_columns,
            "additional_columns": additional_columns,
        }.items():
            if not isinstance(col_list, list):
                raise TypeError(
                    "'{}' must be a list, but received type {}".format(
                        list_name,
                        type(col_list),
                    ),
                )
            if len(col_list) != len(set(col_list)):
                raise ValueError(
                    f"'{list_name}' contains duplicate columns. All columns must be unique.",
                )
            for col_name in col_list:
                if col_name == index:
                    raise ValueError(
                        "Not adding {} as both index and column in {}".format(
                            col_name,
                            list_name,
                        ),
                    )

        for col in additional_columns:
            if col == base_dataframe.ww.time_index:
                raise ValueError(
                    "Not moving {} as it is the base time index column. Perhaps, move the column to the copy_columns.".format(
                        col,
                    ),
                )

        if isinstance(make_time_index, str):
            if make_time_index not in base_dataframe.columns:
                raise ValueError(
                    "'make_time_index' must be a column in the base dataframe",
                )
            elif make_time_index not in additional_columns + copy_columns:
                raise ValueError(
                    "'make_time_index' must be specified in 'additional_columns' or 'copy_columns'",
                )
        if index == base_dataframe.ww.index:
            raise ValueError(
                "'index' must be different from the index column of the base dataframe",
            )

        transfer_types = {}
        # Types will be a tuple of (logical_type, semantic_tags, column_metadata, column_description)
        transfer_types[index] = (
            base_dataframe.ww.logical_types[index],
            base_dataframe.ww.semantic_tags[index],
            base_dataframe.ww.columns[index].metadata,
            base_dataframe.ww.columns[index].description,
        )
        for col_name in additional_columns + copy_columns:
            # Remove any existing time index tags
            transfer_types[col_name] = (
                base_dataframe.ww.logical_types[col_name],
                (base_dataframe.ww.semantic_tags[col_name] - {"time_index"}),
                base_dataframe.ww.columns[col_name].metadata,
                base_dataframe.ww.columns[col_name].description,
            )

        # create and add new dataframe
        new_dataframe = self[base_dataframe_name].copy()

        if make_time_index is None and base_dataframe.ww.time_index is not None:
            make_time_index = True

        if isinstance(make_time_index, str):
            # Set the new time index to make_time_index.
            base_time_index = make_time_index
            new_dataframe_time_index = make_time_index
            already_sorted = new_dataframe_time_index == base_dataframe.ww.time_index
        elif make_time_index:
            # Create a new time index based on the base dataframe time index.
            base_time_index = base_dataframe.ww.time_index
            if new_dataframe_time_index is None:
                new_dataframe_time_index = "first_%s_time" % (base_dataframe.ww.name)

            already_sorted = True

            assert (
                base_dataframe.ww.time_index is not None
            ), "Base dataframe doesn't have time_index defined"

            if base_time_index not in [col for col in copy_columns]:
                copy_columns.append(base_time_index)

                time_index_types = (
                    base_dataframe.ww.logical_types[base_dataframe.ww.time_index],
                    base_dataframe.ww.semantic_tags[base_dataframe.ww.time_index],
                    base_dataframe.ww.columns[base_dataframe.ww.time_index].metadata,
                    base_dataframe.ww.columns[base_dataframe.ww.time_index].description,
                )
            else:
                # If base_time_index is in copy_columns then we've already added the transfer types
                # but since we're changing the name, we have to remove it
                time_index_types = transfer_types[base_dataframe.ww.time_index]
                del transfer_types[base_dataframe.ww.time_index]

            transfer_types[new_dataframe_time_index] = time_index_types

        else:
            new_dataframe_time_index = None
            already_sorted = False

        if new_dataframe_time_index is not None and new_dataframe_time_index == index:
            raise ValueError(
                "time_index and index cannot be the same value, %s"
                % (new_dataframe_time_index),
            )

        selected_columns = (
            [index]
            + [col for col in additional_columns]
            + [col for col in copy_columns]
        )

        new_dataframe = new_dataframe.dropna(subset=[index])
        new_dataframe2 = new_dataframe.drop_duplicates(index, keep="first")[
            selected_columns
        ]

        if make_time_index:
            new_dataframe2 = new_dataframe2.rename(
                columns={base_time_index: new_dataframe_time_index},
            )
        if make_secondary_time_index:
            assert (
                len(make_secondary_time_index) == 1
            ), "Can only provide 1 secondary time index"
            secondary_time_index = list(make_secondary_time_index.keys())[0]

            secondary_columns = [index, secondary_time_index] + list(
                make_secondary_time_index.values(),
            )[0]
            secondary_df = new_dataframe.drop_duplicates(index, keep="last")[
                secondary_columns
            ]
            if new_dataframe_secondary_time_index:
                secondary_df = secondary_df.rename(
                    columns={secondary_time_index: new_dataframe_secondary_time_index},
                )
                secondary_time_index = new_dataframe_secondary_time_index
            else:
                new_dataframe_secondary_time_index = secondary_time_index
            secondary_df = secondary_df.set_index(index)
            new_dataframe = new_dataframe2.join(secondary_df, on=index)
        else:
            new_dataframe = new_dataframe2

        base_dataframe_index = index

        if make_secondary_time_index:
            old_ti_name = list(make_secondary_time_index.keys())[0]
            ti_cols = list(make_secondary_time_index.values())[0]
            ti_cols = [c if c != old_ti_name else secondary_time_index for c in ti_cols]
            make_secondary_time_index = {secondary_time_index: ti_cols}

        if is_instance(new_dataframe, ps, "DataFrame"):
            already_sorted = False

        # will initialize Woodwork on this DataFrame
        logical_types = {}
        semantic_tags = {}
        column_metadata = {}
        column_descriptions = {}
        for col_name, (ltype, tags, metadata, description) in transfer_types.items():
            logical_types[col_name] = ltype
            semantic_tags[col_name] = tags - {"time_index"}
            column_metadata[col_name] = copy.deepcopy(metadata)
            column_descriptions[col_name] = description

        new_dataframe.ww.init(
            name=new_dataframe_name,
            index=index,
            already_sorted=already_sorted,
            time_index=new_dataframe_time_index,
            logical_types=logical_types,
            semantic_tags=semantic_tags,
            column_metadata=column_metadata,
            column_descriptions=column_descriptions,
        )

        self.add_dataframe(
            new_dataframe,
            secondary_time_index=make_secondary_time_index,
        )

        self.dataframe_dict[base_dataframe_name] = self.dataframe_dict[
            base_dataframe_name
        ].ww.drop(additional_columns)

        self.dataframe_dict[base_dataframe_name].ww.add_semantic_tags(
            {base_dataframe_index: "foreign_key"},
        )

        self.add_relationship(
            new_dataframe_name,
            index,
            base_dataframe_name,
            base_dataframe_index,
        )
        self.reset_data_description()
        return self

    # ###########################################################################
    # #  Data wrangling methods  ###############################################
    # ###########################################################################

    def concat(self, other, inplace=False):
        """Combine entityset with another to create a new entityset with the
        combined data of both entitysets.
        """
        if not self.__eq__(other):
            raise ValueError(
                "Entitysets must have the same dataframes, relationships"
                ", and column names",
            )

        if inplace:
            combined_es = self
        else:
            combined_es = copy.deepcopy(self)

        lib = pd
        if self.dataframe_type == Library.SPARK:
            lib = ps
        elif self.dataframe_type == Library.DASK:
            lib = dd

        has_last_time_index = []
        for df in self.dataframes:
            self_df = df
            other_df = other[df.ww.name]
            combined_df = lib.concat([self_df, other_df])
            # If both DataFrames have made indexes, there will likely
            # be overlap in the index column, so we use the other values
            if self_df.ww.metadata.get("created_index") or other_df.ww.metadata.get(
                "created_index",
            ):
                columns = [
                    col
                    for col in combined_df.columns
                    if col != df.ww.index or col != df.ww.time_index
                ]
            else:
                columns = [df.ww.index]
            combined_df.drop_duplicates(columns, inplace=True)

            self_lti_col = df.ww.metadata.get("last_time_index")
            other_lti_col = other[df.ww.name].ww.metadata.get("last_time_index")
            if self_lti_col is not None or other_lti_col is not None:
                has_last_time_index.append(df.ww.name)

            combined_es.replace_dataframe(
                dataframe_name=df.ww.name,
                df=combined_df,
                recalculate_last_time_indexes=False,
                already_sorted=False,
            )

        if has_last_time_index:
            combined_es.add_last_time_indexes(updated_dataframes=has_last_time_index)

        combined_es.reset_data_description()

        return combined_es

    ###########################################################################
    #  Indexing methods  ###############################################
    ###########################################################################
    def add_last_time_indexes(self, updated_dataframes=None):
        """
        Calculates the last time index values for each dataframe (the last time
        an instance or children of that instance were observed).  Used when
        calculating features using training windows. Adds the last time index as
        a series named _ft_last_time on the dataframe.

        Args:
            updated_dataframes (list[str]): List of dataframe names to update last_time_index for
                (will update all parents of those dataframes as well)
        """
        # Generate graph of dataframes to find leaf dataframes
        children = defaultdict(list)  # parent --> child mapping
        child_cols = defaultdict(dict)
        for r in self.relationships:
            children[r._parent_dataframe_name].append(r.child_dataframe)
            child_cols[r._parent_dataframe_name][
                r._child_dataframe_name
            ] = r.child_column

        updated_dataframes = updated_dataframes or []
        if updated_dataframes:
            # find parents of updated_dataframes
            parent_queue = updated_dataframes[:]
            parents = set()
            while len(parent_queue):
                df_name = parent_queue.pop(0)
                if df_name in parents:
                    continue
                parents.add(df_name)

                for parent_name, _ in self.get_forward_dataframes(df_name):
                    parent_queue.append(parent_name)

            queue = [self[p] for p in parents]
            to_explore = parents
        else:
            to_explore = set(self.dataframe_dict.keys())
            queue = self.dataframes[:]

        explored = set()
        # Store the last time indexes for the entire entityset in a dictionary to update
        es_lti_dict = {}
        for df in self.dataframes:
            lti_col = df.ww.metadata.get("last_time_index")
            if lti_col is not None:
                lti_col = df[lti_col]
            es_lti_dict[df.ww.name] = lti_col

        for df in queue:
            es_lti_dict[df.ww.name] = None

        # We will explore children of dataframes on the queue,
        # which may not be in the to_explore set. Therefore,
        # we check whether all elements of to_explore are in
        # explored, rather than just comparing length
        while not to_explore.issubset(explored):
            dataframe = queue.pop(0)

            if es_lti_dict[dataframe.ww.name] is None:
                if dataframe.ww.time_index is not None:
                    lti = dataframe[dataframe.ww.time_index].copy()
                    if is_instance(dataframe, dd, "DataFrame"):
                        # The current Dask implementation doesn't set the index of the dataframe
                        # to the dataframe's index, so we have to do it manually here
                        lti.index = dataframe[dataframe.ww.index].copy()
                else:
                    lti = dataframe.ww[dataframe.ww.index].copy()
                    if is_instance(dataframe, dd, "DataFrame"):
                        lti.index = dataframe[dataframe.ww.index].copy()
                        lti = lti.apply(lambda x: None)
                    elif is_instance(dataframe, ps, "DataFrame"):
                        lti = ps.Series(pd.Series(index=lti.to_list(), name=lti.name))
                    else:
                        # Cannot have a category dtype with nans when calculating last time index
                        lti = lti.astype("object")
                        lti[:] = None

                es_lti_dict[dataframe.ww.name] = lti

            if dataframe.ww.name in children:
                child_dataframes = children[dataframe.ww.name]

                # if all children not explored, skip for now
                if not set([df.ww.name for df in child_dataframes]).issubset(explored):
                    # Now there is a possibility that a child dataframe
                    # was not explicitly provided in updated_dataframes,
                    # and never made it onto the queue. If updated_dataframes
                    # is None then we just load all dataframes onto the queue
                    # so we didn't need this logic
                    for df in child_dataframes:
                        if df.ww.name not in explored and df.ww.name not in [
                            q.ww.name for q in queue
                        ]:
                            # must also reset last time index here
                            es_lti_dict[df.ww.name] = None
                            queue.append(df)
                    queue.append(dataframe)
                    continue

                # updated last time from all children
                for child_df in child_dataframes:
                    # TODO: Figure out if Dask code related to indexes is important for Spark
                    if es_lti_dict[child_df.ww.name] is None:
                        continue
                    link_col = child_cols[dataframe.ww.name][child_df.ww.name].name

                    lti_is_dask = is_instance(
                        es_lti_dict[child_df.ww.name],
                        dd,
                        "Series",
                    )
                    lti_is_spark = is_instance(
                        es_lti_dict[child_df.ww.name],
                        ps,
                        "Series",
                    )

                    if lti_is_dask or lti_is_spark:
                        to_join = child_df[link_col]
                        if lti_is_dask:
                            to_join.index = child_df[child_df.ww.index]

                        lti_df = (
                            es_lti_dict[child_df.ww.name]
                            .to_frame(name="last_time")
                            .join(to_join.to_frame(name=dataframe.ww.index))
                        )

                        if lti_is_dask:
                            new_index = lti_df.index.copy()
                            new_index.name = None
                            lti_df.index = new_index
                        lti_df = lti_df.groupby(lti_df[dataframe.ww.index]).agg("max")

                        lti_df = (
                            es_lti_dict[dataframe.ww.name]
                            .to_frame(name="last_time_old")
                            .join(lti_df)
                        )

                    else:
                        lti_df = pd.DataFrame(
                            {
                                "last_time": es_lti_dict[child_df.ww.name],
                                dataframe.ww.index: child_df[link_col],
                            },
                        )

                        # sort by time and keep only the most recent
                        lti_df.sort_values(
                            ["last_time", dataframe.ww.index],
                            kind="mergesort",
                            inplace=True,
                        )

                        lti_df.drop_duplicates(
                            dataframe.ww.index,
                            keep="last",
                            inplace=True,
                        )

                        lti_df.set_index(dataframe.ww.index, inplace=True)
                        lti_df = lti_df.reindex(es_lti_dict[dataframe.ww.name].index)
                        lti_df["last_time_old"] = es_lti_dict[dataframe.ww.name]
                    if not (lti_is_dask or lti_is_spark) and lti_df.empty:
                        # Pandas errors out if it tries to do fillna and then max on an empty dataframe
                        lti_df = pd.Series([], dtype="object")
                    else:
                        if lti_is_spark:
                            # TODO: Figure out a workaround for fillna and replace
                            if lti_df["last_time_old"].dtype != "datetime64[ns]":
                                lti_df["last_time_old"] = ps.to_datetime(
                                    lti_df["last_time_old"],
                                )
                            if lti_df["last_time"].dtype != "datetime64[ns]":
                                lti_df["last_time"] = ps.to_datetime(
                                    lti_df["last_time"],
                                )
                            lti_df = lti_df.max(axis=1)
                        else:
                            lti_df["last_time"] = lti_df["last_time"].astype(
                                "datetime64[ns]",
                            )
                            lti_df["last_time_old"] = lti_df["last_time_old"].astype(
                                "datetime64[ns]",
                            )
                            lti_df = lti_df.fillna(
                                pd.to_datetime("1800-01-01 00:00"),
                            ).max(axis=1)
                            lti_df = lti_df.replace(
                                pd.to_datetime("1800-01-01 00:00"),
                                pd.NaT,
                            )

                    es_lti_dict[dataframe.ww.name] = lti_df
                    es_lti_dict[dataframe.ww.name].name = "last_time"

            explored.add(dataframe.ww.name)

        # Store the last time index on the DataFrames
        dfs_to_update = {}
        for df in self.dataframes:
            lti = es_lti_dict[df.ww.name]
            if lti is not None:
                lti_ltype = None
                if self.time_type == "numeric":
                    if lti.dtype == "datetime64[ns]":
                        # Woodwork cannot convert from datetime to numeric
                        lti = lti.apply(lambda x: x.value)
                    lti = init_series(lti, logical_type="Double")
                    lti_ltype = "Double"
                else:
                    lti = init_series(lti, logical_type="Datetime")
                    lti_ltype = "Datetime"

                lti.name = LTI_COLUMN_NAME

                if LTI_COLUMN_NAME in df.columns:
                    if "last_time_index" in df.ww.semantic_tags[LTI_COLUMN_NAME]:
                        # Remove any previous last time index placed by featuretools
                        df.ww.pop(LTI_COLUMN_NAME)
                    else:
                        raise ValueError(
                            "Cannot add a last time index on DataFrame with an existing "
                            f"'{LTI_COLUMN_NAME}' column. Please rename '{LTI_COLUMN_NAME}'.",
                        )

                # Add the new column to the DataFrame
                if is_instance(df, dd, "DataFrame"):
                    new_df = df.merge(lti.reset_index(), on=df.ww.index)
                    new_df.ww.init_with_partial_schema(
                        schema=df.ww.schema,
                        logical_types={LTI_COLUMN_NAME: lti_ltype},
                    )

                    new_idx = new_df[new_df.ww.index]
                    new_idx.name = None
                    new_df.index = new_idx
                    dfs_to_update[df.ww.name] = new_df
                elif is_instance(df, ps, "DataFrame"):
                    new_df = df.merge(lti, left_on=df.ww.index, right_index=True)
                    new_df.ww.init_with_partial_schema(
                        schema=df.ww.schema,
                        logical_types={LTI_COLUMN_NAME: lti_ltype},
                    )

                    dfs_to_update[df.ww.name] = new_df
                else:
                    df.ww[LTI_COLUMN_NAME] = lti
                    if "last_time_index" not in df.ww.semantic_tags[LTI_COLUMN_NAME]:
                        df.ww.add_semantic_tags({LTI_COLUMN_NAME: "last_time_index"})
                    df.ww.metadata["last_time_index"] = LTI_COLUMN_NAME

        for df in dfs_to_update.values():
            df.ww.add_semantic_tags({LTI_COLUMN_NAME: "last_time_index"})
            df.ww.metadata["last_time_index"] = LTI_COLUMN_NAME
            self.dataframe_dict[df.ww.name] = df

        self.reset_data_description()
        for df in self.dataframes:
            self._add_references_to_metadata(df)

    # ###########################################################################
    # #  Pickling ###############################################
    # ###########################################################################
    def __getstate__(self):
        return {
            **self.__dict__,
            WW_SCHEMA_KEY: {
                df_name: df.ww.schema for df_name, df in self.dataframe_dict.items()
            },
        }

    def __setstate__(self, state):
        ww_schemas = state.pop(WW_SCHEMA_KEY)
        for df_name, df in state.get("dataframe_dict", {}).items():
            if ww_schemas[df_name] is not None:
                df.ww.init(schema=ww_schemas[df_name], validate=False)

        self.__dict__.update(state)

    # ###########################################################################
    # #  Other ###############################################
    # ###########################################################################
    def add_interesting_values(
        self,
        max_values=5,
        verbose=False,
        dataframe_name=None,
        values=None,
    ):
        """Find or set interesting values for categorical columns, to be used to generate "where" clauses

        Args:
            max_values (int) : Maximum number of values per column to add.
            verbose (bool) : If True, print summary of interesting values found.
            dataframe_name (str) : The dataframe in the EntitySet for which to add interesting values.
                If not specified interesting values will be added for all dataframes.
            values (dict): A dictionary mapping column names to the interesting values to set
                for the column. If specified, a corresponding dataframe_name must also be provided.
                If not specified, interesting values will be set for all eligible columns. If values
                are specified, max_values and verbose parameters will be ignored.

        Notes:

            Finding interesting values is not supported with Dask or Spark EntitySets.
            To set interesting values for Dask or Spark EntitySets, values must be
            specified with the ``values`` parameter.

        Returns:
            None

        """
        if dataframe_name is None and values is not None:
            raise ValueError("dataframe_name must be specified if values are provided")

        if dataframe_name is not None and values is not None:
            for column, vals in values.items():
                self[dataframe_name].ww.columns[column].metadata[
                    "interesting_values"
                ] = vals
            return

        if dataframe_name:
            dataframes = [self[dataframe_name]]
        else:
            dataframes = self.dataframes

        def add_value(df, col, val, verbose):
            if verbose:
                msg = "Column {}: Marking {} as an interesting value"
                logger.info(msg.format(col, val))
            interesting_vals = df.ww.columns[col].metadata.get("interesting_values", [])
            interesting_vals.append(val)
            df.ww.columns[col].metadata["interesting_values"] = interesting_vals

        for df in dataframes:
            value_counts = df.ww.value_counts(top_n=max(25, max_values), dropna=True)
            total_count = len(df)

            for col, counts in value_counts.items():
                if {"index", "foreign_key"}.intersection(df.ww.semantic_tags[col]):
                    continue

                for i in range(min(max_values, len(counts))):
                    # Categorical columns will include counts of 0 for all values
                    # in categories. Stop when we encounter a 0 count.
                    if counts[i]["count"] == 0:
                        break
                    if len(counts) < 25:
                        value = counts[i]["value"]
                        add_value(df, col, value, verbose)
                    else:
                        fraction = counts[i]["count"] / total_count
                        if fraction > 0.05 and fraction < 0.95:
                            value = counts[i]["value"]
                            add_value(df, col, value, verbose)
                        else:
                            break

        self.reset_data_description()

    def plot(self, to_file=None):
        """
        Create a UML diagram-ish graph of the EntitySet.

        Args:
            to_file (str, optional) : Path to where the plot should be saved.
                If set to None (as by default), the plot will not be saved.

        Returns:
            graphviz.Digraph : Graph object that can directly be displayed in
                Jupyter notebooks. Nodes of the graph correspond to the DataFrames
                in the EntitySet, showing the typing information for each column.

        Note:
            The typing information displayed for each column is based off of the Woodwork
            ColumnSchema for that column and is represented as ``LogicalType; semantic_tags``,
            but the standard semantic tags have been removed for brevity.
        """
        graphviz = check_graphviz()
        format_ = get_graphviz_format(graphviz=graphviz, to_file=to_file)

        # Initialize a new directed graph
        graph = graphviz.Digraph(
            self.id,
            format=format_,
            graph_attr={"splines": "ortho"},
        )

        # Draw dataframes
        for df in self.dataframes:
            column_typing_info = []
            for col_name, col_schema in df.ww.columns.items():
                col_string = col_name + " : " + str(col_schema.logical_type)

                tags = col_schema.semantic_tags - col_schema.logical_type.standard_tags
                if tags:
                    col_string += "; "
                    col_string += ", ".join(tags)
                column_typing_info.append(col_string)

            columns_string = "\l".join(column_typing_info)  # noqa: W605
            if is_instance(df, dd, "DataFrame"):  # dataframe is a dask dataframe
                label = "{%s |%s\l}" % (df.ww.name, columns_string)  # noqa: W605
            else:
                nrows = df.shape[0]
                label = "{%s (%d row%s)|%s\l}" % (  # noqa: W605
                    df.ww.name,
                    nrows,
                    "s" * (nrows > 1),
                    columns_string,
                )
            graph.node(df.ww.name, shape="record", label=label)

        # Draw relationships
        for rel in self.relationships:
            # Display the key only once if is the same for both related dataframes
            if rel._parent_column_name == rel._child_column_name:
                label = rel._parent_column_name
            else:
                label = "%s -> %s" % (rel._parent_column_name, rel._child_column_name)

            graph.edge(
                rel._child_dataframe_name,
                rel._parent_dataframe_name,
                xlabel=label,
            )

        if to_file:
            save_graph(graph, to_file, format_)
        return graph

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
        """

        schema = self[dataframe_name].ww.schema
        if is_instance(df, ps, "DataFrame") and isinstance(time_last, np.datetime64):
            time_last = pd.to_datetime(time_last)
        if schema.time_index:
            df_empty = df.empty if isinstance(df, pd.DataFrame) else False
            if time_last is not None and not df_empty:
                if include_cutoff_time:
                    df = df[df[schema.time_index] <= time_last]
                else:
                    df = df[df[schema.time_index] < time_last]
                if training_window is not None:
                    training_window = _check_timedelta(training_window)
                    if include_cutoff_time:
                        mask = df[schema.time_index] > time_last - training_window
                    else:
                        mask = df[schema.time_index] >= time_last - training_window
                    lti_col = schema.metadata.get("last_time_index")
                    if lti_col is not None:
                        if include_cutoff_time:
                            lti_mask = df[lti_col] > time_last - training_window
                        else:
                            lti_mask = df[lti_col] >= time_last - training_window
                        mask = mask | lti_mask
                    else:
                        warnings.warn(
                            "Using training_window but last_time_index is "
                            "not set for dataframe %s" % (dataframe_name),
                        )

                    df = df[mask]

        secondary_time_indexes = schema.metadata.get("secondary_time_index") or {}
        for secondary_time_index, columns in secondary_time_indexes.items():
            # should we use ignore time last here?
            df_empty = df.empty if isinstance(df, pd.DataFrame) else False
            if time_last is not None and not df_empty:
                mask = df[secondary_time_index] >= time_last
                if is_instance(df, dd, "DataFrame"):
                    for col in columns:
                        df[col] = df[col].mask(mask, np.nan)
                elif is_instance(df, ps, "DataFrame"):
                    df.loc[mask, columns] = None
                else:
                    df.loc[mask, columns] = np.nan

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
        """Query instances that have column with given value

        Args:
            dataframe_name (str): The id of the dataframe to query
            instance_vals (pd.Dataframe, pd.Series, list[str] or str) :
                Instance(s) to match.
            column_name (str) : Column to query on. If None, query on index.
            columns (list[str]) : Columns to return. Return all columns if None.
            time_last (pd.TimeStamp) : Query data up to and including this
                time. Only applies if dataframe has a time index.
            training_window (Timedelta, optional):
                Window defining how much time before the cutoff time data
                can be used when calculating features. If None, all data before cutoff time is used.
            include_cutoff_time (bool):
                If True, data at cutoff time are included in calculating features

        Returns:
            pd.DataFrame : instances that match constraints with ids in order of underlying dataframe
        """
        dataframe = self[dataframe_name]
        if not column_name:
            column_name = dataframe.ww.index

        instance_vals = _vals_to_series(instance_vals, column_name)

        training_window = _check_timedelta(training_window)

        if training_window is not None:
            assert (
                training_window.has_no_observations()
            ), "Training window cannot be in observations"

        if instance_vals is None:
            df = dataframe.copy()

        elif isinstance(instance_vals, pd.Series) and instance_vals.empty:
            df = dataframe.head(0)

        else:
            if is_instance(instance_vals, (dd, ps), "Series"):
                df = dataframe.merge(
                    instance_vals.to_frame(),
                    how="inner",
                    on=column_name,
                )
            elif isinstance(instance_vals, pd.Series) and is_instance(
                dataframe,
                ps,
                "DataFrame",
            ):
                df = dataframe.merge(
                    ps.DataFrame({column_name: instance_vals}),
                    how="inner",
                    on=column_name,
                )
            else:
                df = dataframe[dataframe[column_name].isin(instance_vals)]

            if isinstance(dataframe, pd.DataFrame):
                df = df.set_index(dataframe.ww.index, drop=False)

            # ensure filtered df has same categories as original
            # workaround for issue below
            # github.com/pandas-dev/pandas/issues/22501#issuecomment-415982538
            #
            # Pandas claims that bug is fixed but it still shows up in some
            # cases.  More investigation needed.
            #
            # Note: Woodwork stores categorical columns with a `string` dtype for Spark
            if dataframe.ww.columns[column_name].is_categorical and not is_instance(
                df,
                ps,
                "DataFrame",
            ):
                categories = pd.api.types.CategoricalDtype(
                    categories=dataframe[column_name].cat.categories,
                )
                df[column_name] = df[column_name].astype(categories)

        df = self._handle_time(
            dataframe_name=dataframe_name,
            df=df,
            time_last=time_last,
            training_window=training_window,
            include_cutoff_time=include_cutoff_time,
        )

        if columns is not None:
            df = df[columns]

        return df

    def replace_dataframe(
        self,
        dataframe_name,
        df,
        already_sorted=False,
        recalculate_last_time_indexes=True,
    ):
        """Replace the internal dataframe of an EntitySet table, keeping Woodwork typing information the same.
        Optionally makes sure that data is sorted, that reference indexes to other dataframes are consistent,
        and that last_time_indexes are updated to reflect the new data. If an index was created for the original
        dataframe and is not present on the new dataframe, an index column of the same name will be added to the
        new dataframe.
        """
        if not isinstance(df, type(self[dataframe_name])):
            raise TypeError("Incorrect DataFrame type used")

        # If the original DataFrame has a last time index column and the new one doesnt
        # remove the column and the reference to last time index from that dataframe
        last_time_index_column = self[dataframe_name].ww.metadata.get("last_time_index")
        if (
            last_time_index_column is not None
            and last_time_index_column not in df.columns
        ):
            self[dataframe_name].ww.pop(last_time_index_column)
            del self[dataframe_name].ww.metadata["last_time_index"]

        # If the original DataFrame had an index created via make_index,
        # we may need to remake the index if it's not in the new DataFrame
        created_index = self[dataframe_name].ww.metadata.get("created_index")
        if created_index is not None and created_index not in df.columns:
            df = _create_index(df, created_index)

        old_column_names = list(self[dataframe_name].columns)
        if len(df.columns) != len(old_column_names):
            raise ValueError(
                "New dataframe contains {} columns, expecting {}".format(
                    len(df.columns),
                    len(old_column_names),
                ),
            )
        for col_name in old_column_names:
            if col_name not in df.columns:
                raise ValueError(
                    "New dataframe is missing new {} column".format(col_name),
                )

        if df.ww.schema is not None:
            warnings.warn(
                "Woodwork typing information on new dataframe will be replaced "
                f"with existing typing information from {dataframe_name}",
            )

        df.ww.init(
            schema=self[dataframe_name].ww._schema,
            already_sorted=already_sorted,
        )
        # Make sure column ordering matches original ordering
        df = df.ww[old_column_names]

        df = self._normalize_values(df)

        self.dataframe_dict[dataframe_name] = df

        if self[dataframe_name].ww.time_index is not None:
            self._check_uniform_time_index(self[dataframe_name])

        df_metadata = self[dataframe_name].ww.metadata
        self.set_secondary_time_index(
            dataframe_name,
            df_metadata.get("secondary_time_index"),
        )
        if recalculate_last_time_indexes and last_time_index_column is not None:
            self.add_last_time_indexes(updated_dataframes=[dataframe_name])
        self.reset_data_description()
        self._add_references_to_metadata(df)

    def _check_time_indexes(self):
        for dataframe in self.dataframe_dict.values():
            self._check_uniform_time_index(dataframe)
            self._check_secondary_time_index(dataframe)

    def _check_secondary_time_index(self, dataframe, secondary_time_index=None):
        secondary_time_index = secondary_time_index or dataframe.ww.metadata.get(
            "secondary_time_index",
            {},
        )

        if secondary_time_index and dataframe.ww.time_index is None:
            raise ValueError(
                "Cannot set secondary time index on a DataFrame that has no primary time index.",
            )

        for time_index, columns in secondary_time_index.items():
            self._check_uniform_time_index(dataframe, column_name=time_index)
            if time_index not in columns:
                columns.append(time_index)

    def _check_uniform_time_index(self, dataframe, column_name=None):
        column_name = column_name or dataframe.ww.time_index
        if column_name is None:
            return

        time_type = self._get_time_type(dataframe, column_name)
        if self.time_type is None:
            self.time_type = time_type
        elif self.time_type != time_type:
            info = "%s time index is %s type which differs from other entityset time indexes"
            raise TypeError(info % (dataframe.ww.name, time_type))

    def _get_time_type(self, dataframe, column_name=None):
        column_name = column_name or dataframe.ww.time_index

        column_schema = dataframe.ww.columns[column_name]

        time_type = None
        if column_schema.is_numeric:
            time_type = "numeric"
        elif column_schema.is_datetime:
            time_type = Datetime

        if time_type is None:
            info = "%s time index not recognized as numeric or datetime"
            raise TypeError(info % dataframe.ww.name)
        return time_type

    def _add_references_to_metadata(self, dataframe):
        dataframe.ww.metadata.update(entityset_id=self.id)
        for column in dataframe.columns:
            metadata = dataframe.ww._schema.columns[column].metadata
            metadata.update(dataframe_name=dataframe.ww.name)
            metadata.update(entityset_id=self.id)
        _ES_REF[self.id] = self

    def _normalize_values(self, dataframe):
        def replace(x, is_spark=False):
            if not isinstance(x, (list, tuple, np.ndarray)) and pd.isna(x):
                if is_spark:
                    return [np.nan, np.nan]
                else:
                    return (np.nan, np.nan)
            else:
                return x

        for column, logical_type in dataframe.ww.logical_types.items():
            if isinstance(logical_type, LatLong):
                series = dataframe[column]
                if ps and isinstance(series, ps.Series):
                    if len(series):
                        dataframe[column] = dataframe[column].apply(
                            replace,
                            args=(True,),
                        )
                elif is_instance(dataframe, dd, "DataFrame"):
                    dataframe[column] = dataframe[column].apply(
                        replace,
                        meta=(column, logical_type.primary_dtype),
                    )
                else:
                    dataframe[column] = dataframe[column].apply(replace)
        return dataframe


def _vals_to_series(instance_vals, column_id):
    """
    instance_vals may be a pd.Dataframe, a pd.Series, a list, a single
    value, or None. This function always returns a Series or None.
    """
    if instance_vals is None:
        return None

    # If this is a single value, make it a list
    if not hasattr(instance_vals, "__iter__"):
        instance_vals = [instance_vals]

    # convert iterable to pd.Series
    if isinstance(instance_vals, pd.DataFrame):
        out_vals = instance_vals[column_id]
    elif is_instance(instance_vals, (pd, dd, ps), "Series"):
        out_vals = instance_vals.rename(column_id)
    else:
        out_vals = pd.Series(instance_vals)

    # no duplicates or NaN values
    out_vals = out_vals.drop_duplicates().dropna()

    # want index to have no name for the merge in query_by_values
    out_vals.index.name = None

    return out_vals


def _get_or_create_index(index, make_index, df):
    """Handles index creation logic base on user input"""
    index_was_created = False

    if index is None:
        # Case 1: user wanted to make index but did not specify column name
        assert not make_index, "Must specify an index name if make_index is True"
        # Case 2: make_index not specified but no index supplied, use first column
        warnings.warn(
            (
                "Using first column as index. "
                "To change this, specify the index parameter"
            ),
        )
        index = df.columns[0]
    elif make_index and index in df.columns:
        # Case 3: user wanted to make index but column already exists
        raise RuntimeError(
            f"Cannot make index: column with name {index} already present",
        )
    elif index not in df.columns:
        if not make_index:
            # Case 4: user names index, it is not in df. does not specify
            # make_index.  Make new index column and warn
            warnings.warn(
                "index {} not found in dataframe, creating new "
                "integer column".format(index),
            )
        # Case 5: make_index with no errors or warnings
        # (Case 4 also uses this code path)
        df = _create_index(df, index)
        index_was_created = True
    # Case 6: user specified index, which is already in df. No action needed.
    return index_was_created, index, df


def _create_index(df, index):
    if is_instance(df, dd, "DataFrame") or is_instance(df, ps, "DataFrame"):
        df[index] = 1
        df[index] = df[index].cumsum() - 1
    else:
        df.insert(0, index, range(len(df)))
    return df
