from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import pandas.api.types as pdtypes

from featuretools.entityset.relationship import RelationshipPath
from featuretools.exceptions import UnknownFeature
from featuretools.feature_base import (
    AggregationFeature,
    DirectFeature,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature,
)
from featuretools.utils import Trie
from featuretools.utils.gen_utils import get_relationship_column_id


class FeatureSetCalculator(object):
    """
    Calculates the values of a set of features for given instance ids.
    """

    def __init__(
        self,
        entityset,
        feature_set,
        time_last=None,
        training_window=None,
        precalculated_features=None,
    ):
        """
        Args:
            feature_set (FeatureSet): The features to calculate values for.

            time_last (pd.Timestamp, optional): Last allowed time. Data from exactly this
                time not allowed.

            training_window (Timedelta, optional): Window defining how much time before the cutoff time data
                can be used when calculating features. If None, all data before cutoff time is used.

            precalculated_features (Trie[RelationshipPath -> pd.DataFrame]):
                Maps RelationshipPaths to dataframes of precalculated_features

        """
        self.entityset = entityset
        self.feature_set = feature_set
        self.training_window = training_window

        if time_last is None:
            time_last = datetime.now()

        self.time_last = time_last

        if precalculated_features is None:
            precalculated_features = Trie(path_constructor=RelationshipPath)

        self.precalculated_features = precalculated_features

        # total number of features (including dependencies) to be calculate
        self.num_features = sum(
            len(features1) + len(features2)
            for _, (_, features1, features2) in self.feature_set.feature_trie
        )

    def run(self, instance_ids, progress_callback=None, include_cutoff_time=True):
        """
        Calculate values of features for the given instances of the target
        dataframe.

        Summary of algorithm:
        1. Construct a trie where the edges are relationships and each node
            contains a set of features for a single dataframe. See
            FeatureSet._build_feature_trie.
        2. Initialize a trie for storing dataframes.
        3. Traverse the trie using depth first search. At each node calculate
            the features and store the resulting dataframe in the dataframe
            trie (so that its values can be used by features which depend on
            these features). See _calculate_features_for_dataframe.
        4. Get the dataframe at the root of the trie (for the target dataframe) and
            return the columns corresponding to the requested features.

        Args:
            instance_ids (np.ndarray or pd.Categorical): Instance ids for which
                to build features.

            progress_callback (callable): function to be called with incremental progress updates

            include_cutoff_time (bool): If True, data at cutoff time are included
                in calculating features.

        Returns:
            pd.DataFrame : Pandas DataFrame of calculated feature values.
                Indexed by instance_ids. Columns in same order as features
                passed in.
        """
        assert len(instance_ids) > 0, "0 instance ids provided"

        if progress_callback is None:
            # do nothing for the progress call back if not provided
            def progress_callback(*args):
                pass

        feature_trie = self.feature_set.feature_trie

        df_trie = Trie(path_constructor=RelationshipPath)
        full_dataframe_trie = Trie(path_constructor=RelationshipPath)

        target_dataframe = self.entityset[self.feature_set.target_df_name]

        self._calculate_features_for_dataframe(
            dataframe_name=self.feature_set.target_df_name,
            feature_trie=feature_trie,
            df_trie=df_trie,
            full_dataframe_trie=full_dataframe_trie,
            precalculated_trie=self.precalculated_features,
            filter_column=target_dataframe.ww.index,
            filter_values=instance_ids,
            progress_callback=progress_callback,
            include_cutoff_time=include_cutoff_time,
        )

        # The dataframe for the target dataframe should be stored at the root of
        # df_trie.
        df = df_trie.value

        # Fill in empty rows with default values.
        index_dtype = df.index.dtype.name
        if df.empty:
            return self.generate_default_df(instance_ids=instance_ids)

        missing_ids = [
            i for i in instance_ids if i not in df[target_dataframe.ww.index]
        ]
        if missing_ids:
            default_df = self.generate_default_df(
                instance_ids=missing_ids,
                extra_columns=df.columns,
            )

            df = pd.concat([df, default_df], sort=True)

        df.index.name = self.entityset[self.feature_set.target_df_name].ww.index

        # Order by instance_ids
        unique_instance_ids = pd.unique(instance_ids)
        unique_instance_ids = unique_instance_ids.astype(instance_ids.dtype)
        df = df.reindex(unique_instance_ids)

        # Keep categorical index if original index was categorical
        if index_dtype == "category":
            df.index = df.index.astype("category")

        column_list = []

        for feat in self.feature_set.target_features:
            column_list.extend(feat.get_feature_names())

        return df[column_list]

    def _calculate_features_for_dataframe(
        self,
        dataframe_name,
        feature_trie,
        df_trie,
        full_dataframe_trie,
        precalculated_trie,
        filter_column,
        filter_values,
        parent_data=None,
        progress_callback=None,
        include_cutoff_time=True,
    ):
        """
        Generate dataframes with features calculated for this node of the trie,
        and all descendant nodes. The dataframes will be stored in df_trie.

        Args:
            dataframe_name (str): The name of the dataframe to calculate features for.

            feature_trie (Trie): the trie with sets of features to calculate.
                The root contains features for the given dataframe.

            df_trie (Trie): a parallel trie for storing dataframes. The
                dataframe with features calculated will be placed in the root.

            full_dataframe_trie (Trie): a trie storing dataframes will all dataframe
                rows, for features that are uses_full_dataframe.

            precalculated_trie (Trie): a parallel trie containing dataframes
                with precalculated features. The dataframe specified by dataframe_name
                will be at the root.

            filter_column (str): The name of the column to filter this
                dataframe by.

            filter_values (pd.Series): The values to filter the filter_column
                to.

            parent_data (tuple[Relationship, list[str], pd.DataFrame]): Data
                related to the parent of this trie. This will only be present if
                the relationship points from this dataframe to the parent dataframe. A
                3 tuple of (parent_relationship,
                ancestor_relationship_columns, parent_df).
                ancestor_relationship_columns is the names of columns which
                link the parent dataframe to its ancestors.

            include_cutoff_time (bool): If True, data at cutoff time are included
                in calculating features.

        """
        # Step 1: Get a dataframe for the given dataframe name, filtered by the given
        # conditions.

        (
            need_full_dataframe,
            full_dataframe_features,
            not_full_dataframe_features,
        ) = feature_trie.value

        all_features = full_dataframe_features | not_full_dataframe_features
        columns = self._necessary_columns(dataframe_name, all_features)

        # If we need the full dataframe then don't filter by filter_values.
        if need_full_dataframe:
            query_column = None
            query_values = None
        else:
            query_column = filter_column
            query_values = filter_values

        df = self.entityset.query_by_values(
            dataframe_name=dataframe_name,
            instance_vals=query_values,
            column_name=query_column,
            columns=columns,
            time_last=self.time_last,
            training_window=self.training_window,
            include_cutoff_time=include_cutoff_time,
        )

        # call to update timer
        progress_callback(0)

        # Step 2: Add columns to the dataframe linking it to all ancestors.
        new_ancestor_relationship_columns = []
        if parent_data:
            parent_relationship, ancestor_relationship_columns, parent_df = parent_data
            new_ancestor_relationship_columns =\
                self._add_ancestor_relationship_columns(
                    df, parent_df, ancestor_relationship_columns, parent_relationship
                )

        df_trie.value = self._calculate_features(df, df_trie, full_dataframe_features, progress_callback)

        # Add dataframe to full_dataframe_trie. We do this in case a feature depends on
        # a full dataframe with certain columns (i.e. parent features needed for a direct
        # feature). The full_dataframe_trie gets pruned at the end of the dfs so
        # only the necessary dataframes and columns are saved.
        full_dataframe_trie.value = self._calculate_features(df, full_dataframe_trie, all_features, progress_callback)

        # Step 3: Traverse the trie of features, calculating the features for children
        # and adding them to the dataframe.
        for relationship, child_feature_trie in feature_trie.items():
            self._calculate_features_for_dataframe(
                dataframe_name=relationship.child_dataframe.ww.name,
                feature_trie=child_feature_trie,
                df_trie=df_trie.get_node(relationship),
                full_dataframe_trie=full_dataframe_trie.get_node(relationship),
                precalculated_trie=precalculated_trie.get_node(relationship),
                filter_column=relationship.child_column.ww.name,
                filter_values=df[relationship.parent_column.ww.name],
                parent_data=(
                    relationship,
                    new_ancestor_relationship_columns,
                    df,
                ),
                progress_callback=progress_callback,
                include_cutoff_time=include_cutoff_time,
            )

        # Step 4: After calculating features for children, calculate transform and
        # agg features for this dataframe.
        final_df = self._calculate_features(df, df_trie, not_full_dataframe_features, progress_callback)
        df_trie.value = final_df

    def _calculate_features(self, df, df_trie, features, progress_callback):
        # Group the features so that each group can be calculated together.
        # The groups must also be in topological order (if A is a transform of B
        # then B must be in a group before A).
        for f in features.get_topologically_sorted_features():
            feature_type_handler = self._feature_type_handler(f)
            df = feature_type_handler(f, df, df_trie, progress_callback)

        return df

    def _add_ancestor_relationship_columns(
        self,
        child_df,
        parent_df,
        ancestor_relationship_columns,
        relationship,
    ):
        # add all ancestor relationship columns from parent to child
        for ancestor_column in ancestor_relationship_columns:
            new_col = parent_df.ww.pop(ancestor_column)
            new_col.ww.add.relationship(ancestor_column, relationship.parent_dataframe.ww.name)
            child_df[ancestor_column] = new_col
            child_df.ww.set_logical_type(ancestor_column, "id")

        # add the relationship column to child
        new_col = parent_df.ww.pop(relationship.parent_column.ww.name)
        new_col.ww.add.relationship(
            relationship.parent_column.ww.name,
            relationship.parent_dataframe.ww.name,
        )
        child_df[relationship.parent_column.ww.name] = new_col
        child_df.ww.set_logical_type(relationship.parent_column.ww.name, "id")

        ancestor_relationship_columns.append(relationship.parent_column.ww.name)
        return ancestor_relationship_columns

    def generate_default_df(self, instance_ids, extra_columns=None):
        # make a series of default values for each feature
        default_values = {}
        columns_to_make = extra_columns
        if columns_to_make is None:
            columns_to_make = [
                feat.get_name() for feat in self.feature_set.target_features
            ]
        for column_name in columns_to_make:
            default_values[column_name] = np.nan

        # convert to dataframe with right index and make sure values are correct
        default_df = pd.DataFrame(default_values, index=instance_ids)

        for feat in self.feature_set.target_features:
            # only update type of pre_existing columns
            if feat.get_name() not in default_df.columns:
                continue
            if feat.variable_type == pdtypes.CategoricalDtype():
                values = []
                for x in instance_ids:
                    values.append(feat.default_value)
                default_df[feat.get_name()] = pd.Series(values, dtype="category")
            else:
                default_df[feat.get_name()] = default_df[feat.get_name()].astype(
                    feat.variable_type,
                )

        return default_df

    def _feature_type_handler(self, f):
        # Order matters: DirectFeature is a subclass of TransformFeature
        if isinstance(f, IdentityFeature):
            return self._calculate_identity_features
        elif isinstance(f, DirectFeature):
            return self._calculate_direct_features
        elif isinstance(f, TransformFeature):
            return self._calculate_transform_features
        elif isinstance(f, GroupByTransformFeature):
            return self._calculate_groupby_features
        elif isinstance(f, AggregationFeature):
            return self._calculate_agg_features
        raise UnknownFeature("{} is not a recognized feature type".format(f))

    def _calculate_identity_features(self, features, df, _df_trie, progress_callback):
        for f in features:
            df[f.get_name()] = f.column

        progress_callback(len(features) / float(self.num_features))
        return df

    def _calculate_transform_features(
        self,
        features,
        frame,
        _df_trie,
        progress_callback,
    ):
        frame_empty = frame.empty
        feature_values = []
        for f in features:
            # handle when no data
            if frame_empty:
                # Even though we are adding the default values here, when these new
                # features are added to the dataframe in update_feature_columns, they
                # are added as empty columns since the dataframe itself is empty.
                feature_values.append(pd.Series([], dtype=f.variable_type))
                continue

            # If primitive uses a time index, set values to NaN where time index is missing
            if f.primitive.uses_calc_time and frame[f.base_dataframe.ww.time_index].isna().any():
                original_series = f.column.loc[frame.index]
                # For each feature, create a new series with NaNs at missing time indices
                nan_mask = frame[f.base_dataframe.ww.time_index].isna()
                if not nan_mask.any():
                    new_series = original_series
                else:
                    # Calculate feature values for non-missing time indices
                    calculated_values = f.primitive.get_function()(
                        frame.loc[~nan_mask, f.base_dataframe.ww.name]
                    )
                    # Reindex with original index to align NaNs
                    new_series = pd.Series(np.nan, index=original_series.index, dtype=original_series.dtype)
                    new_series.loc[~nan_mask] = calculated_values

                feature_values.append(new_series)
            else:
                feature_values.append(f.primitive.get_function()(f.column.loc[frame.index]))

        updated = update_feature_columns(
            zip(features, feature_values),
            frame,
        )

        progress_callback(len(features) / float(self.num_features))
        return updated

    def _calculate_groupby_features(self, features, frame, _df_trie, progress_callback):
        # set default values to handle the null group
        group_ids = []
        for f in features:
            if frame.empty:
                group_ids.append(pd.Series([], dtype=f.variable_type))
                continue

            feature_values = f.primitive.get_function()(f.column.loc[frame.index])
            group_id = f.groupby.column.loc[frame.index].values
            group_ids.append(
                pd.Series(feature_values, index=frame.index, name=f.get_name()),
            )

            # if a group is all null, this sets its result to null
            if (feature_values.isnull()).all():
                group_ids.append(
                    pd.Series([np.nan for _ in range(len(feature_values))]),
                )

        updated = update_feature_columns(
            zip(features, group_ids),
            frame,
        )

        progress_callback(len(features) / float(self.num_features))
        return updated

    def _calculate_direct_features(
        self,
        features,
        child_df,
        df_trie,
        progress_callback,
    ):
        # need to grab the features from the parent dataframe to attach to the child
        for f in features:
            parent_df = df_trie.get_node(f.relationship_path).value
            parent_col = parent_df[[f.parent_feature.get_name()]]
            parent_col.index.name = f.relationship.parent_column.ww.name

            # need to make sure the column that merges is not the index or a
            # foreign key. If it is, then the column will not be unique, and
            # the merge will fail.
            if f.relationship.parent_column.ww.name == parent_col.index.name:
                parent_col = parent_col.reset_index()

            # The child dataframe may have fewer rows than the parent dataframe.
            # We must only merge in values for rows that exist in the child dataframe.
            # So we grab the values from the parent dataframe's column that have the
            # foreign key in the child dataframe.
            child_col = child_df[[f.relationship.child_column.ww.name]]
            child_col.index.name = f.relationship.child_column.ww.name
            child_col = child_col.merge(
                parent_col,
                left_on=f.relationship.child_column.ww.name,
                right_on=f.relationship.parent_column.ww.name,
                how="left",
            )

            child_df[f.get_name()] = child_col[f.parent_feature.get_name()].values

        progress_callback(len(features) / float(self.num_features))
        return child_df

    def _calculate_agg_features(self, features, frame, df_trie, progress_callback):
        groupby_col_names = []
        for f in features:
            if f.relationship_path not in df_trie:
                # This case is when no data was found for the child dataframe.
                # In that situation, the default value for the feature will be used.
                # Nothing needs to be done here.
                continue
            child_df = df_trie.get_node(f.relationship_path).value
            child_df = child_df.copy()

            # if the Dask series has a name that matches an existing column
            # on the dataframe being added, Woodwork will raise an error
            # so we drop the name before adding
            child_df.ww.name = None

            to_agg = f.base_feature.get_name()
            # deal with multi-output primitives by only grabbing the one column
            if isinstance(to_agg, list):
                to_agg = to_agg[0]

            # the column that connects the child to the parent
            groupby_col = f.relationship_path.second_to_last_dataframe.ww.name
            groupby_col_name = get_relationship_column_id(groupby_col)
            child_df[groupby_col_name] = child_df[
                f.relationship_path.child_column.ww.name
            ].values

            # If the feature has a where clause, filter the child dataframe.
            if f.where is not None:
                child_df = child_df[child_df[f.where.get_name()]]

            if f.primitive.uses_previous:
                # must sort and group so that we can use `pd.Series.expanding`
                child_df = child_df.sort_values(f.base_dataframe.ww.time_index)
                to_merge = child_df.groupby(groupby_col_name).apply(
                    agg_wrapper(f.primitive.get_function(), f.use_previous),
                )
            else:
                to_merge = child_df.groupby(groupby_col_name).agg(f.primitive.get_function(), to_agg)

            to_merge = to_merge.reset_index()

            if isinstance(to_merge, pd.Series):
                to_merge = to_merge.to_frame()

            # if a primitive returns multiple columns, add all to the dataframe
            for col_name in f.get_feature_names():
                if col_name in to_merge.columns:
                    values = to_merge[[groupby_col_name, col_name]]
                else:
                    # if the name doesn't match then it must be because it is a
                    # multi-output primitive and this output has no value
                    # (e.g. NMostCommon when n > number of unique values)
                    values = to_merge[groupby_col_name]
                    values = pd.DataFrame(values)
                    values[col_name] = np.nan
                values = values.set_index(groupby_col_name)
                values = values.reindex(frame.index)
                frame[col_name] = values[col_name].values

        progress_callback(len(features) / float(self.num_features))
        return frame

    def _necessary_columns(self, dataframe_name, feature_names):
        # We have to keep all index and foreign columns because we don't know what forward
        # relationships will come from this node.
        columns = set()
        dataframe = self.entityset[dataframe_name]
        columns.add(dataframe.ww.index)
        if dataframe.ww.time_index:
            columns.add(dataframe.ww.time_index)
        for col in dataframe.ww.foreign_keys:
            columns.add(col)
        for f in feature_names:
            columns.add(f.base_dataframe.ww.name)
            if f.where is not None:
                columns.add(f.where.base_dataframe.ww.name)

        return list(columns)


def _can_agg(feature):
    # can't agg if there is an agg in the path that doesn't have
    # use_previous set.
    if feature.is_end_node:
        return True
    return False


def agg_wrapper(feats, time_last):
    def wrap(df):
        # if there are no events in the window, return 0
        if df.empty:
            return pd.Series([0 for _ in range(len(feats.get_feature_names()))])
        # if there is only one event, the output of the primitive could be a scalar
        # or a series. If it is a scalar, we must return a series with one value.
        # If it is a series with one value, we must make sure the name is None
        # so that it does not conflict with the feature names.
        result = feats(df)
        if isinstance(result, pd.Series):
            if result.empty:
                return pd.Series([0 for _ in range(len(feats.get_feature_names()))])
            result.name = None
            return result
        return pd.Series([result])

    return wrap


def update_feature_columns(feature_data, data):
    for f, feature_values in feature_data:
        name = f.get_name()
        if name not in data.columns:
            if isinstance(feature_values, pd.Series):
                feature_values = feature_values.to_frame(name=name)
            data[name] = feature_values
        else:
            data[name].update(feature_values)
    return data


def strip_values_if_series(values):
    if isinstance(values, pd.Series):
        values = values.values
    return values
