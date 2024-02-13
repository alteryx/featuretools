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
from featuretools.utils.gen_utils import (
    Library,
    get_relationship_column_id,
    import_or_none,
    is_instance,
)

dd = import_or_none("dask.dataframe")
ps = import_or_none("pyspark.pandas")


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

        # Fill in empty rows with default values. This only works for pandas dataframes
        # and is not currently supported for Dask dataframes.
        if isinstance(df, pd.DataFrame):
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

        if is_instance(df, (dd, ps), "DataFrame"):
            column_list.extend([target_dataframe.ww.index])

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

            if ancestor_relationship_columns:
                (
                    df,
                    new_ancestor_relationship_columns,
                ) = self._add_ancestor_relationship_columns(
                    df,
                    parent_df,
                    ancestor_relationship_columns,
                    parent_relationship,
                )

            # Add the column linking this dataframe to its parent, so that
            # descendants get linked to the parent.
            new_ancestor_relationship_columns.append(
                parent_relationship._child_column_name,
            )

        # call to update timer
        progress_callback(0)

        # Step 3: Recurse on children.

        # Pass filtered values, even if we are using a full df.
        if need_full_dataframe:
            if is_instance(filter_values, dd, "Series"):
                msg = "Cannot use primitives that require full dataframe with Dask EntitySets"
                raise ValueError(msg)
            filtered_df = df[df[filter_column].isin(filter_values)]
        else:
            filtered_df = df

        for edge, sub_trie in feature_trie.children():
            is_forward, relationship = edge
            if is_forward:
                sub_dataframe_name = relationship.parent_dataframe.ww.name
                sub_filter_column = relationship._parent_column_name
                sub_filter_values = filtered_df[relationship._child_column_name]
                parent_data = None
            else:
                sub_dataframe_name = relationship.child_dataframe.ww.name
                sub_filter_column = relationship._child_column_name
                sub_filter_values = filtered_df[relationship._parent_column_name]

                parent_data = (relationship, new_ancestor_relationship_columns, df)

            sub_df_trie = df_trie.get_node([edge])
            sub_full_dataframe_trie = full_dataframe_trie.get_node([edge])
            sub_precalc_trie = precalculated_trie.get_node([edge])
            self._calculate_features_for_dataframe(
                dataframe_name=sub_dataframe_name,
                feature_trie=sub_trie,
                df_trie=sub_df_trie,
                full_dataframe_trie=sub_full_dataframe_trie,
                precalculated_trie=sub_precalc_trie,
                filter_column=sub_filter_column,
                filter_values=sub_filter_values,
                parent_data=parent_data,
                progress_callback=progress_callback,
                include_cutoff_time=include_cutoff_time,
            )

        # Step 4: Calculate the features for this dataframe.
        #
        # All dependencies of the features for this dataframe have been calculated
        # by the above recursive calls, and their results stored in df_trie.

        # Add any precalculated features.
        precalculated_features_df = precalculated_trie.value
        if precalculated_features_df is not None:
            # Left outer merge to keep all rows of df.
            df = df.merge(
                precalculated_features_df,
                how="left",
                left_index=True,
                right_index=True,
                suffixes=("", "_precalculated"),
            )

        # call to update timer
        progress_callback(0)

        # First, calculate any features that require the full dataframe. These can
        # be calculated first because all of their dependents are included in
        # full_dataframe_features.
        if need_full_dataframe:
            df = self._calculate_features(
                df,
                full_dataframe_trie,
                full_dataframe_features,
                progress_callback,
            )

            # Store full dataframe
            full_dataframe_trie.value = df

            # Filter df so that features that don't require the full dataframe are
            # only calculated on the necessary instances.
            df = df[df[filter_column].isin(filter_values)]

        # Calculate all features that don't require the full dataframe.
        df = self._calculate_features(
            df,
            df_trie,
            not_full_dataframe_features,
            progress_callback,
        )

        # Step 5: Store the dataframe for this dataframe at the root of df_trie, so
        # that it can be accessed by the caller.
        df_trie.value = df

    def _calculate_features(self, df, df_trie, features, progress_callback):
        # Group the features so that each group can be calculated together.
        # The groups must also be in topological order (if A is a transform of B
        # then B must be in a group before A).
        feature_groups = self.feature_set.group_features(features)

        for group in feature_groups:
            representative_feature = group[0]
            handler = self._feature_type_handler(representative_feature)
            df = handler(group, df, df_trie, progress_callback)

        return df

    def _add_ancestor_relationship_columns(
        self,
        child_df,
        parent_df,
        ancestor_relationship_columns,
        relationship,
    ):
        """
        Merge ancestor_relationship_columns from parent_df into child_df, adding a prefix to
        each column name specifying the relationship.

        Return the updated df and the new relationship column names.

        Args:
            child_df (pd.DataFrame): The dataframe to add relationship columns to.
            parent_df (pd.DataFrame): The dataframe to copy relationship columns from.
            ancestor_relationship_columns (list[str]): The names of
                relationship columns in the parent_df to copy into child_df.
            relationship (Relationship): the relationship through which the
                child is connected to the parent.
        """
        relationship_name = relationship.parent_name
        new_relationship_columns = [
            "%s.%s" % (relationship_name, col) for col in ancestor_relationship_columns
        ]

        # create an intermediate dataframe which shares a column
        # with the child dataframe and has a column with the
        # original parent's id.
        col_map = {relationship._parent_column_name: relationship._child_column_name}
        for child_column, parent_column in zip(
            new_relationship_columns,
            ancestor_relationship_columns,
        ):
            col_map[parent_column] = child_column

        merge_df = parent_df[list(col_map.keys())].rename(columns=col_map)

        merge_df.index.name = None  # change index name for merge

        # Merge the dataframe, adding the relationship columns to the child.
        # Left outer join so that all rows in child are kept (if it contains
        # all rows of the dataframe then there may not be corresponding rows in the
        # parent_df).
        df = child_df.merge(
            merge_df,
            how="left",
            left_on=relationship._child_column_name,
            right_on=relationship._child_column_name,
        )

        # ensure index is maintained
        # TODO: Review for dask dataframes
        if isinstance(df, pd.DataFrame):
            df.set_index(
                relationship.child_dataframe.ww.index,
                drop=False,
                inplace=True,
            )

        return df, new_relationship_columns

    def generate_default_df(self, instance_ids, extra_columns=None):
        default_row = []
        default_cols = []
        for f in self.feature_set.target_features:
            for name in f.get_feature_names():
                default_cols.append(name)
                default_row.append(f.default_value)

        default_matrix = [default_row] * len(instance_ids)
        default_df = pd.DataFrame(
            default_matrix,
            columns=default_cols,
            index=instance_ids,
            dtype="object",
        )
        index_name = self.entityset[self.feature_set.target_df_name].ww.index
        default_df.index.name = index_name
        if extra_columns is not None:
            for c in extra_columns:
                if c not in default_df.columns:
                    default_df[c] = [np.nan] * len(instance_ids)
        return default_df

    def _feature_type_handler(self, f):
        if type(f) == TransformFeature:
            return self._calculate_transform_features
        elif type(f) == GroupByTransformFeature:
            return self._calculate_groupby_features
        elif type(f) == DirectFeature:
            return self._calculate_direct_features
        elif type(f) == AggregationFeature:
            return self._calculate_agg_features
        elif type(f) == IdentityFeature:
            return self._calculate_identity_features
        else:
            raise UnknownFeature("{} feature unknown".format(f.__class__))

    def _calculate_identity_features(self, features, df, _df_trie, progress_callback):
        for f in features:
            assert f.get_name() in df.columns, (
                'Column "%s" missing frome dataframe' % f.get_name()
            )

        progress_callback(len(features) / float(self.num_features))

        return df

    def _calculate_transform_features(
        self,
        features,
        frame,
        _df_trie,
        progress_callback,
    ):
        frame_empty = frame.empty if isinstance(frame, pd.DataFrame) else False
        feature_values = []
        for f in features:
            # handle when no data
            if frame_empty:
                # Even though we are adding the default values here, when these new
                # features are added to the dataframe in update_feature_columns, they
                # are added as empty columns since the dataframe itself is empty.
                feature_values.append(
                    (f, [f.default_value for _ in range(f.number_output_features)]),
                )
                progress_callback(1 / float(self.num_features))
                continue

            # collect only the columns we need for this transformation

            column_data = [frame[bf.get_name()] for bf in f.base_features]

            feature_func = f.get_function()
            # apply the function to the relevant dataframe slice and add the
            # feature row to the results dataframe.
            if f.primitive.uses_calc_time:
                values = feature_func(*column_data, time=self.time_last)
            else:
                values = feature_func(*column_data)

            # if we don't get just the values, the assignment breaks when indexes don't match
            if f.number_output_features > 1:
                values = [strip_values_if_series(value) for value in values]
            else:
                values = [strip_values_if_series(values)]

            feature_values.append((f, values))

            progress_callback(1 / float(self.num_features))

        frame = update_feature_columns(feature_values, frame)
        return frame

    def _calculate_groupby_features(self, features, frame, _df_trie, progress_callback):
        # set default values to handle the null group
        default_values = {}
        for f in features:
            for name in f.get_feature_names():
                default_values[name] = f.default_value

        frame = pd.concat(
            [frame, pd.DataFrame(default_values, index=frame.index)],
            axis=1,
        )

        # handle when no data
        if frame.shape[0] == 0:
            progress_callback(len(features) / float(self.num_features))

            return frame

        groupby = features[0].groupby.get_name()
        grouped = frame.groupby(groupby)
        groups = frame[
            groupby
        ].unique()  # get all the unique group name to iterate over later

        for f in features:
            feature_vals = []
            for _ in range(f.number_output_features):
                feature_vals.append([])

            for group in groups:
                # skip null key if it exists
                if pd.isnull(group):
                    continue

                column_names = [bf.get_name() for bf in f.base_features]
                # exclude the groupby column from being passed to the function
                column_data = [
                    grouped[name].get_group(group) for name in column_names[:-1]
                ]
                feature_func = f.get_function()

                # apply the function to the relevant dataframe slice and add the
                # feature row to the results dataframe.
                if f.primitive.uses_calc_time:
                    values = feature_func(*column_data, time=self.time_last)
                else:
                    values = feature_func(*column_data)

                if f.number_output_features == 1:
                    values = [values]

                # make sure index is aligned
                for i, value in enumerate(values):
                    if isinstance(value, pd.Series):
                        value.index = column_data[0].index
                    else:
                        value = pd.Series(value, index=column_data[0].index)
                    feature_vals[i].append(value)

            if any(feature_vals):
                assert len(feature_vals) == len(f.get_feature_names())
                for col_vals, name in zip(feature_vals, f.get_feature_names()):
                    frame[name].update(pd.concat(col_vals))

            progress_callback(1 / float(self.num_features))

        return frame

    def _calculate_direct_features(
        self,
        features,
        child_df,
        df_trie,
        progress_callback,
    ):
        path = features[0].relationship_path
        assert len(path) == 1, "Error calculating DirectFeatures, len(path) != 1"

        parent_df = df_trie.get_node([path[0]]).value
        _is_forward, relationship = path[0]
        merge_col = relationship._child_column_name

        # generate a mapping of old column names (in the parent dataframe) to
        # new column names (in the child dataframe) for the merge
        col_map = {relationship._parent_column_name: merge_col}
        index_as_feature = None

        fillna_dict = {}
        for f in features:
            feature_defaults = {
                name: f.default_value
                for name in f.get_feature_names()
                if not pd.isna(f.default_value)
            }
            fillna_dict.update(feature_defaults)
            if f.base_features[0].get_name() == relationship._parent_column_name:
                index_as_feature = f
            base_names = f.base_features[0].get_feature_names()
            for name, base_name in zip(f.get_feature_names(), base_names):
                if name in child_df.columns:
                    continue
                col_map[base_name] = name

        # merge the identity feature from the parent dataframe into the child
        merge_df = parent_df[list(col_map.keys())].rename(columns=col_map)
        if is_instance(merge_df, (dd, ps), "DataFrame"):
            new_df = child_df.merge(
                merge_df,
                left_on=merge_col,
                right_on=merge_col,
                how="left",
            )
        else:
            if index_as_feature is not None:
                merge_df.set_index(
                    index_as_feature.get_name(),
                    inplace=True,
                    drop=False,
                )
            else:
                merge_df.set_index(merge_col, inplace=True)

            new_df = child_df.merge(
                merge_df,
                left_on=merge_col,
                right_index=True,
                how="left",
            )

        progress_callback(len(features) / float(self.num_features))

        return new_df.fillna(fillna_dict)

    def _calculate_agg_features(self, features, frame, df_trie, progress_callback):
        test_feature = features[0]
        child_dataframe = test_feature.base_features[0].dataframe
        base_frame = df_trie.get_node(test_feature.relationship_path).value
        parent_merge_col = test_feature.relationship_path[0][1]._parent_column_name
        # Sometimes approximate features get computed in a previous filter frame
        # and put in the current one dynamically,
        # so there may be existing features here
        fl = []
        for f in features:
            for ind in f.get_feature_names():
                if ind not in frame.columns:
                    fl.append(f)
                    break
        features = fl
        if not len(features):
            progress_callback(len(features) / float(self.num_features))
            return frame

        # handle where
        base_frame_empty = (
            base_frame.empty if isinstance(base_frame, pd.DataFrame) else False
        )
        where = test_feature.where
        if where is not None and not base_frame_empty:
            base_frame = base_frame.loc[base_frame[where.get_name()]]

        # when no child data, just add all the features to frame with nan
        base_frame_empty = (
            base_frame.empty if isinstance(base_frame, pd.DataFrame) else False
        )
        if base_frame_empty:
            feature_values = []
            for f in features:
                feature_values.append((f, np.full(f.number_output_features, np.nan)))
                progress_callback(1 / float(self.num_features))
            frame = update_feature_columns(feature_values, frame)
        else:
            relationship_path = test_feature.relationship_path

            groupby_col = get_relationship_column_id(relationship_path)

            # if the use_previous property exists on this feature, include only the
            # instances from the child dataframe included in that Timedelta
            use_previous = test_feature.use_previous
            if use_previous:
                # Filter by use_previous values
                time_last = self.time_last
                if use_previous.has_no_observations():
                    time_first = time_last - use_previous
                    ti = child_dataframe.ww.time_index
                    if ti is not None:
                        base_frame = base_frame[base_frame[ti] >= time_first]
                else:
                    n = use_previous.get_value("o")

                    def last_n(df):
                        return df.iloc[-n:]

                    base_frame = base_frame.groupby(
                        groupby_col,
                        observed=True,
                        sort=False,
                        group_keys=False,
                    ).apply(last_n)

            to_agg = {}
            agg_rename = {}
            to_apply = set()
            # apply multi-column and time-dependent features as we find them, and
            # save aggregable features for later
            for f in features:
                if _can_agg(f):
                    column_id = f.base_features[0].get_name()
                    if column_id not in to_agg:
                        to_agg[column_id] = []
                    if is_instance(base_frame, dd, "DataFrame"):
                        func = f.get_function(agg_type=Library.DASK)
                    elif is_instance(base_frame, ps, "DataFrame"):
                        func = f.get_function(agg_type=Library.SPARK)
                    else:
                        func = f.get_function()

                    # for some reason, using the string count is significantly
                    # faster than any method a primitive can return
                    # https://stackoverflow.com/questions/55731149/use-a-function-instead-of-string-in-pandas-groupby-agg
                    if func == pd.Series.count:
                        func = "count"

                    funcname = func
                    if callable(func):
                        # if the same function is being applied to the same
                        # column twice, wrap it in a partial to avoid
                        # duplicate functions
                        funcname = str(id(func))
                        if "{}-{}".format(column_id, funcname) in agg_rename:
                            func = partial(func)
                            funcname = str(id(func))

                        func.__name__ = funcname

                    if dd and isinstance(func, dd.Aggregation):
                        # TODO: handle aggregation being applied to same column twice
                        # (see above partial wrapping of functions)
                        funcname = func.__name__

                    to_agg[column_id].append(func)
                    # this is used below to rename columns that pandas names for us
                    agg_rename["{}-{}".format(column_id, funcname)] = f.get_name()
                    continue

                to_apply.add(f)

            # Apply the non-aggregable functions generate a new dataframe, and merge
            # it with the existing one
            if len(to_apply):
                wrap = agg_wrapper(to_apply, self.time_last)
                # groupby_col can be both the name of the index and a column,
                # to silence pandas warning about ambiguity we explicitly pass
                # the column (in actuality grouping by both index and group would
                # work)
                to_merge = base_frame.groupby(
                    base_frame[groupby_col],
                    observed=True,
                    sort=False,
                    group_keys=False,
                ).apply(wrap)
                frame = pd.merge(
                    left=frame,
                    right=to_merge,
                    left_index=True,
                    right_index=True,
                    how="left",
                )

                progress_callback(len(to_apply) / float(self.num_features))

            # Apply the aggregate functions to generate a new dataframe, and merge
            # it with the existing one
            if len(to_agg):
                # groupby_col can be both the name of the index and a column,
                # to silence pandas warning about ambiguity we explicitly pass
                # the column (in actuality grouping by both index and group would
                # work)
                if is_instance(base_frame, (dd, ps), "DataFrame"):
                    to_merge = base_frame.groupby(groupby_col).agg(to_agg)
                else:
                    to_merge = base_frame.groupby(
                        base_frame[groupby_col],
                        observed=True,
                        sort=False,
                    ).agg(to_agg)
                # rename columns to the correct feature names
                to_merge.columns = [agg_rename["-".join(x)] for x in to_merge.columns]
                to_merge = to_merge[list(agg_rename.values())]

                # Workaround for pandas bug where categories are in the wrong order
                # see: https://github.com/pandas-dev/pandas/issues/22501
                #
                # Pandas claims that bug is fixed but it still shows up in some
                # cases.  More investigation needed.
                if isinstance(frame.index, pd.CategoricalDtype):
                    categories = pdtypes.CategoricalDtype(
                        categories=frame.index.categories,
                    )
                    to_merge.index = to_merge.index.astype(object).astype(categories)

                if is_instance(frame, (dd, ps), "DataFrame"):
                    frame = frame.merge(
                        to_merge,
                        left_on=parent_merge_col,
                        right_index=True,
                        how="left",
                    )
                else:
                    frame = pd.merge(
                        left=frame,
                        right=to_merge,
                        left_index=True,
                        right_index=True,
                        how="left",
                    )

                # determine number of features that were just merged
                progress_callback(len(to_merge.columns) / float(self.num_features))

        # Handle default values
        fillna_dict = {}
        for f in features:
            feature_defaults = {name: f.default_value for name in f.get_feature_names()}
            fillna_dict.update(feature_defaults)

        frame = frame.fillna(fillna_dict)

        return frame

    def _necessary_columns(self, dataframe_name, feature_names):
        # We have to keep all index and foreign columns because we don't know what forward
        # relationships will come from this node.
        df = self.entityset[dataframe_name]
        index_columns = {
            col
            for col in df.columns
            if {"index", "foreign_key", "time_index"} & df.ww.semantic_tags[col]
        }
        features = (self.feature_set.features_by_name[name] for name in feature_names)

        feature_columns = {
            f.column_name for f in features if isinstance(f, IdentityFeature)
        }
        return list(index_columns | feature_columns)


def _can_agg(feature):
    assert isinstance(feature, AggregationFeature)
    base_features = feature.base_features
    if feature.where is not None:
        base_features = [
            bf.get_name()
            for bf in base_features
            if bf.get_name() != feature.where.get_name()
        ]

    if feature.primitive.uses_calc_time:
        return False
    single_output = feature.primitive.number_output_features == 1
    return len(base_features) == 1 and single_output


def agg_wrapper(feats, time_last):
    def wrap(df):
        d = {}
        feature_values = []
        for f in feats:
            func = f.get_function()
            column_ids = [bf.get_name() for bf in f.base_features]
            args = [df[v] for v in column_ids]

            if f.primitive.uses_calc_time:
                values = func(*args, time=time_last)
            else:
                values = func(*args)

            if f.number_output_features == 1:
                values = [values]
            feature_values.append((f, values))

        d = update_feature_columns(feature_values, d)

        return pd.Series(d)

    return wrap


def update_feature_columns(feature_data, data):
    new_cols = {}
    for item in feature_data:
        names = item[0].get_feature_names()
        values = item[1]
        assert len(names) == len(values)
        for name, value in zip(names, values):
            new_cols[name] = value

    # Handle the case where a dict is being updated
    if isinstance(data, dict):
        data.update(new_cols)
        return data

    # Handle pandas input
    if isinstance(data, pd.DataFrame):
        return pd.concat([data, pd.DataFrame(new_cols, index=data.index)], axis=1)

    # Handle dask/spark input
    for name, col in new_cols.items():
        col.name = name
        if is_instance(data, dd, "DataFrame"):
            data = dd.concat([data, col], axis=1)
        else:
            data = ps.concat([data, col], axis=1)
    return data


def strip_values_if_series(values):
    if isinstance(values, pd.Series):
        values = values.values
    return values
