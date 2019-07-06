import warnings
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import pandas.api.types as pdtypes

from featuretools import variable_types
from featuretools.entityset.relationship import RelationshipPath
from featuretools.exceptions import UnknownFeature
from featuretools.feature_base import (
    AggregationFeature,
    DirectFeature,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature
)
from featuretools.utils import Trie, is_python_2
from featuretools.utils.gen_utils import get_relationship_variable_id

warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)


class FeatureSetCalculator(object):
    """
    Calculates the values of a set of features for given instance ids.
    """

    def __init__(self, entityset, feature_set, time_last=None,
                 training_window=None, precalculated_features=None, ignored=None):
        """
        Args:
            feature_set (FeatureSet): The features to calculate values for.

            time_last (pd.Timestamp, optional): Last allowed time. Data from exactly this
                time not allowed.

            training_window (Timedelta, optional): Window defining how much time before the cutoff time data
                can be used when calculating features. If None, all data before cutoff time is used.

            precalculated_features (Trie[RelationshipPath -> pd.DataFrame]):
                Maps RelationshipPaths to dataframes of precalculated_features

            ignored (set[str], optional): Unique names of precalculated features.
        """
        self.entityset = entityset
        self.feature_set = feature_set
        self.training_window = training_window
        self.ignored = ignored

        if time_last is None:
            time_last = datetime.now()

        self.time_last = time_last

        if precalculated_features is None:
            precalculated_features = Trie(path_constructor=RelationshipPath)

        self.precalculated_features = precalculated_features

    def run(self, instance_ids):
        """
        Calculate values of features for the given instances of the target
        entity.

        Summary of algorithm:
        1. Construct a trie where the edges are relationships and each node
            contains a set of features for a single entity. See
            FeatureSet._build_feature_trie.
        2. Initialize a trie for storing dataframes.
        3. Traverse the trie using depth first search. At each node calculate
            the features and store the resulting dataframe in the dataframe
            trie (so that its values can be used by features which depend on
            these features). See _calculate_features_for_entity.
        4. Get the dataframe at the root of the trie (for the target entity) and
            return the columns corresponding to the requested features.

        Args:
            instance_ids (np.ndarray or pd.Categorical): Instance ids for which
                to build features.

        Returns:
            pd.DataFrame : Pandas DataFrame of calculated feature values.
                Indexed by instance_ids. Columns in same order as features
                passed in.
        """
        assert len(instance_ids) > 0, "0 instance ids provided"

        feature_trie = self.feature_set.feature_trie

        df_trie = Trie(path_constructor=RelationshipPath)
        full_entity_df_trie = Trie(path_constructor=RelationshipPath)

        target_entity = self.entityset[self.feature_set.target_eid]
        self._calculate_features_for_entity(entity_id=self.feature_set.target_eid,
                                            feature_trie=feature_trie,
                                            df_trie=df_trie,
                                            full_entity_df_trie=full_entity_df_trie,
                                            precalculated_trie=self.precalculated_features,
                                            filter_variable=target_entity.index,
                                            filter_values=instance_ids)

        # The dataframe for the target entity should be stored at the root of
        # df_trie.
        df = df_trie.value

        if df.empty:
            return self.generate_default_df(instance_ids=instance_ids)

        # fill in empty rows with default values
        missing_ids = [i for i in instance_ids if i not in
                       df[target_entity.index]]
        if missing_ids:
            default_df = self.generate_default_df(instance_ids=missing_ids,
                                                  extra_columns=df.columns)
            df = df.append(default_df, sort=True)

        df.index.name = self.entityset[self.feature_set.target_eid].index
        column_list = []

        # Order by instance_ids
        unique_instance_ids = pd.unique(instance_ids)
        # pd.unique changes the dtype for Categorical, so reset it.
        unique_instance_ids = unique_instance_ids.astype(instance_ids.dtype)
        df = df.reindex(unique_instance_ids)

        for feat in self.feature_set.target_features:
            column_list.extend(feat.get_feature_names())
        return df[column_list]

    def _calculate_features_for_entity(self, entity_id, feature_trie, df_trie,
                                       full_entity_df_trie,
                                       precalculated_trie,
                                       filter_variable, filter_values,
                                       parent_data=None):
        """
        Generate dataframes with features calculated for this node of the trie,
        and all descendant nodes. The dataframes will be stored in df_trie.

        Args:
            entity_id (str): The name of the entity to calculate features for.

            feature_trie (Trie): the trie with sets of features to calculate.
                The root contains features for the given entity.

            df_trie (Trie): a parallel trie for storing dataframes. The
                dataframe with features calculated will be placed in the root.

            full_entity_df_trie (Trie): a trie storing dataframes will all entity
                rows, for features that are uses_full_entity.

            precalculated_trie (Trie): a parallel trie containing dataframes
                with precalculated features. The dataframe for this entity will
                be at the root.

            filter_variable (str): The name of the variable to filter this
                dataframe by.

            filter_values (pd.Series): The values to filter the filter_variable
                to.

            parent_data (tuple[Relationship, list[str], pd.DataFrame]): Data
                related to the parent of this trie. This will only be present if
                the relationship points from this entity to the parent entity. A
                3 tuple of (parent_relationship,
                ancestor_relationship_variables, parent_df).
                ancestor_relationship_variables is the names of variables which
                link the parent entity to its ancestors.
        """

        # Step 1: Get a dataframe for the given entity, filtered by the given
        # conditions.

        need_full_entity, full_entity_features, not_full_entity_features = feature_trie.value
        if self.ignored:
            full_entity_features -= self.ignored
            not_full_entity_features -= self.ignored

        all_features = full_entity_features | not_full_entity_features
        entity = self.entityset[entity_id]
        columns = self._necessary_columns(entity, all_features)

        # If we need the full entity then don't filter by filter_values.
        if need_full_entity:
            query_variable = None
            query_values = None
        else:
            query_variable = filter_variable
            query_values = filter_values

        df = entity.query_by_values(query_values,
                                    variable_id=query_variable,
                                    columns=columns,
                                    time_last=self.time_last,
                                    training_window=self.training_window)

        # Step 2: Add variables to the dataframe linking it to all ancestors.

        new_ancestor_relationship_variables = []
        if parent_data:
            parent_relationship, ancestor_relationship_variables, parent_df = \
                parent_data

            if ancestor_relationship_variables:
                df, new_ancestor_relationship_variables = self._add_ancestor_relationship_variables(
                    df, parent_df, ancestor_relationship_variables, parent_relationship)

            # Add the variable linking this entity to its parent, so that
            # descendants get linked to the parent.
            new_ancestor_relationship_variables.append(parent_relationship.child_variable.id)

        # Step 3: Recurse on children.

        # Pass filtered values, even if we are using a full df.
        if need_full_entity:
            filtered_df = df[df[filter_variable].isin(filter_values)]
        else:
            filtered_df = df

        for edge, sub_trie in feature_trie.children():
            is_forward, relationship = edge
            if is_forward:
                sub_entity = relationship.parent_entity.id
                sub_filter_variable = relationship.parent_variable.id
                sub_filter_values = filtered_df[relationship.child_variable.id]
                parent_data = None
            else:
                sub_entity = relationship.child_entity.id
                sub_filter_variable = relationship.child_variable.id
                sub_filter_values = filtered_df[relationship.parent_variable.id]

                parent_data = (relationship,
                               new_ancestor_relationship_variables,
                               df)

            sub_df_trie = df_trie.get_node([edge])
            sub_full_entity_df_trie = full_entity_df_trie.get_node([edge])
            sub_precalc_trie = precalculated_trie.get_node([edge])
            self._calculate_features_for_entity(
                entity_id=sub_entity,
                feature_trie=sub_trie,
                df_trie=sub_df_trie,
                full_entity_df_trie=sub_full_entity_df_trie,
                precalculated_trie=sub_precalc_trie,
                filter_variable=sub_filter_variable,
                filter_values=sub_filter_values,
                parent_data=parent_data)

        # Step 4: Calculate the features for this entity.
        #
        # All dependencies of the features for this entity have been calculated
        # by the above recursive calls, and their results stored in df_trie.

        # Add any precalculated features.
        precalculated_features_df = precalculated_trie.value
        if precalculated_features_df is not None:
            # Left outer merge to keep all rows of df.
            df = df.merge(precalculated_features_df,
                          how='left',
                          left_index=True,
                          right_index=True,
                          suffixes=('', '_precalculated'))

        # First, calculate any features that require the full entity. These can
        # be calculated first because all of their dependents are included in
        # full_entity_features.
        if need_full_entity:
            df = self._calculate_features(df, full_entity_df_trie, full_entity_features)

            # Store full entity df.
            full_entity_df_trie.value = df

            # Filter df so that features that don't require the full entity are
            # only calculated on the necessary instances.
            df = df[df[filter_variable].isin(filter_values)]

        # Calculate all features that don't require the full entity.
        df = self._calculate_features(df, df_trie, not_full_entity_features)

        # Step 5: Store the dataframe for this entity at the root of df_trie, so
        # that it can be accessed by the caller.
        df_trie.value = df

    def _calculate_features(self, df, df_trie, features):
        # Group the features so that each group can be calculated together.
        # The groups must also be in topological order (if A is a transform of B
        # then B must be in a group before A).
        feature_groups = self.feature_set.group_features(features)

        for group in feature_groups:
            representative_feature = group[0]
            handler = self._feature_type_handler(representative_feature)
            df = handler(group, df, df_trie)

        return df

    def _add_ancestor_relationship_variables(self, child_df, parent_df,
                                             ancestor_relationship_variables,
                                             relationship):
        """
        Merge ancestor_relationship_variables from parent_df into child_df, adding a prefix to
        each column name specifying the relationship.

        Return the updated df and the new relationship variable names.

        Args:
            child_df (pd.DataFrame): The dataframe to add relationship variables to.
            parent_df (pd.DataFrame): The dataframe to copy relationship variables from.
            ancestor_relationship_variables (list[str]): The names of
                relationship variables in the parent_df to copy into child_df.
            relationship (Relationship): the relationship through which the
                child is connected to the parent.
        """
        relationship_name = relationship.parent_name
        new_relationship_variables = ['%s.%s' % (relationship_name, var)
                                      for var in ancestor_relationship_variables]

        # create an intermediate dataframe which shares a column
        # with the child dataframe and has a column with the
        # original parent's id.
        col_map = {relationship.parent_variable.id: relationship.child_variable.id}
        for child_var, parent_var in zip(new_relationship_variables, ancestor_relationship_variables):
            col_map[parent_var] = child_var

        merge_df = parent_df[list(col_map.keys())].rename(columns=col_map)

        merge_df.index.name = None  # change index name for merge

        # Merge the dataframe, adding the relationship variables to the child.
        # Left outer join so that all rows in child are kept (if it contains
        # all rows of the entity then there may not be corresponding rows in the
        # parent_df).
        df = child_df.merge(merge_df,
                            how='left',
                            left_on=relationship.child_variable.id,
                            right_on=relationship.child_variable.id)

        return df, new_relationship_variables

    def generate_default_df(self, instance_ids, extra_columns=None):
        default_row = []
        default_cols = []
        for f in self.feature_set.target_features:
            for name in f.get_feature_names():
                default_cols.append(name)
                default_row.append(f.default_value)

        default_matrix = [default_row] * len(instance_ids)
        default_df = pd.DataFrame(default_matrix,
                                  columns=default_cols,
                                  index=instance_ids)
        index_name = self.entityset[self.feature_set.target_eid].index
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
            raise UnknownFeature(u"{} feature unknown".format(f.__class__))

    def _calculate_identity_features(self, features, df, _df_trie):
        for f in features:
            assert f.get_name() in df, (
                'Column "%s" missing frome dataframe' % f.get_name())

        return df

    def _calculate_transform_features(self, features, frame, _df_trie):
        for f in features:
            # handle when no data
            if frame.shape[0] == 0:
                set_default_column(frame, f)
                continue

            # collect only the variables we need for this transformation
            variable_data = [frame[bf.get_name()]
                             for bf in f.base_features]

            feature_func = f.get_function()
            # apply the function to the relevant dataframe slice and add the
            # feature row to the results dataframe.
            if f.primitive.uses_calc_time:
                values = feature_func(*variable_data, time=self.time_last)
            else:
                values = feature_func(*variable_data)

            # if we don't get just the values, the assignment breaks when indexes don't match
            if f.number_output_features > 1:
                values = [strip_values_if_series(value) for value in values]
            else:
                values = [strip_values_if_series(values)]
            update_feature_columns(f, frame, values)

        return frame

    def _calculate_groupby_features(self, features, frame, _df_trie):
        for f in features:
            set_default_column(frame, f)

        # handle when no data
        if frame.shape[0] == 0:
            return frame

        groupby = features[0].groupby.get_name()
        grouped = frame.groupby(groupby)
        groups = frame[groupby].unique()  # get all the unique group name to iterate over later

        for f in features:
            feature_vals = []
            for group in groups:
                # skip null key if it exists
                if pd.isnull(group):
                    continue

                column_names = [bf.get_name() for bf in f.base_features]
                # exclude the groupby variable from being passed to the function
                variable_data = [grouped[name].get_group(group) for name in column_names[:-1]]
                feature_func = f.get_function()

                # apply the function to the relevant dataframe slice and add the
                # feature row to the results dataframe.
                if f.primitive.uses_calc_time:
                    values = feature_func(*variable_data, time=self.time_last)
                else:
                    values = feature_func(*variable_data)

                # make sure index is aligned
                if isinstance(values, pd.Series):
                    values.index = variable_data[0].index
                else:
                    values = pd.Series(values, index=variable_data[0].index)

                feature_vals.append(values)

            # Note
            # more efficient in pandas to concat and update only once
            if feature_vals:
                frame[f.get_name()].update(pd.concat(feature_vals))

        return frame

    def _calculate_direct_features(self, features, child_df, df_trie):
        path = features[0].relationship_path
        assert len(path) == 1, \
            "Error calculating DirectFeatures, len(path) != 1"

        parent_df = df_trie.get_node([path[0]]).value
        _is_forward, relationship = path[0]
        merge_var = relationship.child_variable.id

        # generate a mapping of old column names (in the parent entity) to
        # new column names (in the child entity) for the merge
        col_map = {relationship.parent_variable.id: merge_var}
        index_as_feature = None
        for f in features:
            if f.base_features[0].get_name() == relationship.parent_variable.id:
                index_as_feature = f
            base_names = f.base_features[0].get_feature_names()
            for name, base_name in zip(f.get_feature_names(), base_names):
                if name in child_df.columns:
                    continue
                col_map[base_name] = name

        # merge the identity feature from the parent entity into the child
        merge_df = parent_df[list(col_map.keys())].rename(columns=col_map)
        if index_as_feature is not None:
            merge_df.set_index(index_as_feature.get_name(), inplace=True,
                               drop=False)
        else:
            merge_df.set_index(merge_var, inplace=True)

        new_df = child_df.merge(merge_df, left_on=merge_var, right_index=True,
                                how='left')

        return new_df

    def _calculate_agg_features(self, features, frame, df_trie):
        test_feature = features[0]
        child_entity = test_feature.base_features[0].entity

        base_frame = df_trie.get_node(test_feature.relationship_path).value
        # Sometimes approximate features get computed in a previous filter frame
        # and put in the current one dynamically,
        # so there may be existing features here
        features = [f for f in features if f.get_name()
                    not in frame.columns]
        if not len(features):
            return frame

        # handle where
        where = test_feature.where
        if where is not None and not base_frame.empty:
            base_frame = base_frame.loc[base_frame[where.get_name()]]

        # when no child data, just add all the features to frame with nan
        if base_frame.empty:
            for f in features:
                frame[f.get_name()] = np.nan
        else:
            relationship_path = test_feature.relationship_path

            groupby_var = get_relationship_variable_id(relationship_path)

            # if the use_previous property exists on this feature, include only the
            # instances from the child entity included in that Timedelta
            use_previous = test_feature.use_previous
            if use_previous and not base_frame.empty:
                # Filter by use_previous values
                time_last = self.time_last
                if use_previous.is_absolute():
                    time_first = time_last - use_previous
                    ti = child_entity.time_index
                    if ti is not None:
                        base_frame = base_frame[base_frame[ti] >= time_first]
                else:
                    n = use_previous.value

                    def last_n(df):
                        return df.iloc[-n:]

                    base_frame = base_frame.groupby(groupby_var, observed=True, sort=False).apply(last_n)

            to_agg = {}
            agg_rename = {}
            to_apply = set()
            # apply multivariable and time-dependent features as we find them, and
            # save aggregable features for later
            for f in features:
                if _can_agg(f):
                    variable_id = f.base_features[0].get_name()

                    if variable_id not in to_agg:
                        to_agg[variable_id] = []

                    func = f.get_function()

                    # for some reason, using the string count is significantly
                    # faster than any method a primitive can return
                    # https://stackoverflow.com/questions/55731149/use-a-function-instead-of-string-in-pandas-groupby-agg
                    if is_python_2() and func == pd.Series.count.__func__:
                        func = "count"
                    elif func == pd.Series.count:
                        func = "count"

                    funcname = func
                    if callable(func):
                        # if the same function is being applied to the same
                        # variable twice, wrap it in a partial to avoid
                        # duplicate functions
                        funcname = str(id(func))
                        if u"{}-{}".format(variable_id, funcname) in agg_rename:
                            func = partial(func)
                            funcname = str(id(func))

                        func.__name__ = funcname

                    to_agg[variable_id].append(func)
                    # this is used below to rename columns that pandas names for us
                    agg_rename[u"{}-{}".format(variable_id, funcname)] = f.get_name()
                    continue

                to_apply.add(f)

            # Apply the non-aggregable functions generate a new dataframe, and merge
            # it with the existing one
            if len(to_apply):
                wrap = agg_wrapper(to_apply, self.time_last)
                # groupby_var can be both the name of the index and a column,
                # to silence pandas warning about ambiguity we explicitly pass
                # the column (in actuality grouping by both index and group would
                # work)
                to_merge = base_frame.groupby(base_frame[groupby_var], observed=True, sort=False).apply(wrap)
                frame = pd.merge(left=frame, right=to_merge,
                                 left_index=True,
                                 right_index=True, how='left')

            # Apply the aggregate functions to generate a new dataframe, and merge
            # it with the existing one
            if len(to_agg):
                # groupby_var can be both the name of the index and a column,
                # to silence pandas warning about ambiguity we explicitly pass
                # the column (in actuality grouping by both index and group would
                # work)
                to_merge = base_frame.groupby(base_frame[groupby_var],
                                              observed=True, sort=False).agg(to_agg)
                # rename columns to the correct feature names
                to_merge.columns = [agg_rename["-".join(x)] for x in to_merge.columns.ravel()]
                to_merge = to_merge[list(agg_rename.values())]

                # workaround for pandas bug where categories are in the wrong order
                # see: https://github.com/pandas-dev/pandas/issues/22501
                if pdtypes.is_categorical_dtype(frame.index):
                    categories = pdtypes.CategoricalDtype(categories=frame.index.categories)
                    to_merge.index = to_merge.index.astype(object).astype(categories)

                frame = pd.merge(left=frame, right=to_merge,
                                 left_index=True, right_index=True, how='left')

        # Handle default values
        fillna_dict = {}
        for f in features:
            feature_defaults = {name: f.default_value
                                for name in f.get_feature_names()}
            fillna_dict.update(feature_defaults)

        frame.fillna(fillna_dict, inplace=True)

        # convert boolean dtypes to floats as appropriate
        # pandas behavior: https://github.com/pydata/pandas/issues/3752
        for f in features:
            if (f.number_output_features == 1 and
                    f.variable_type == variable_types.Numeric and
                    frame[f.get_name()].dtype.name in ['object', 'bool']):
                frame[f.get_name()] = frame[f.get_name()].astype(float)

        return frame

    def _necessary_columns(self, entity, feature_names):
        # We have to keep all Id columns because we don't know what forward
        # relationships will come from this node.
        index_columns = {v.id for v in entity.variables
                         if isinstance(v, (variable_types.Index,
                                           variable_types.Id,
                                           variable_types.TimeIndex))}
        features = (self.feature_set.features_by_name[name]
                    for name in feature_names)
        feature_columns = {f.variable.id for f in features
                           if isinstance(f, IdentityFeature)}
        return list(index_columns | feature_columns)


def _can_agg(feature):
    assert isinstance(feature, AggregationFeature)
    base_features = feature.base_features
    if feature.where is not None:
        base_features = [bf.get_name() for bf in base_features
                         if bf.get_name() != feature.where.get_name()]

    if feature.primitive.uses_calc_time:
        return False
    single_output = feature.primitive.number_output_features == 1
    return len(base_features) == 1 and single_output


def agg_wrapper(feats, time_last):
    def wrap(df):
        d = {}
        for f in feats:
            func = f.get_function()
            variable_ids = [bf.get_name() for bf in f.base_features]
            args = [df[v] for v in variable_ids]

            if f.primitive.uses_calc_time:
                values = func(*args, time=time_last)
            else:
                values = func(*args)

            if f.number_output_features == 1:
                values = [values]
            update_feature_columns(f, d, values)

        return pd.Series(d)
    return wrap


def set_default_column(frame, f):
    for name in f.get_feature_names():
        frame[name] = f.default_value


def update_feature_columns(feature, data, values):
    names = feature.get_feature_names()
    assert len(names) == len(values)
    for name, value in zip(names, values):
        data[name] = value


def strip_values_if_series(values):
    if isinstance(values, pd.Series):
        values = values.values
    return values
