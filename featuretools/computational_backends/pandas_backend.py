import cProfile
import os
import pstats
import sys
import warnings
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import pandas.api.types as pdtypes

from .base_backend import ComputationalBackend
from .feature_tree import FeatureTree

from featuretools import variable_types
from featuretools.exceptions import UnknownFeature
from featuretools.feature_base import (
    AggregationFeature,
    DirectFeature,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature
)
from featuretools.utils import is_python_2
from featuretools.utils.gen_utils import (
    get_relationship_variable_id,
    make_tqdm_iterator
)

warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)


class PandasBackend(ComputationalBackend):

    def __init__(self, entityset, features):
        assert len(set(f.entity.id for f in features)) == 1, \
            "Features must all be defined on the same entity"

        self.entityset = entityset
        self.target_eid = features[0].entity.id
        self.features = features
        self.feature_tree = FeatureTree(entityset, features)

    def __sizeof__(self):
        return self.entityset.__sizeof__()

    def calculate_all_features(self, instance_ids, time_last,
                               training_window=None, profile=False,
                               precalculated_features=None, ignored=None,
                               verbose=False):
        """
        Given a list of instance ids and features with a shared time window,
        generate and return a mapping of instance -> feature values.

        Args:
            instance_ids (list): List of instance id for which to build features.

            time_last (pd.Timestamp): Last allowed time. Data from exactly this
                time not allowed.

            training_window (Timedelta, optional): Data older than
                time_last by more than this will be ignored.

            profile (bool): Enable profiler if True.

            verbose (bool): Print output progress if True.

        Returns:
            pd.DataFrame : Pandas DataFrame of calculated feature values.
                Indexed by instance_ids. Columns in same order as features
                passed in.

        """
        assert len(instance_ids) > 0, "0 instance ids provided"
        self.instance_ids = instance_ids

        self.time_last = time_last
        if self.time_last is None:
            self.time_last = datetime.now()

        # For debugging
        if profile:
            pr = cProfile.Profile()
            pr.enable()

        if precalculated_features is None:
            precalculated_features = {}
        # Access the index to get the filtered data we need
        target_entity = self.entityset[self.target_eid]
        if ignored:
            # TODO: Just want to remove entities if don't have any (sub)features defined
            # on them anymore, rather than recreating
            ordered_entities = FeatureTree(self.entityset, self.features, ignored=ignored).ordered_entities
        else:
            ordered_entities = self.feature_tree.ordered_entities

        necessary_columns = self.feature_tree.necessary_columns
        eframes_by_filter = \
            self.entityset.get_pandas_data_slice(filter_entity_ids=ordered_entities,
                                                 index_eid=self.target_eid,
                                                 instances=instance_ids,
                                                 entity_columns=necessary_columns,
                                                 time_last=time_last,
                                                 training_window=training_window,
                                                 verbose=verbose)
        large_eframes_by_filter = None
        if any([f.primitive.uses_full_entity
                for f in self.feature_tree.all_features
                if isinstance(f, (GroupByTransformFeature, TransformFeature))]):
            large_necessary_columns = self.feature_tree.necessary_columns_for_all_values_features
            large_eframes_by_filter = \
                self.entityset.get_pandas_data_slice(filter_entity_ids=ordered_entities,
                                                     index_eid=self.target_eid,
                                                     instances=None,
                                                     entity_columns=large_necessary_columns,
                                                     time_last=time_last,
                                                     training_window=training_window,
                                                     verbose=verbose)

        # Handle an empty time slice by returning a dataframe with defaults
        if eframes_by_filter is None:
            return self.generate_default_df(instance_ids=instance_ids)

        finished_entity_ids = []
        # Populate entity_frames with precalculated features
        if len(precalculated_features) > 0:
            for entity_id, precalc_feature_values in precalculated_features.items():
                if entity_id in eframes_by_filter:
                    frame = eframes_by_filter[entity_id][entity_id]
                    eframes_by_filter[entity_id][entity_id] = pd.merge(frame,
                                                                       precalc_feature_values,
                                                                       left_index=True,
                                                                       right_index=True)
                else:
                    # Only features we're taking from this entity
                    # are precomputed
                    # Make sure the id variable is a column as well as an index
                    entity_id_var = self.entityset[entity_id].index
                    precalc_feature_values[entity_id_var] = precalc_feature_values.index.values
                    eframes_by_filter[entity_id] = {entity_id: precalc_feature_values}
                    finished_entity_ids.append(entity_id)

        # Iterate over the top-level entities (filter entities) in sorted order
        # and calculate all relevant features under each one.
        if verbose:
            total_groups_to_compute = sum(len(group)
                                          for group in self.feature_tree.ordered_feature_groups.values())

            pbar = make_tqdm_iterator(total=total_groups_to_compute,
                                      desc="Computing features",
                                      unit="feature group")
            if verbose:
                pbar.update(0)

        for filter_eid in ordered_entities:
            entity_frames = eframes_by_filter[filter_eid]
            large_entity_frames = None
            if large_eframes_by_filter is not None:
                large_entity_frames = large_eframes_by_filter[filter_eid]

            # update the current set of entity frames with the computed features
            # from previously finished entities
            for eid in finished_entity_ids:
                # only include this frame if it's not from a descendent entity:
                # descendent entity frames will have to be re-calculated.
                # TODO: this check might not be necessary, depending on our
                # constraints
                if not self.entityset.find_backward_path(start_entity_id=filter_eid,
                                                         goal_entity_id=eid):
                    entity_frames[eid] = eframes_by_filter[eid][eid]
                    # TODO: look this over again
                    # precalculated features will only be placed in entity_frames,
                    # and it's possible that that they are the only features computed
                    # for an entity. In this case, the entity won't be present in
                    # large_eframes_by_filter. The relevant lines that this case passes
                    # through are 136-143
                    if (large_eframes_by_filter is not None and
                            eid in large_eframes_by_filter and eid in large_eframes_by_filter[eid]):
                        large_entity_frames[eid] = large_eframes_by_filter[eid][eid]

            if filter_eid in self.feature_tree.ordered_feature_groups:
                for group in self.feature_tree.ordered_feature_groups[filter_eid]:
                    if verbose:
                        pbar.set_postfix({'running': 0})

                    test_feature = group[0]
                    entity_id = test_feature.entity.id

                    input_frames_type = self.feature_tree.input_frames_type(test_feature)

                    input_frames = large_entity_frames
                    if input_frames_type == "subset_entity_frames":
                        input_frames = entity_frames

                    handler = self._feature_type_handler(test_feature)
                    result_frame = handler(group, input_frames)

                    output_frames_type = self.feature_tree.output_frames_type(test_feature)
                    if output_frames_type in ['full_and_subset_entity_frames', 'subset_entity_frames']:
                        index = entity_frames[entity_id].index
                        # If result_frame came from a uses_full_entity feature,
                        # and the input was large_entity_frames,
                        # then it's possible it doesn't contain some of the features
                        # in the output entity_frames
                        # We thus need to concatenate the existing frame with the result frame,
                        # making sure not to duplicate any columns
                        _result_frame = result_frame.reindex(index)
                        cols_to_keep = [c for c in _result_frame.columns
                                        if c not in entity_frames[entity_id].columns]
                        entity_frames[entity_id] = pd.concat([entity_frames[entity_id],
                                                              _result_frame[cols_to_keep]],
                                                             axis=1)

                    if output_frames_type in ['full_and_subset_entity_frames', 'full_entity_frames']:
                        index = large_entity_frames[entity_id].index
                        _result_frame = result_frame.reindex(index)
                        cols_to_keep = [c for c in _result_frame.columns
                                        if c not in large_entity_frames[entity_id].columns]
                        large_entity_frames[entity_id] = pd.concat([large_entity_frames[entity_id],
                                                                    _result_frame[cols_to_keep]],
                                                                   axis=1)

                    if verbose:
                        pbar.update(1)

            finished_entity_ids.append(filter_eid)

        if verbose:
            pbar.set_postfix({'running': 0})
            pbar.refresh()
            sys.stdout.flush()
            pbar.close()

        # debugging
        if profile:
            pr.disable()
            ROOT_DIR = os.path.expanduser("~")
            prof_folder_path = os.path.join(ROOT_DIR, 'prof')
            if not os.path.exists(prof_folder_path):
                os.mkdir(prof_folder_path)
            with open(os.path.join(prof_folder_path, 'inst-%s.log' %
                                   list(instance_ids)[0]), 'w') as f:
                pstats.Stats(pr, stream=f).strip_dirs().sort_stats("cumulative", "tottime").print_stats()

        df = eframes_by_filter[self.target_eid][self.target_eid]

        # fill in empty rows with default values
        missing_ids = [i for i in instance_ids if i not in
                       df[target_entity.index]]
        if missing_ids:
            default_df = self.generate_default_df(instance_ids=missing_ids,
                                                  extra_columns=df.columns)
            df = df.append(default_df, sort=True)

        df.index.name = self.entityset[self.target_eid].index
        column_list = []
        for feat in self.features:
            column_list.extend(feat.get_feature_names())
        return df[column_list]

    def generate_default_df(self, instance_ids, extra_columns=None):
        index_name = self.features[0].entity.index
        default_row = []
        default_cols = []
        for f in self.features:
            for name in f.get_feature_names():
                default_cols.append(name)
                default_row.append(f.default_value)

        default_matrix = [default_row] * len(instance_ids)
        default_df = pd.DataFrame(default_matrix,
                                  columns=default_cols,
                                  index=instance_ids)
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

    def _calculate_identity_features(self, features, entity_frames):
        entity_id = features[0].entity.id
        return entity_frames[entity_id][[f.get_name() for f in features]]

    def _calculate_transform_features(self, features, entity_frames):
        entity_id = features[0].entity.id
        assert len(set([f.entity.id for f in features])) == 1, \
            "features must share base entity"
        assert entity_id in entity_frames

        frame = entity_frames[entity_id]
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

    def _calculate_groupby_features(self, features, entity_frames):
        entity_id = features[0].entity.id
        assert len(set([f.entity.id for f in features])) == 1, \
            "features must share base entity"
        assert entity_id in entity_frames

        frame = entity_frames[entity_id]

        # handle when no data
        if frame.shape[0] == 0:
            for f in features:
                set_default_column(frame, f)
            return frame

        groupby = features[0].groupby.get_name()
        for index, group in frame.groupby(groupby):
            for f in features:
                column_names = [bf.get_name() for bf in f.base_features]
                # exclude the groupby variable from being passed to the function
                variable_data = [group[name] for name in column_names[:-1]]
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

                feature_name = f.get_name()
                if feature_name in frame.columns:
                    frame[feature_name].update(values)
                else:
                    frame[feature_name] = values

        return frame

    def _calculate_direct_features(self, features, entity_frames):
        entity_id = features[0].entity.id
        parent_entity_id = features[0].parent_entity.id

        assert entity_id in entity_frames and parent_entity_id in entity_frames

        path = self.entityset.find_forward_path(entity_id, parent_entity_id)
        assert len(path) == 1, \
            "Error calculating DirectFeatures, len(path) > 1"

        parent_df = entity_frames[parent_entity_id]
        child_df = entity_frames[entity_id]
        merge_var = path[0].child_variable.id

        # generate a mapping of old column names (in the parent entity) to
        # new column names (in the child entity) for the merge
        col_map = {path[0].parent_variable.id: merge_var}
        index_as_feature = None
        for f in features:
            if f.base_features[0].get_name() == path[0].parent_variable.id:
                index_as_feature = f
            # Sometimes entityset._add_multigenerational_links adds link variables
            # that would ordinarily get calculated as direct features,
            # so we make sure not to attempt to calculate again
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

        new_df = pd.merge(left=child_df, right=merge_df,
                          left_on=merge_var, right_index=True,
                          how='left')

        return new_df

    def _calculate_agg_features(self, features, entity_frames):
        test_feature = features[0]
        entity = test_feature.entity
        child_entity = test_feature.base_features[0].entity

        assert entity.id in entity_frames and child_entity.id in entity_frames

        frame = entity_frames[entity.id]
        base_frame = entity_frames[child_entity.id]
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
            relationship_path = self.entityset.find_backward_path(entity.id,
                                                                  child_entity.id)

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
