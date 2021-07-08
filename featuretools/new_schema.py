import os
import tempfile

import featuretools as ft
from featuretools.entityset.serialize import \
    SCHEMA_VERSION as entity_schema_version
from featuretools.entityset.serialize import (
    create_archive,
    dump_data_description
)
from featuretools.feature_base.features_serializer import \
    SCHEMA_VERSION as feature_schema_version
from featuretools.feature_base.features_serializer import save_features
from featuretools.primitives import (
    Count,
    CumSum,
    Day,
    Haversine,
    Max,
    Mean,
    Min,
    Mode,
    Month,
    NumCharacters,
    NumUnique,
    NumWords,
    PercentTrue,
    Skew,
    Std,
    Sum,
    Weekday,
    Year,
    make_agg_primitive
)
from featuretools.tests.testing_utils.mock_ds import make_ecommerce_entityset

entityset = make_ecommerce_entityset()

with tempfile.TemporaryDirectory() as tmpdir:
    os.makedirs(os.path.join(tmpdir, 'data'))
    dump_data_description(entityset, tmpdir, index=False, sep=',', encoding='utf-8', engine='python', compression=None)
    file_path = create_archive(tmpdir)
    os.rename(file_path, "./test_serialization_data_entityset_schema_{}.tar".format(SCHEMA_VERSION))



es = make_ecommerce_entityset()
agg_primitives = [Sum, Std, Max, Skew, Min, Mean, Count, PercentTrue,
                  NumUnique, Mode]
trans_primitives = [Day, Year, Month, Weekday, Haversine, NumWords,
                    NumCharacters]
features = ft.dfs(target_entity='sessions', entityset=es, features_only=True, agg_primitives=agg_primitives, trans_primitives=trans_primitives)
save_features(features, "./test_feature_serialization_feature_schema_{}_entityset_schema_{}.json".format(feature_schema_version, entity_schema_version))
