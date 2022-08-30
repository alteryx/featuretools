"""
dfs_profile.py

Helper module to allow profiling of the dfs operations.  At some point we may
want to use pstats to output the results to a log, but I'm anticipating that
LookingGlass will provide the performance data we want.

Notes:
  - output currently goes to the root directory and is in dfs_profile.stats
  - *.stats is gitignored
  - it uses the demo customers dataset for testing
  - max_depth > 2 is very slow (currently)
  - stats output can be viewed online with https://nejc.saje.info/pstats-viewer.html
"""
import cProfile
from pathlib import Path

import featuretools as ft
import featuretools.demo as demo
from featuretools.synthesis.dfs import dfs

es = demo.load_retail()

all_aggs = ft.primitives.get_aggregation_primitives()
all_trans = ft.primitives.get_transform_primitives()

profiler = cProfile.Profile(builtins=False)
profiler.enable()
feature_defs = dfs(
    entityset=es,
    target_dataframe_name="customers",
    trans_primitives=all_trans,
    agg_primitives=all_aggs,
    max_depth=2,
    features_only=True,
)
profiler.disable()
profiler.dump_stats(Path.cwd() / "dfs_profile.stats")
