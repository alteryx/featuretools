import featuretools as ft

es = ft.demo.load_mock_customer(return_entityset=True)

fm, fl = ft.dfs(target_entity="transactions",
       entityset=es,
       agg_primitives=[],
       trans_primitives=[],
       groupby_trans_primitives=['cum_count'])
#       features_only=True)
