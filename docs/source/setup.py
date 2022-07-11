import os

import featuretools as ft


def load_feature_plots():
    es = ft.demo.load_mock_customer(return_entityset=True)
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "getting_started/graphs/",
    )
    agg_feat = ft.AggregationFeature(
        ft.IdentityFeature(es["sessions"].ww["session_id"]),
        "customers",
        ft.primitives.Count,
    )
    trans_feat = ft.TransformFeature(
        ft.IdentityFeature(es["customers"].ww["join_date"]),
        ft.primitives.TimeSincePrevious,
    )
    demo_feat = ft.AggregationFeature(
        ft.TransformFeature(
            ft.IdentityFeature(es["transactions"].ww["transaction_time"]),
            ft.primitives.Weekday,
        ),
        "sessions",
        ft.primitives.Mode,
    )
    ft.graph_feature(agg_feat, to_file=os.path.join(path, "agg_feat.dot"))
    ft.graph_feature(trans_feat, to_file=os.path.join(path, "trans_feat.dot"))
    ft.graph_feature(demo_feat, to_file=os.path.join(path, "demo_feat.dot"))


if __name__ == "__main__":
    load_feature_plots()
