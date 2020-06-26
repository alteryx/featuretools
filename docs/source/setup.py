import os

import featuretools as ft

def load_feature_plots():
    es = ft.demo.load_mock_customer(return_entityset=True)
    path = os.path.dirname(os.path.abspath(__file__)) + '/automated_feature_engineering/graphs/'
    agg_feat = ft.AggregationFeature(es['sessions']['session_id'], es['customers'], ft.primitives.Count)
    trans_feat = ft.TransformFeature(es['customers']['join_date'], ft.primitives.TimeSincePrevious)
    ft.graph_feature(agg_feat).render(path + 'agg_feat', format='dot')
    ft.graph_feature(trans_feat).render(path + 'trans_feat', format='dot')


if __name__ == "__main__":
    load_feature_plots()
