import json
import os
import re

import graphviz
import pytest

from featuretools.feature_base import (
    AggregationFeature,
    DirectFeature,
    FeatureOutputSlice,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature,
    graph_feature,
)
from featuretools.primitives import Count, CumMax, Mode, NMostCommon, Year


@pytest.fixture
def simple_feat(es):
    return IdentityFeature(es["log"].ww["id"])


@pytest.fixture
def trans_feat(es):
    return TransformFeature(IdentityFeature(es["customers"].ww["cancel_date"]), Year)


def test_returns_digraph_object(simple_feat):
    graph = graph_feature(simple_feat)
    assert isinstance(graph, graphviz.Digraph)


def test_saving_png_file(simple_feat, tmp_path):
    output_path = str(tmp_path.joinpath("test1.png"))
    graph_feature(simple_feat, to_file=output_path)
    assert os.path.isfile(output_path)


def test_missing_file_extension(simple_feat):
    output_path = "test1"
    with pytest.raises(ValueError, match="Please use a file extension"):
        graph_feature(simple_feat, to_file=output_path)


def test_invalid_format(simple_feat):
    output_path = "test1.xyz"
    with pytest.raises(ValueError, match="Unknown format"):
        graph_feature(simple_feat, to_file=output_path)


def test_transform(es, trans_feat):
    feat = trans_feat
    graph = graph_feature(feat).source

    feat_name = feat.get_name()
    prim_node = "0_{}_year".format(feat_name)
    dataframe_table = "\u2605 customers (target)"
    prim_edge = 'customers:cancel_date -> "{}"'.format(prim_node)
    feat_edge = '"{}" -> customers:"{}"'.format(prim_node, feat_name)

    graph_components = [feat_name, dataframe_table, prim_node, prim_edge, feat_edge]
    for component in graph_components:
        assert component in graph

    matches = re.findall(r"customers \[label=<\n<TABLE.*?</TABLE>>", graph, re.DOTALL)
    assert len(matches) == 1
    rows = re.findall(r"<TR.*?</TR>", matches[0], re.DOTALL)
    assert len(rows) == 3
    to_match = ["customers", "cancel_date", feat_name]
    for match, row in zip(to_match, rows):
        assert match in row


def test_html_symbols(es, tmp_path):
    output_path_template = str(tmp_path.joinpath("test{}.png"))
    value = IdentityFeature(es["log"].ww["value"])
    gt = value > 5
    lt = value < 5
    ge = value >= 5
    le = value <= 5

    for i, feat in enumerate([gt, lt, ge, le]):
        output_path = output_path_template.format(i)
        graph = graph_feature(feat, to_file=output_path).source
        assert os.path.isfile(output_path)
        assert feat.get_name() in graph


def test_groupby_transform(es):
    feat = GroupByTransformFeature(
        IdentityFeature(es["customers"].ww["age"]),
        CumMax,
        IdentityFeature(es["customers"].ww["cohort"]),
    )
    graph = graph_feature(feat).source

    feat_name = feat.get_name()
    prim_node = "0_{}_cum_max".format(feat_name)
    groupby_node = "{}_groupby_customers--cohort".format(feat_name)
    dataframe_table = "\u2605 customers (target)"

    groupby_edge = 'customers:cohort -> "{}"'.format(groupby_node)
    groupby_input = 'customers:age -> "{}"'.format(groupby_node)
    prim_input = '"{}" -> "{}"'.format(groupby_node, prim_node)
    feat_edge = '"{}" -> customers:"{}"'.format(prim_node, feat_name)

    graph_components = [
        feat_name,
        prim_node,
        groupby_node,
        dataframe_table,
        groupby_edge,
        groupby_input,
        prim_input,
        feat_edge,
    ]
    for component in graph_components:
        assert component in graph

    matches = re.findall(r"customers \[label=<\n<TABLE.*?</TABLE>>", graph, re.DOTALL)
    assert len(matches) == 1
    rows = re.findall(r"<TR.*?</TR>", matches[0], re.DOTALL)
    assert len(rows) == 4
    assert dataframe_table in rows[0]
    assert feat_name in rows[-1]
    assert ("age" in rows[1] and "cohort" in rows[2]) or (
        "age" in rows[2] and "cohort" in rows[1]
    )


def test_groupby_transform_direct_groupby(es):
    groupby = DirectFeature(
        IdentityFeature(es["cohorts"].ww["cohort_name"]),
        "customers",
    )
    feat = GroupByTransformFeature(
        IdentityFeature(es["customers"].ww["age"]),
        CumMax,
        groupby,
    )
    graph = graph_feature(feat).source

    groupby_name = groupby.get_name()
    feat_name = feat.get_name()
    join_node = "1_{}_join".format(groupby_name)
    prim_node = "0_{}_cum_max".format(feat_name)
    groupby_node = "{}_groupby_customers--{}".format(feat_name, groupby_name)
    customers_table = "\u2605 customers (target)"
    cohorts_table = "cohorts"

    join_groupby = '"{}" -> customers:cohort'.format(join_node)
    join_input = 'cohorts:cohort_name -> "{}"'.format(join_node)
    join_out_edge = '"{}" -> customers:"{}"'.format(join_node, groupby_name)
    groupby_edge = 'customers:"{}" -> "{}"'.format(groupby_name, groupby_node)
    groupby_input = 'customers:age -> "{}"'.format(groupby_node)
    prim_input = '"{}" -> "{}"'.format(groupby_node, prim_node)
    feat_edge = '"{}" -> customers:"{}"'.format(prim_node, feat_name)

    graph_components = [
        groupby_name,
        feat_name,
        join_node,
        prim_node,
        groupby_node,
        customers_table,
        cohorts_table,
        join_groupby,
        join_input,
        join_out_edge,
        groupby_edge,
        groupby_input,
        prim_input,
        feat_edge,
    ]
    for component in graph_components:
        assert component in graph

    dataframes = {
        "cohorts": [cohorts_table, "cohort_name"],
        "customers": [customers_table, "cohort", "age", groupby_name, feat_name],
    }
    for dataframe in dataframes:
        regex = r"{} \[label=<\n<TABLE.*?</TABLE>>".format(dataframe)
        matches = re.findall(regex, graph, re.DOTALL)
        assert len(matches) == 1

        rows = re.findall(r"<TR.*?</TR>", matches[0], re.DOTALL)
        assert len(rows) == len(dataframes[dataframe])

        for row in rows:
            matched = False
            for i in dataframes[dataframe]:
                if i in row:
                    matched = True
                    dataframes[dataframe].remove(i)
                    break
            assert matched


def test_aggregation(es):
    feat = AggregationFeature(IdentityFeature(es["log"].ww["id"]), "sessions", Count)
    graph = graph_feature(feat).source

    feat_name = feat.get_name()
    prim_node = "0_{}_count".format(feat_name)
    groupby_node = "{}_groupby_log--session_id".format(feat_name)

    sessions_table = "\u2605 sessions (target)"
    log_table = "log"
    groupby_edge = 'log:session_id -> "{}"'.format(groupby_node)
    groupby_input = 'log:id -> "{}"'.format(groupby_node)
    prim_input = '"{}" -> "{}"'.format(groupby_node, prim_node)
    feat_edge = '"{}" -> sessions:"{}"'.format(prim_node, feat_name)

    graph_components = [
        feat_name,
        prim_node,
        groupby_node,
        sessions_table,
        log_table,
        groupby_edge,
        groupby_input,
        prim_input,
        feat_edge,
    ]

    for component in graph_components:
        assert component in graph

    dataframes = {
        "log": [log_table, "id", "session_id"],
        "sessions": [sessions_table, feat_name],
    }
    for dataframe in dataframes:
        regex = r"{} \[label=<\n<TABLE.*?</TABLE>>".format(dataframe)
        matches = re.findall(regex, graph, re.DOTALL)
        assert len(matches) == 1

        rows = re.findall(r"<TR.*?</TR>", matches[0], re.DOTALL)
        assert len(rows) == len(dataframes[dataframe])
        for row in rows:
            matched = False
            for i in dataframes[dataframe]:
                if i in row:
                    matched = True
                    dataframes[dataframe].remove(i)
                    break
            assert matched


def test_multioutput(es):
    multioutput = AggregationFeature(
        IdentityFeature(es["log"].ww["zipcode"]),
        "sessions",
        NMostCommon,
    )
    feat = FeatureOutputSlice(multioutput, 0)
    graph = graph_feature(feat).source

    feat_name = feat.get_name()
    prim_node = "0_{}_n_most_common".format(multioutput.get_name())
    groupby_node = "{}_groupby_log--session_id".format(multioutput.get_name())

    sessions_table = "\u2605 sessions (target)"
    log_table = "log"
    groupby_edge = 'log:session_id -> "{}"'.format(groupby_node)
    groupby_input = 'log:zipcode -> "{}"'.format(groupby_node)
    prim_input = '"{}" -> "{}"'.format(groupby_node, prim_node)
    feat_edge = '"{}" -> sessions:"{}"'.format(prim_node, feat_name)

    graph_components = [
        feat_name,
        prim_node,
        groupby_node,
        sessions_table,
        log_table,
        groupby_edge,
        groupby_input,
        prim_input,
        feat_edge,
    ]

    for component in graph_components:
        assert component in graph

    dataframes = {
        "log": [log_table, "zipcode", "session_id"],
        "sessions": [sessions_table, feat_name],
    }
    for dataframe in dataframes:
        regex = r"{} \[label=<\n<TABLE.*?</TABLE>>".format(dataframe)
        matches = re.findall(regex, graph, re.DOTALL)
        assert len(matches) == 1

        rows = re.findall(r"<TR.*?</TR>", matches[0], re.DOTALL)
        assert len(rows) == len(dataframes[dataframe])
        for row in rows:
            matched = False
            for i in dataframes[dataframe]:
                if i in row:
                    matched = True
                    dataframes[dataframe].remove(i)
                    break
            assert matched


def test_direct(es):
    d1 = DirectFeature(
        IdentityFeature(es["customers"].ww["engagement_level"]),
        "sessions",
    )
    d2 = DirectFeature(d1, "log")
    graph = graph_feature(d2).source

    d1_name = d1.get_name()
    d2_name = d2.get_name()
    prim_node1 = "1_{}_join".format(d1_name)
    prim_node2 = "0_{}_join".format(d2_name)

    log_table = "\u2605 log (target)"
    sessions_table = "sessions"
    customers_table = "customers"
    groupby_edge1 = '"{}" -> sessions:customer_id'.format(prim_node1)
    groupby_edge2 = '"{}" -> log:session_id'.format(prim_node2)
    groupby_input1 = 'customers:engagement_level -> "{}"'.format(prim_node1)
    groupby_input2 = 'sessions:"{}" -> "{}"'.format(d1_name, prim_node2)
    d1_edge = '"{}" -> sessions:"{}"'.format(prim_node1, d1_name)
    d2_edge = '"{}" -> log:"{}"'.format(prim_node2, d2_name)

    graph_components = [
        d1_name,
        d2_name,
        prim_node1,
        prim_node2,
        log_table,
        sessions_table,
        customers_table,
        groupby_edge1,
        groupby_edge2,
        groupby_input1,
        groupby_input2,
        d1_edge,
        d2_edge,
    ]
    for component in graph_components:
        assert component in graph

    dataframes = {
        "customers": [customers_table, "engagement_level"],
        "sessions": [sessions_table, "customer_id", d1_name],
        "log": [log_table, "session_id", d2_name],
    }

    for dataframe in dataframes:
        regex = r"{} \[label=<\n<TABLE.*?</TABLE>>".format(dataframe)
        matches = re.findall(regex, graph, re.DOTALL)
        assert len(matches) == 1

        rows = re.findall(r"<TR.*?</TR>", matches[0], re.DOTALL)
        assert len(rows) == len(dataframes[dataframe])
        for row in rows:
            matched = False
            for i in dataframes[dataframe]:
                if i in row:
                    matched = True
                    dataframes[dataframe].remove(i)
                    break
            assert matched


def test_stacked(es, trans_feat):
    stacked = AggregationFeature(trans_feat, "cohorts", Mode)
    graph = graph_feature(stacked).source

    feat_name = stacked.get_name()
    intermediate_name = trans_feat.get_name()
    agg_primitive = "0_{}_mode".format(feat_name)
    trans_primitive = "1_{}_year".format(intermediate_name)
    groupby_node = "{}_groupby_customers--cohort".format(feat_name)

    trans_prim_edge = 'customers:cancel_date -> "{}"'.format(trans_primitive)
    intermediate_edge = '"{}" -> customers:"{}"'.format(
        trans_primitive,
        intermediate_name,
    )
    groupby_edge = 'customers:cohort -> "{}"'.format(groupby_node)
    groupby_input = 'customers:"{}" -> "{}"'.format(intermediate_name, groupby_node)
    agg_input = '"{}" -> "{}"'.format(groupby_node, agg_primitive)
    feat_edge = '"{}" -> cohorts:"{}"'.format(agg_primitive, feat_name)

    graph_components = [
        feat_name,
        intermediate_name,
        agg_primitive,
        trans_primitive,
        groupby_node,
        trans_prim_edge,
        intermediate_edge,
        groupby_edge,
        groupby_input,
        agg_input,
        feat_edge,
    ]
    for component in graph_components:
        assert component in graph

    agg_primitive = agg_primitive.replace("(", "\\(").replace(")", "\\)")
    agg_node = re.findall('"{}" \\[label.*'.format(agg_primitive), graph)
    assert len(agg_node) == 1
    assert "Step 2" in agg_node[0]

    trans_primitive = trans_primitive.replace("(", "\\(").replace(")", "\\)")
    trans_node = re.findall('"{}" \\[label.*'.format(trans_primitive), graph)
    assert len(trans_node) == 1
    assert "Step 1" in trans_node[0]


def test_description_auto_caption(trans_feat):
    default_graph = graph_feature(trans_feat, description=True).source
    default_label = 'label="The year of the \\"cancel_date\\"."'
    assert default_label in default_graph


def test_description_auto_caption_metadata(trans_feat, tmp_path):
    feature_descriptions = {"customers: cancel_date": "the date the customer cancelled"}
    primitive_templates = {"year": "the year that {} occurred"}
    metadata_graph = graph_feature(
        trans_feat,
        description=True,
        feature_descriptions=feature_descriptions,
        primitive_templates=primitive_templates,
    ).source

    metadata_label = 'label="The year that the date the customer cancelled occurred."'
    assert metadata_label in metadata_graph

    metadata = {
        "feature_descriptions": feature_descriptions,
        "primitive_templates": primitive_templates,
    }
    metadata_path = os.path.join(tmp_path, "description_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    json_metadata_graph = graph_feature(
        trans_feat,
        description=True,
        metadata_file=metadata_path,
    ).source
    assert metadata_label in json_metadata_graph


def test_description_custom_caption(trans_feat):
    custom_description = "A custom feature description"
    custom_description_graph = graph_feature(
        trans_feat,
        description=custom_description,
    ).source
    custom_description_label = 'label="A custom feature description"'
    assert custom_description_label in custom_description_graph
