
import os
import re

import graphviz
import pytest

from featuretools.feature_base import (
    AggregationFeature,
    DirectFeature,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature,
    graph_feature
)
from featuretools.primitives import CumMax, Mode, Year


@pytest.fixture
def simple_feat(es):
    return IdentityFeature(es['log']['id'])


def test_returns_digraph_object(simple_feat):
    graph = graph_feature(simple_feat)
    assert isinstance(graph, graphviz.Digraph)


def test_saving_png_file(simple_feat, tmpdir):
    output_path = str(tmpdir.join("test1.png"))
    graph_feature(simple_feat, to_file=output_path)
    assert os.path.isfile(output_path)


def test_missing_file_extension(simple_feat):
    output_path = 'test1'
    with pytest.raises(ValueError, match="Please use a file extension"):
        graph_feature(simple_feat, to_file=output_path)


def test_invalid_format(simple_feat):
    output_path = 'test1.xyz'
    with pytest.raises(ValueError, match='Unknown format'):
        graph_feature(simple_feat, to_file=output_path)


def test_transform(es):
    feat = TransformFeature(es['customers']['cancel_date'], Year)
    graph = graph_feature(feat).source

    feat_name = feat.get_name()
    prim_node = '0_{}_year'.format(feat_name)
    entity_table = '\u2605 customers (target)'
    prim_edge = 'customers:cancel_date -> "{}"'.format(prim_node)
    feat_edge = '"{}" -> customers:"{}"'.format(prim_node, feat_name)

    graph_components = [feat_name, entity_table, prim_node, prim_edge, feat_edge]
    for component in graph_components:
        assert component in graph

    matches = re.findall(r"customers \[label=<\n<TABLE.*?</TABLE>>", graph, re.DOTALL)
    assert len(matches) == 1
    rows = re.findall(r"<TR.*?</TR>", matches[0], re.DOTALL)
    assert len(rows) == 3
    to_match = ['customers', 'cancel_date', feat_name]
    for match, row in zip(to_match, rows):
        assert match in row


def test_groupby_transform(es):
    feat = GroupByTransformFeature(es['customers']['age'], CumMax, es['customers']['cohort'])
    graph = graph_feature(feat).source

    feat_name = feat.get_name()
    prim_node = "0_{}_cum_max".format(feat_name)
    groupby_node = '{}_groupby_customers--cohort'.format(feat_name)
    entity_table = '\u2605 customers (target)'

    groupby_edge = 'customers:cohort -> "{}"'.format(groupby_node)
    groupby_input = 'customers:age -> "{}"'.format(groupby_node)
    prim_input = '"{}" -> "{}"'.format(groupby_node, prim_node)
    feat_edge = '"{}" -> customers:"{}"'.format(prim_node, feat_name)

    graph_components = [feat_name, prim_node, groupby_node, entity_table,
                        groupby_edge, groupby_input, prim_input, feat_edge]
    for component in graph_components:
        assert component in graph

    matches = re.findall(r"customers \[label=<\n<TABLE.*?</TABLE>>", graph, re.DOTALL)
    assert len(matches) == 1
    rows = re.findall(r"<TR.*?</TR>", matches[0], re.DOTALL)
    assert len(rows) == 4
    assert entity_table in rows[0]
    assert feat_name in rows[-1]
    if 'age' in rows[1]:
        assert 'cohort' in rows[2]
    elif 'age' in rows[2]:
        assert 'cohort' in rows[1]
    else:
        assert False


def test_aggregation(es):
    feat = AggregationFeature(es['log']['zipcode'], es['sessions'], Mode)
    graph = graph_feature(feat).source

    feat_name = feat.get_name()
    prim_node = '0_{}_mode'.format(feat_name)
    groupby_node = '{}_groupby_log--session_id'.format(feat_name)

    sessions_table = '\u2605 sessions (target)'
    log_table = 'log'
    groupby_edge = 'log:session_id -> "{}"'.format(groupby_node)
    groupby_input = 'log:zipcode -> "{}"'.format(groupby_node)
    prim_input = '"{}" -> "{}"'.format(groupby_node, prim_node)
    feat_edge = '"{}" -> sessions:"{}"'.format(prim_node, feat_name)

    graph_components = [feat_name, prim_node, groupby_node, sessions_table,
                        log_table, groupby_edge, groupby_input, prim_input,
                        feat_edge]

    for component in graph_components:
        assert component in graph

    entities = {'log': [log_table, 'zipcode', 'session_id'],
                'sessions': [sessions_table, feat_name]}
    for entity in entities:
        regex = r"{} \[label=<\n<TABLE.*?</TABLE>>".format(entity)
        matches = re.findall(regex, graph, re.DOTALL)
        assert len(matches) == 1

        rows = re.findall(r"<TR.*?</TR>", matches[0], re.DOTALL)
        assert len(rows) == len(entities[entity])
        for row in rows:
            matched = False
            for i in entities[entity]:
                if i in row:
                    matched = True
                    entities[entity].remove(i)
                    break
            assert matched


def test_direct(es):
    feat = DirectFeature(es['sessions']['device_name'], es['log'])
    graph = graph_feature(feat).source

    feat_name = feat.get_name()
    prim_node = '0_{}_join'.format(feat_name)

    log_table = '\u2605 log (target)'
    sessions_table = 'sessions'
    groupby_edge = '"{}" -> log:session_id'.format(prim_node)
    groupby_input = 'sessions:device_name -> "{}"'.format(prim_node)
    feat_edge = '"{}" -> log:"{}"'.format(prim_node, feat_name)

    graph_components = [feat_name, prim_node, log_table, sessions_table,
                        groupby_edge, groupby_input, feat_edge]
    for component in graph_components:
        assert component in graph

    entities = {'sessions': [sessions_table, 'device_name'],
                'log': [log_table, 'session_id', feat_name]}

    for entity in entities:
        regex = r"{} \[label=<\n<TABLE.*?</TABLE>>".format(entity)
        matches = re.findall(regex, graph, re.DOTALL)
        assert len(matches) == 1

        rows = re.findall(r"<TR.*?</TR>", matches[0], re.DOTALL)
        assert len(rows) == len(entities[entity])
        for row in rows:
            matched = False
            for i in entities[entity]:
                if i in row:
                    matched = True
                    entities[entity].remove(i)
                    break
            assert matched


def test_stacked(es):
    trans_feat = TransformFeature(es['customers']['cancel_date'], Year)
    stacked = AggregationFeature(trans_feat, es['cohorts'], Mode)
    graph = graph_feature(stacked).source

    feat_name = stacked.get_name()
    intermediate_name = trans_feat.get_name()
    agg_primitive = '0_{}_mode'.format(feat_name)
    trans_primitive = '1_{}_year'.format(intermediate_name)
    groupby_node = '{}_groupby_customers--cohort'.format(feat_name)

    trans_prim_edge = 'customers:cancel_date -> "{}"'.format(trans_primitive)
    intermediate_edge = '"{}" -> customers:"{}"'.format(trans_primitive, intermediate_name)
    groupby_edge = 'customers:cohort -> "{}"'.format(groupby_node)
    groupby_input = 'customers:"{}" -> "{}"'.format(intermediate_name, groupby_node)
    agg_input = '"{}" -> "{}"'.format(groupby_node, agg_primitive)
    feat_edge = '"{}" -> cohorts:"{}"'.format(agg_primitive, feat_name)

    graph_components = [feat_name, intermediate_name, agg_primitive, trans_primitive,
                        groupby_node, trans_prim_edge, intermediate_edge, groupby_edge,
                        groupby_input, agg_input, feat_edge]
    for component in graph_components:
        assert component in graph

    agg_primitive = agg_primitive.replace('(', '\\(').replace(')', '\\)')
    agg_node = re.findall('"{}" \\[label.*'.format(agg_primitive), graph)
    assert len(agg_node) == 1
    assert 'Step 2' in agg_node[0]

    trans_primitive = trans_primitive.replace('(', '\\(').replace(')', '\\)')
    trans_node = re.findall('"{}" \\[label.*'.format(trans_primitive), graph)
    assert len(trans_node) == 1
    assert 'Step 1' in trans_node[0]
