import html

from featuretools.feature_base.feature_base import (
    AggregationFeature,
    DirectFeature,
    FeatureOutputSlice,
    IdentityFeature,
    TransformFeature,
)
from featuretools.feature_base.feature_descriptions import describe_feature
from featuretools.utils.plot_utils import (
    check_graphviz,
    get_graphviz_format,
    save_graph,
)

TARGET_COLOR = "#D9EAD3"
TABLE_TEMPLATE = """<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="10">
    <TR>
        <TD colspan="1" bgcolor="#A9A9A9"><B>{dataframe_name}</B></TD>
    </TR>{table_cols}
</TABLE>>"""
COL_TEMPLATE = """<TR><TD ALIGN="LEFT" port="{}">{}</TD></TR>"""
TARGET_TEMPLATE = """
    <TR>
        <TD ALIGN="LEFT" port="{}" BGCOLOR="{target_color}">{}</TD>
    </TR>""".format(
    "{}",
    "{}",
    target_color=TARGET_COLOR,
)


def graph_feature(feature, to_file=None, description=False, **kwargs):
    """Generates a feature lineage graph for the given feature

    Args:
        feature (FeatureBase) : Feature to generate lineage graph for
        to_file (str, optional) : Path to where the plot should be saved.
            If set to None (as by default), the plot will not be saved.
        description (bool or str, optional): The feature description to use as a caption
            for the graph. If False, no description is added. Set to True
            to use an auto-generated description. Defaults to False.
        kwargs (keywords): Additional keyword arguments to pass as keyword arguments
            to the ft.describe_feature function.

    Returns:
        graphviz.Digraph : Graph object that can directly be displayed in Jupyter notebooks.
    """
    graphviz = check_graphviz()
    format_ = get_graphviz_format(graphviz=graphviz, to_file=to_file)

    # Initialize a new directed graph
    graph = graphviz.Digraph(
        feature.get_name(),
        format=format_,
        graph_attr={"rankdir": "LR"},
    )

    dataframes = {}
    edges = ([], [])
    primitives = []
    groupbys = []

    _, max_depth = get_feature_data(
        feature,
        dataframes,
        groupbys,
        edges,
        primitives,
        layer=0,
    )
    dataframes[feature.dataframe_name]["targets"].add(feature.get_name())

    for df_name in dataframes:
        dataframe_name = (
            "\u2605 {} (target)".format(df_name)
            if df_name == feature.dataframe_name
            else df_name
        )
        dataframe_table = get_dataframe_table(dataframe_name, dataframes[df_name])
        graph.attr("node", shape="plaintext")
        graph.node(df_name, dataframe_table)

    graph.attr("node", shape="diamond")
    num_primitives = len(primitives)
    for prim_name, prim_label, layer, prim_type in primitives:
        step_num = max_depth - layer
        if num_primitives == 1:
            type_str = (
                '<FONT POINT-SIZE="12"><B>{}</B><BR></BR></FONT>'.format(prim_type)
                if prim_type
                else ""
            )
            prim_label = "<{}{}>".format(type_str, prim_label)
        else:
            step = "Step {}".format(step_num)
            type_str = "   " + prim_type if prim_type else ""
            prim_label = (
                '<<FONT POINT-SIZE="12"><B>{}:</B>{}<BR></BR></FONT>{}>'.format(
                    step,
                    type_str,
                    prim_label,
                )
            )

        # sink first layer transform primitive if multiple primitives
        if step_num == 1 and prim_type == "Transform" and num_primitives > 1:
            with graph.subgraph() as init_transform:
                init_transform.attr(rank="min")
                init_transform.node(name=prim_name, label=prim_label)
        else:
            graph.node(name=prim_name, label=prim_label)

    graph.attr("node", shape="box")
    for groupby_name, groupby_label in groupbys:
        graph.node(name=groupby_name, label=groupby_label)

    graph.attr("edge", style="solid", dir="forward")
    for edge in edges[1]:
        graph.edge(*edge)

    graph.attr("edge", style="dotted", arrowhead="none", dir="forward")
    for edge in edges[0]:
        graph.edge(*edge)

    if description is True:
        graph.attr(label=describe_feature(feature, **kwargs))
    elif description is not False:
        graph.attr(label=description)

    if to_file:
        save_graph(graph, to_file, format_)

    return graph


def get_feature_data(feat, dataframes, groupbys, edges, primitives, layer=0):
    # 1) add feature to dataframes tables:
    feat_name = feat.get_name()
    if feat.dataframe_name not in dataframes:
        add_dataframe(feat.dataframe, dataframes)
    dataframe_dict = dataframes[feat.dataframe_name]

    # if we've already explored this feat, continue
    feat_node = "{}:{}".format(feat.dataframe_name, feat_name)
    if feat_name in dataframe_dict["columns"] or feat_name in dataframe_dict["feats"]:
        return feat_node, layer

    if isinstance(feat, IdentityFeature):
        dataframe_dict["columns"].add(feat_name)
    else:
        dataframe_dict["feats"].add(feat_name)
    base_node = feat_node

    # 2) if multi-output, convert feature to generic base
    if isinstance(feat, FeatureOutputSlice):
        feat = feat.base_feature
        feat_name = feat.get_name()

    # 3) add primitive node
    if feat.primitive.name or isinstance(feat, DirectFeature):
        prim_name = feat.primitive.name if feat.primitive.name else "join"
        prim_type = ""
        if isinstance(feat, AggregationFeature):
            prim_type = "Aggregation"
        elif isinstance(feat, TransformFeature):
            prim_type = "Transform"
        primitive_node = "{}_{}_{}".format(layer, feat_name, prim_name)
        primitives.append((primitive_node, prim_name.upper(), layer, prim_type))

        edges[1].append([primitive_node, base_node])
        base_node = primitive_node

    # 4) add groupby/join edges and nodes
    dependencies = [(dep.hash(), dep) for dep in feat.get_dependencies()]
    for is_forward, r in feat.relationship_path:
        if is_forward:
            if r.child_dataframe.ww.name not in dataframes:
                add_dataframe(r.child_dataframe, dataframes)
            dataframes[r.child_dataframe.ww.name]["columns"].add(r._child_column_name)
            child_node = "{}:{}".format(r.child_dataframe.ww.name, r._child_column_name)
            edges[0].append([base_node, child_node])
        else:
            if r.child_dataframe.ww.name not in dataframes:
                add_dataframe(r.child_dataframe, dataframes)
            dataframes[r.child_dataframe.ww.name]["columns"].add(r._child_column_name)
            child_node = "{}:{}".format(r.child_dataframe.ww.name, r._child_column_name)
            child_name = child_node.replace(":", "--")
            groupby_node = "{}_groupby_{}".format(feat_name, child_name)
            groupby_name = "group by\n{}".format(r._child_column_name)
            groupbys.append((groupby_node, groupby_name))
            edges[0].append([child_node, groupby_node])
            edges[1].append([groupby_node, base_node])
            base_node = groupby_node

    if hasattr(feat, "groupby"):
        groupby = feat.groupby
        _ = get_feature_data(
            groupby,
            dataframes,
            groupbys,
            edges,
            primitives,
            layer + 1,
        )
        dependencies.remove((groupby.hash(), groupby))

        groupby_name = groupby.get_name()
        if isinstance(groupby, IdentityFeature):
            dataframes[groupby.dataframe_name]["columns"].add(groupby_name)
        else:
            dataframes[groupby.dataframe_name]["feats"].add(groupby_name)

        child_node = "{}:{}".format(groupby.dataframe_name, groupby_name)
        child_name = child_node.replace(":", "--")
        groupby_node = "{}_groupby_{}".format(feat_name, child_name)
        groupby_name = "group by\n{}".format(groupby_name)
        groupbys.append((groupby_node, groupby_name))
        edges[0].append([child_node, groupby_node])
        edges[1].append([groupby_node, base_node])
        base_node = groupby_node

    # 5) recurse over dependents
    max_depth = layer
    for _, f in dependencies:
        dependent_node, depth = get_feature_data(
            f,
            dataframes,
            groupbys,
            edges,
            primitives,
            layer + 1,
        )
        edges[1].append([dependent_node, base_node])

        max_depth = max(depth, max_depth)

    return feat_node, max_depth


def add_dataframe(dataframe, dataframe_dict):
    dataframe_dict[dataframe.ww.name] = {
        "index": dataframe.ww.index,
        "targets": set(),
        "columns": set(),
        "feats": set(),
    }


def get_dataframe_table(dataframe_name, dataframe_dict):
    """
    given a dict of columns and feats, construct the html table for it
    """
    index = dataframe_dict["index"]
    targets = dataframe_dict["targets"]
    columns = dataframe_dict["columns"].difference(targets)
    feats = dataframe_dict["feats"].difference(targets)

    # If the index is used, make sure it's the first element in the table
    clean_index = html.escape(index)
    if index in columns:
        rows = [COL_TEMPLATE.format(clean_index, clean_index + " (index)")]
        columns.discard(index)
    elif index in targets:
        rows = [TARGET_TEMPLATE.format(clean_index, clean_index + " (index)")]
        targets.discard(index)
    else:
        rows = []

    for col in list(columns) + list(feats) + list(targets):
        template = COL_TEMPLATE
        if col in targets:
            template = TARGET_TEMPLATE

        col = html.escape(col)
        rows.append(template.format(col, col))

    table = TABLE_TEMPLATE.format(
        dataframe_name=dataframe_name,
        table_cols="\n".join(rows),
    )
    return table
