from featuretools.feature_base.feature_base import (
    AggregationFeature,
    DirectFeature,
    FeatureOutputSlice,
    IdentityFeature,
    TransformFeature
)
from featuretools.utils.plot_utils import (
    check_graphviz,
    get_graphviz_format,
    save_graph
)

TARGET_COLOR = '#D9EAD3'
TABLE_TEMPLATE = '''<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="10">
    <TR>
        <TD colspan="1" bgcolor="#A9A9A9"><B>{entity_name}</B></TD>
    </TR>{table_cols}
</TABLE>>'''
COL_TEMPLATE = '''<TR><TD ALIGN="LEFT" port="{}">{}</TD></TR>'''
TARGET_TEMPLATE = '''
    <TR>
        <TD ALIGN="LEFT" port="{}" BGCOLOR="{target_color}">{}</TD>
    </TR>'''.format('{}', '{}', target_color=TARGET_COLOR)


def graph_feature(feature, to_file=None):
    '''Generates a feature lineage graph for the given feature

    Args:
        feature (FeatureBase) : Feature to generate lineage graph for
        to_file (str, optional) : Path to where the plot should be saved.
            If set to None (as by default), the plot will not be saved.

    Returns:
        graphviz.Digraph : Graph object that can directly be displayed in Jupyter notebooks.
    '''
    graphviz = check_graphviz()
    format_ = get_graphviz_format(graphviz=graphviz,
                                  to_file=to_file)

    # Initialize a new directed graph
    graph = graphviz.Digraph(feature.get_name(), format=format_,
                             graph_attr={'rankdir': 'LR'})

    entities = {}
    edges = ([], [])
    primitives = []
    groupbys = []

    _, max_depth = get_feature_data(feature, entities, groupbys, edges, primitives, layer=0)
    entities[feature.entity.id]['targets'].add(feature.get_name())

    for entity in entities:
        entity_name = '\u2605 {} (target)'.format(entity) if entity == feature.entity.id else entity
        entity_table = get_entity_table(entity_name, entities[entity])
        graph.attr('node', shape='plaintext')
        graph.node(entity, entity_table)

    graph.attr('node', shape='diamond')
    num_primitives = len(primitives)
    for prim_name, prim_label, layer, prim_type in primitives:
        step_num = max_depth - layer
        if num_primitives == 1:
            type_str = '<FONT POINT-SIZE="12"><B>{}</B><BR></BR></FONT>'.format(prim_type) if prim_type else ''
            prim_label = '<{}{}>'.format(type_str, prim_label)
        else:
            step = 'Step {}'.format(step_num)
            type_str = '   ' + prim_type if prim_type else ''
            prim_label = '<<FONT POINT-SIZE="12"><B>{}:</B>{}<BR></BR></FONT>{}>'.format(step, type_str, prim_label)

        # sink first layer transform primitive if multiple primitives
        if step_num == 1 and prim_type == 'Transform' and num_primitives > 1:
            with graph.subgraph() as init_transform:
                init_transform.attr(rank='min')
                init_transform.node(name=prim_name, label=prim_label)
        else:
            graph.node(name=prim_name, label=prim_label)

    graph.attr('node', shape='box')
    for groupby_name, groupby_label in groupbys:
        graph.node(name=groupby_name, label=groupby_label)

    graph.attr('edge', style='solid', dir='forward')
    for edge in edges[1]:
        graph.edge(*edge)

    graph.attr('edge', style='dotted', arrowhead='none', dir='forward')
    for edge in edges[0]:
        graph.edge(*edge)

    if to_file:
        save_graph(graph, to_file, format_)

    return graph


def get_feature_data(feat, entities, groupbys, edges, primitives, layer=0):
    # 1) add feature to entities tables:
    feat_name = feat.get_name()
    if feat.entity.id not in entities:
        add_entity(feat.entity, entities)
    entity_dict = entities[feat.entity.id]

    # if we've already explored this feat, continue
    feat_node = "{}:{}".format(feat.entity.id, feat_name)
    if feat_name in entity_dict['vars'] or feat_name in entity_dict['feats']:
        return feat_node, layer

    if isinstance(feat, IdentityFeature):
        entity_dict['vars'].add(feat_name)
    else:
        entity_dict['feats'].add(feat_name)
    base_node = feat_node

    # 2) if multi-output, convert feature to generic base
    if isinstance(feat, FeatureOutputSlice):
        feat = feat.base_feature
        feat_name = feat.get_name()

    # 3) add primitive node
    if feat.primitive.name or isinstance(feat, DirectFeature):
        prim_name = feat.primitive.name if feat.primitive.name else 'join'
        prim_type = ''
        if isinstance(feat, AggregationFeature):
            prim_type = 'Aggregation'
        elif isinstance(feat, TransformFeature):
            prim_type = 'Transform'
        primitive_node = "{}_{}_{}".format(layer, feat_name, prim_name)
        primitives.append((primitive_node, prim_name.upper(), layer, prim_type))

        edges[1].append([primitive_node, base_node])
        base_node = primitive_node

    # 4) add groupby/join edges and nodes
    dependencies = [(dep.hash(), dep) for dep in feat.get_dependencies()]
    for is_forward, r in feat.relationship_path:
        if is_forward:
            if r.child_entity.id not in entities:
                add_entity(r.child_entity, entities)
            entities[r.child_entity.id]['vars'].add(r.child_variable.name)
            child_node = '{}:{}'.format(r.child_entity.id, r.child_variable.name)
            edges[0].append([base_node, child_node])
        else:
            if r.child_entity.id not in entities:
                add_entity(r.child_entity, entities)
            entities[r.child_entity.id]['vars'].add(r.child_variable.name)
            child_node = '{}:{}'.format(r.child_entity.id, r.child_variable.name)
            child_name = child_node.replace(':', '--')
            groupby_node = "{}_groupby_{}".format(feat_name, child_name)
            groupby_name = 'group by\n{}'.format(r.child_variable.name)
            groupbys.append((groupby_node, groupby_name))
            edges[0].append([child_node, groupby_node])
            edges[1].append([groupby_node, base_node])
            base_node = groupby_node

    if hasattr(feat, 'groupby'):
        groupby = feat.groupby
        _ = get_feature_data(groupby, entities, groupbys, edges, primitives, layer + 1)
        dependencies.remove((groupby.hash(), groupby))

        groupby_name = groupby.get_name()
        if isinstance(groupby, IdentityFeature):
            entities[groupby.entity.id]['vars'].add(groupby_name)
        else:
            entities[groupby.entity.id]['feats'].add(groupby_name)

        child_node = '{}:{}'.format(groupby.entity.id, groupby_name)
        child_name = child_node.replace(':', '--')
        groupby_node = "{}_groupby_{}".format(feat_name, child_name)
        groupby_name = 'group by\n{}'.format(groupby_name)
        groupbys.append((groupby_node, groupby_name))
        edges[0].append([child_node, groupby_node])
        edges[1].append([groupby_node, base_node])
        base_node = groupby_node

    # 5) recurse over dependents
    max_depth = layer
    for _, f in dependencies:
        dependent_node, depth = get_feature_data(f, entities, groupbys, edges, primitives, layer + 1)
        edges[1].append([dependent_node, base_node])

        max_depth = max(depth, max_depth)

    return feat_node, max_depth


def add_entity(entity, entity_dict):
    entity_dict[entity.id] = {
        'index': entity.index,
        'targets': set(),
        'vars': set(),
        'feats': set()
    }


def get_entity_table(entity_name, entity_dict):
    '''
    given a dict of vars and feats, construct the html table for it
    '''
    index = entity_dict['index']
    targets = entity_dict['targets']
    variables = entity_dict['vars'].difference(targets)
    feats = entity_dict['feats'].difference(targets)

    # If the index is used, make sure it's the first element in the table
    if index in variables:
        rows = [COL_TEMPLATE.format(entity_dict['index'], entity_dict['index'] + " (index)")]
        variables.discard(index)
    elif index in targets:
        rows = [TARGET_TEMPLATE.format(entity_dict['index'], entity_dict['index'] + " (index)")]
        targets.discard(index)
    else:
        rows = []

    for var in list(variables) + list(feats) + list(targets):
        template = COL_TEMPLATE
        if var in targets:
            template = TARGET_TEMPLATE

        rows.append(template.format(var, var))

    table = TABLE_TEMPLATE.format(entity_name=entity_name,
                                  table_cols="\n".join(rows))
    return table
