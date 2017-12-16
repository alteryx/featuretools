from __future__ import print_function
def print_graph(self, start_eid):
    """
    Print a representation of the entityset with parents & children, like
    this:
    TopLevel    TopLevel
    |           |
    |-----------|
    |
    MidLevel
    |
    |-----------|-----------|
    |           |           |
    LowLevel    LowLevel    LowLevel

    Only supports tree-like structures for now.
    """
    back_graph = {start_eid: {}}
    forward_graph = {start_eid: {}}
    self._build_back_graph(back_graph, start_eid)
    self._build_fwd_graph(forward_graph, start_eid)

    top_strings = self._build_strings(back_graph[start_eid])[::-1]
    bot_strings = self._build_strings(forward_graph[start_eid])

    print('\n'.join(reversed(top_strings + [start_eid] + bot_strings)))


def _build_fwd_graph(self, graph, eid):
    for br in self.get_forward_relationships(eid):
        parent_eid = br.parent_variable.entity.id
        graph[eid][parent_eid] = {}
        self._build_fwd_graph(graph[eid], parent_eid)


def _build_back_graph(self, graph, eid):
    for br in self.get_backward_relationships(eid):
        child_eid = br.child_variable.entity.id
        graph[eid][child_eid] = {}
        self._build_back_graph(graph[eid], child_eid)


def _build_strings(self, graph):
    if not len(graph):
        return []

    strings = ['^'] + [''] * 3
    prefix_len = 0

    for eid, g in graph.items():
        recurse_strings = self._build_strings(g)
        strlen = max(len(recurse_strings[0]) if recurse_strings else 0,
                     len(eid) + 4)
        strings[1] = strings[1].replace(' ', '-')
        strings[1] += '|' + (strlen - 1) * ' '
        strings[2] += '|' + (strlen - 1) * ' '
        strings[3] += eid + (strlen - len(eid)) * ' '
        for i, s in enumerate(recurse_strings):
            if len(strings) < i + 5:
                strings.append('')
                assert len(strings) == i + 5

            if len(strings[i + 4]) < prefix_len:
                strings[i + 4] += ' ' * (prefix_len - len(strings[i + 4]))
            strings[i + 4] += s

        prefix_len += strlen

    return strings
