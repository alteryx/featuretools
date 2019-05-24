class Trie(object):
    """
    A trie (prefix tree) where the keys are lists of hashable objects.
    """
    def __init__(self, default=lambda: None):
        """
        root_value: The value for the root node.
        default: A function returning the value to use for new nodes.
        """
        self._value = default()
        self._children = {}  # TODO: should this be OrderedDict?
        self._default = default

    def children(self):
        """
        A list of pairs of the edges from this node and the nodes they point
        to.
        """
        return self._children.items()

    def get_node(self, path):
        """
        Get the sub-trie at the given path. If it does not yet exist initialize
        it with the default value.
        """
        if path:
            first = path[0]
            rest = path[1:]

            if first in self._children:
                sub_trie = self._children[first]
            else:
                sub_trie = Trie(default=self._default)
                self._children[first] = sub_trie

            return sub_trie.get_node(rest)
        else:
            return self

    def __getitem__(self, path):
        """Get the value at the given path."""
        return self.get_node(path)._value

    def __setitem__(self, path, value):
        """Update the value at the given path."""
        sub_trie = self.get_node(path)
        sub_trie._value = value

    def __iter__(self, stack=None):
        """
        An iterator over the paths and values of the Trie.

        Implemented using depth first search.
        """
        stack = stack or []

        yield(stack, self._value)

        for relationship, sub_trie in self.children():
            stack.append(relationship)

            for value in sub_trie.__iter__(stack):
                yield(value)

            stack.pop()
