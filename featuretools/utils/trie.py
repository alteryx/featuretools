class Trie(object):
    """
    A trie (prefix tree) where the keys are lists of hashable objects.

    It behaves similarly to a dictionary, except that the keys are lists.

    Examples:
        .. code-block:: python

            from featuretools.utils import Trie

            trie = Trie(default=str)

            # Set a value
            trie[[1, 2, 3]] = '123'

            # Get a value
            assert trie[[1, 2, 3]] == '123'

            # Overwrite a value
            trie[[1, 2, 3]] = 'updated'
            assert trie[[1, 2, 3]] == 'updated'

            # Getting a key that has not been set returns the default value.
            assert trie[[1, 2]] == ''
    """
    def __init__(self, default=lambda: None):
        """
        default: A function returning the value to use for new nodes.
        """
        self._value = default()
        self._children = {}
        self._default = default

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v

    def children(self):
        """
        A list of pairs of the edges from this node and the nodes they point
        to.

        Examples:
            .. code-block:: python

                from featuretools.utils import Trie

                trie = Trie()
                trie[[1, 2]] = '12'
                trie[[3]] = '3'

                children = trie.children()
                first_edge, first_child = children[0]
                second_edge, second_child = children[1]

                assert (first_edge, first_child[[]]) == (1, None)
                assert (second_edge, second_child[[]]) == (3, '3')
        """
        return list(self._children.items())

    def get_node(self, path):
        """
        Get the sub-trie at the given path. If it does not yet exist initialize
        it with the default value.

        Examples:
            .. code-block:: python

                from featuretools.utils import Trie

                t = Trie()

                t[[1, 2, 3]] = '123'
                t[[1, 2, 4]] = '124'
                sub = t.get_node([1, 2])
                assert sub[[3]] == '123'
                assert sub[[4]] == '124'
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

    def __iter__(self, stack=None):
        """
        Iterate over all values in the trie. Yields tuples of (path, value).

        Implemented using depth first search.
        """
        stack = stack or []

        yield(stack, self._value)

        for relationship, sub_trie in self.children():
            stack.append(relationship)

            for path_and_value in sub_trie.__iter__(stack):
                yield(path_and_value)

            stack.pop()
