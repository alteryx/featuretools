class Trie(object):
    """
    A trie (prefix tree) where the keys are sequences of hashable objects.

    It behaves similarly to a dictionary, except that the keys can be lists or
    other sequences.

    Examples:
        .. code-block:: python
            from featuretools.utils import Trie

            trie = Trie(default=str)

            # Set a value
            trie.get_node([1, 2, 3]).value = '123'

            # Get a value
            assert trie.get_node([1, 2, 3]).value == '123'

            # Overwrite a value
            trie.get_node([1, 2, 3]).value = 'updated'
            assert trie.get_node([1, 2, 3]).value == 'updated'

            # Getting a key that has not been set returns the default value.
            assert trie.get_node([1, 2]).value == ''
    """

    def __init__(self, default=lambda: None, path_constructor=list):
        """
        default: A function returning the value to use for new nodes.
        path_constructor: A function which constructs a path from a list. The
            path type must support addition (concatenation).
        """
        self.value = default()
        self._children = {}
        self._default = default
        self._path_constructor = path_constructor

    def children(self):
        """
        A list of pairs of the edges from this node and the nodes they point
        to.

        Examples:
            .. code-block:: python

                from featuretools.utils import Trie

                trie = Trie()
                trie.get_node([1, 2]).value = '12'
                trie.get_node([3]).value = '3'

                children = trie.children()
                first_edge, first_child = children[0]
                second_edge, second_child = children[1]

                assert (first_edge, first_child.get_node([]).value) == (1, None)
                assert (second_edge, second_child.get_node([]).value) == (3, '3')
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

                t.get_node([1, 2, 3]).value = '123'
                t.get_node([1, 2, 4]).value = '124'
                sub = t.get_node([1, 2])
                assert sub.get_node([3]).value == '123'
                assert sub.get_node([4]).value == '124'
        """
        if path:
            first = path[0]
            rest = path[1:]

            if first in self._children:
                sub_trie = self._children[first]
            else:
                sub_trie = Trie(default=self._default,
                                path_constructor=self._path_constructor)
                self._children[first] = sub_trie

            return sub_trie.get_node(rest)
        else:
            return self

    def __iter__(self):
        """
        Iterate over all values in the trie. Yields tuples of (path, value).

        Implemented using depth first search.
        """
        yield self._path_constructor([]), self.value

        for key, sub_trie in self.children():
            path_to_children = self._path_constructor([key])

            for sub_path, value in sub_trie:
                path = path_to_children + sub_path
                yield path, value
