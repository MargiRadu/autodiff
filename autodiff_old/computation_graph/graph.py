import numpy as np


class Graph:

    def __init__(self):
        pass

    def add_node(self, node_identifier):
        pass

    def add_edge(self, a, b, value):
        pass

    def outgoing(self, node_identifier):
        pass

    def incoming(self, node_identifier):
        pass


class GraphAdjacencyList(Graph):
    """
    Graph implemented using outgoing/incoming lists.
    """

    def __init__(self):
        super().__init__()
        self._outgoing = {}
        self._incoming = {}

    def add_node(self, node_identifier):
        self._outgoing[node_identifier] = []
        self._incoming[node_identifier] = []

    def add_edge(self, a, b, value):
        self._outgoing[a].append((b, value))
        self._incoming[b].append((a, value))

    def outgoing(self, node_identifier):
        """
        Collects the nodes to which the specified node connects.

        :param node_identifier:  Index of the node in the matrix.
        :return:                 A list of tuples (id, value) representing the outgoing edge values.
        """
        return self._outgoing[node_identifier]

    def incoming(self, node_identifier):
        """
        Collects the nodes that connect to the specified node.

        :param node_identifier:  Index of the node in the matrix.
        :return:                 A list of tuples (id, value) representing the incoming edge values.
        """

        return self._incoming[node_identifier]


class GraphAdjacencyMatrix(Graph):
    """
    Graph implemented using an adjacency matrix.
    """

    def __init__(self, resize_amount=1):
        super().__init__()
        self._n = 0
        self._resize_amount = resize_amount
        self._adj = np.zeros(shape=(resize_amount, resize_amount), dtype=np.int)
        self._id_index_map = {}

    def _resize(self):
        self._adj = np.append(self._adj, [0] * self._resize_amount, 0)
        self._adj = np.append(self._adj, [[0]] * self._resize_amount, 1)

    def add_node(self, node_identifier):
        # Add first, resize later.
        self._n += 1

        if self._adj.shape[0] <= self._n:
            self._resize()

        # Register the new index
        self._id_index_map[node_identifier] = self._n

    def add_edge(self, a, b, value):
        a_ix = self._id_index_map[a]
        b_ix = self._id_index_map[b]

        self._adj[a_ix, b_ix] = value

    def outgoing(self, node_identifier):
        """
        Collects the nodes to which the specified node connects.

        :param node_identifier:  Index of the node in the matrix.
        :return:                 A list of tuples (id, value) representing the outgoing edge values.
        """

        ix = self._id_index_map[node_identifier]

        nodes = self._adj[ix, :self._n].tolist()
        return [(i, nodes[i]) for i in range(len(nodes)) if nodes[i] != 0]

    def incoming(self, node_identifier):
        """
        Collects the nodes that connect to the specified node.

        :param node_identifier:  Index of the node in the matrix.
        :return:                 A list of tuples (id, value) representing the incoming edge values.
        """

        ix = self._id_index_map[node_identifier]
        nodes = self._adj[:self._n, ix].tolist()
        return [(i, nodes[i]) for i in range(len(nodes)) if nodes[i] != 0]
