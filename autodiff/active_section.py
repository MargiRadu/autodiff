from itertools import count
from singleton_decorator import singleton
from autodiff.graph import GraphAdjacencyList


def active_section():
    return ActiveSection()


def init_active_section():
    a = ActiveSection()
    a.reset_section()
    return a


@singleton
class ActiveSection:

    def __init__(self):
        self._node_id_counter = count(0, 1)

        # Node info
        self.graph = GraphAdjacencyList()
        self.graph_ops_map = {}
        self.op_id_set = set()
        self.variables = []

        # Optimizer
        self.optimizer = None

        # Backend
        self.backend = None

        # Loss node
        self.loss_id = None

    def __enter__(self):
        self.reset_section()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True

    def reset_section(self):
        self._node_id_counter = count(0, 1)
        self.graph = GraphAdjacencyList()
        self.graph_ops_map = {}
        self.op_id_set = set()
        self.variables = []
        self.constants = {}

    def next_id(self):
        return next(self._node_id_counter)

    def register_node(self, graph_node):
        graph_node_id = graph_node.identifier
        op_id = graph_node.op_id

        self.graph.add_node(graph_node_id)
        self.graph_ops_map[graph_node_id] = op_id
        self.op_id_set.add(op_id)
        for incoming_node in graph_node.incoming:
            self.graph.add_edge(incoming_node.identifier, graph_node_id, 1)

        # When dealing with a probe/variable, register its reference so that it can be updated after optimization.
        if op_id == 'variable':
            self.variables.append(graph_node)

        if op_id == 'constant':
            self.constants[graph_node_id] = graph_node.extra_info

        # When dealing with a loss node, register it as the loss node.
        if op_id == 'loss':
            self.register_loss(graph_node)

    def register_backend(self, backend):
        self.backend = backend

        if self.optimizer is not None:
            if self.optimizer.gradients_f is None:
                self.optimizer.gradients_f = self.backend.gradients

    def register_optimizer(self, optimizer):
        self.optimizer = optimizer

        if self.backend is not None:
            self.optimizer.gradients_f = self.backend.gradients

    def register_loss(self, loss_node):
        if self.loss_id is None:
            self.loss_id = loss_node.identifier
        else:
            raise ValueError('Cannot define more than one loss function per model.')

    def flatten_graph(self):
        graph_dict = {}
        for node_id in self.graph_ops_map.keys():
            op = self.graph_ops_map[node_id]
            input_ids = [input_id for input_id, _ in self.graph.incoming(node_id)]
            output_ids = [output_id for output_id, _ in self.graph.outgoing(node_id)]
            graph_dict[node_id] = {
                'op': op,
                'input_ids': input_ids,
                'output_ids': output_ids
            }
        return graph_dict

    def optimize_model(self, feed_dict):
        if (self.optimizer is None) or (self.backend is None) or (len(self.variables) == 0) or (self.loss_id is None):
            raise AttributeError('Incomplete definition of model. Missing loss/optimizer/backend/variables.')

        # Translate graph into backend-compatible-graph.
        # TODO: Use a graph hash to enable caching in the backend.
        flattened_graph = self.flatten_graph()

        # Initialize backend
        needed_ops = set(self.graph_ops_map.values()).difference({'loss'})
        self.backend.init_capabilities(needed_ops)

        # Call optimizer using selected backend
        self.optimizer.optimize(graph=flattened_graph,
                                variable_ids=[v.identifier for v in self.variables],
                                loss_id=self.loss_id,
                                feed_dict={k.identifier: v for k, v in feed_dict.items()},
                                variable_init_feed_dict={v.identifier: 0.0 for v in self.variables},
                                constant_feed_dict=self.constants)

        # Update variable values.
        var_values = self.optimizer.get_variable_values()
        for v in self.variables:
            v.extra_info = var_values[v.identifier]

    def eval(self, node, feed_dict):
        if (self.optimizer is None) or (self.backend is None) or (len(self.variables) == 0) or (self.loss_id is None):
            raise AttributeError('Incomplete definition of model. Missing loss/optimizer/backend/variables.')

        flattened_graph = self.flatten_graph()

        # Initialize backend
        needed_ops = set(self.graph_ops_map.values()).difference({'loss'})
        self.backend.init_capabilities(needed_ops)

        # Call optimizer using selected backend
        return self.backend.values(graph=flattened_graph,
                                   feed_dict={k.identifier: v for k, v in feed_dict.items()},
                                   target_node_id=node.identifier,
                                   variable_feed_dict={v.identifier: v.extra_info for v in self.variables},
                                   constant_feed_dict=self.constants)
