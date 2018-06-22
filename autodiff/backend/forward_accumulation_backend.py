import numpy as np
from autodiff.backend.dual_number_engine import DualNumberEngine


class ForwardAccumulationBackend:
    """
        Local backend that implements forward accumulation for gradient computation.
        The forward accumulation algorithm implemented:

        For each variable v:
            dual(v) = 1
            dualTargetNode = ComputeDualGraph(targetNode)
            gradient(v) = dual(dualTargetNode)
            dual(v) = 0

        return gradient
    """

    def __init__(self):
        self.engine = None
        self.name = 'forward-acc'

    def init_capabilities(self, ops_set):
        self.engine = DualNumberEngine(ops_set)

    def values(self, graph, target_node_id, feed_dict, variable_feed_dict, constant_feed_dict):
        """
        Computes the values for the specified node using forward accumulation.

        :param graph:               Computation graph. i.e. dict mapping node ids to <op name, input ids, output ids>.
        :param target_node_id:      Node for which to compute the value.
        :param feed_dict:           Dict mapping node ids to a list of input values.
        :param variable_feed_dict:  Dict mapping variable ids to their current values.
        :param constant_feed_dict:  Dict mapping constant ids to their constant values.
        :return:                    A list of values for all input values.
        """

        variable_id = list(variable_feed_dict.keys())[0]
        node_duals = self.batch_forward_sweep(graph, variable_id, constant_feed_dict, variable_feed_dict, feed_dict)
        target_values = [d.real for d in node_duals[target_node_id]]

        return target_values

    def gradients(self, graph, target_node_id, feed_dict, variable_feed_dict, constant_feed_dict, reduce_strategy):
        """
        Computes the gradients for the variable nodes using forward accumulation.

        :param graph:               Computation graph. i.e. dict mapping node ids to <op name, input ids, output ids>.
        :param target_node_id:      Node for which to compute the derivative.
        :param feed_dict:           Dict mapping node ids to a list of input values.
        :param variable_feed_dict:  Dict mapping variable ids to their current values.
        :param constant_feed_dict:  Dict mapping constant ids to their constant values.
        :param reduce_strategy:     Method of processing a set of gradients.
                                    Supports <avg>, <median>, <None> - returns list of gradients.
        :return:                    A map that connects variable ids to
                                    a gradient or list of gradients (if reduce strategy is set to None).
        """

        # Select reduce strategy based on parameter:
        if reduce_strategy == 'avg':
            reducer = np.mean
        elif reduce_strategy == 'median':
            reducer = np.median
        elif reduce_strategy is None:
            def _nop(x): return x
            reducer = _nop
        else:
            raise NotImplementedError('Reduce strategy {} is not implemented.'.format(reduce_strategy))

        variable_gradient_map = {}
        for variable_id in variable_feed_dict.keys():
            node_duals = self.batch_forward_sweep(graph, variable_id, constant_feed_dict, variable_feed_dict, feed_dict)
            target_gradients = [d.dual for d in node_duals[target_node_id]]

            variable_gradient_map[variable_id] = reducer(target_gradients)

        return variable_gradient_map

    def batch_forward_sweep(self, graph, active_variable_id, constant_feed, variable_feed, feeder_batch):
        """
        Performs a series of sweeps of the computation graph, one for each input configuration of the feeders.

        :param graph:                   The computation graph.
        :param active_variable_id:      The id of the current variable considered for differentiation.
        :param constant_feed:           A dict mapping constant ids to their constant value.
        :param variable_feed:           A dict mapping variable ids to their current value.
        :param feeder_batch:            A dict mapping feeder ids to a list of their input values.
        :return:                        A dict mapping all graph node ids to a list of resulting dual numbers.
        """

        # Pick the minimum of the feed lists to represent the entire feed dict.
        n = min([len(vals) for feeder_id, vals in feeder_batch.items()])

        # Build a list of feed dicts.
        feed_dicts = [{k: v[i] for k, v in feeder_batch.items()} for i in range(n)]

        # Gather results from consecutive forward sweeps.
        sweep_results = [self.sweep(graph, active_variable_id, constant_feed, variable_feed, feed_dicts[i])
                         for i in range(n)]

        # Build lists of dual numbers for each node.
        graph_dual_map = {}
        for graph_result in sweep_results:
            for k, v in graph_result.items():
                if k not in graph_dual_map:
                    graph_dual_map[k] = []

                graph_dual_map[k].append(v)

        return graph_dual_map

    def sweep(self, graph, active_variable_id, constant_feed, variable_feed, feeder_feed):
        """
        Performs a forward sweep of the computation graph. The function returns the dual values for the entire
        computation graph s.t. it can be used for both gradient computation and function evaluation.

        :param graph:                   The computation graph.
        :param active_variable_id:      The id of the current variable considered for differentiation.
        :param constant_feed:           A dict mapping constant ids to their constant value.
        :param variable_feed:           A dict mapping variable ids to their current value.
        :param feeder_feed:             A dict mapping feeder ids to their input value.
        :return:                        A dict mapping all graph node ids to their resulting dual numbers.
        """

        # Used to mark variables/constants/feeders as ready from the start (i.e. automatically add them to the queue.
        graph_input_ids = set(constant_feed.keys()).union(set(variable_feed.keys())).union(set(feeder_feed.keys()))

        # We store the computed node values here.
        node_values = {}

        # In forward sweeps, we must consider all inputs to the graph.
        eval_queue = list(graph_input_ids)
        while len(eval_queue) > 0:
            node_id = eval_queue[0]
            node = graph[node_id]
            eval_queue = eval_queue[1:]

            if node_id not in node_values.keys():
                # A node is ready when all it's incoming edges have been evaluated.
                node_is_ready = (node_id in graph_input_ids) or \
                                (all([incoming_id in node_values for incoming_id in node['input_ids']]))

                if node_is_ready:
                    # Evaluate the node
                    if node['op'] == 'feeder':
                        node_values[node_id] = self.engine.do('feeder', feeder_feed[node_id])
                    elif node['op'] == 'constant':
                        node_values[node_id] = self.engine.do('constant', constant_feed[node_id])
                    elif node['op'] == 'variable':
                        # Set the seed of the variable.
                        seed = int(node_id == active_variable_id)
                        node_values[node_id] = self.engine.do('variable', variable_feed[node_id], seed)
                    else:
                        # Generic op. Gather incoming values and eval.
                        incoming_values = [node_values[input_id] for input_id in node['input_ids']]
                        node_values[node_id] = self.engine.do(node['op'], *incoming_values)

                    # Enqueue nodes downstream.
                    eval_queue += node['output_ids']

                else:
                    # Re-enqueue the node's dependencies.
                    eval_queue += node['input_ids']

        return node_values
