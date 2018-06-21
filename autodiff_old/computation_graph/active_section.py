from autodiff_old.engine.container import container



# def _register(op, name, collection):
#         if name not in collection:
#             collection = op
#         else:
#             # TODO: Can use CoR to propagate this to outer sections and simulate scope.
#             raise ValueError('Opeartion name already exists in current section')
#
#
# class ActiveSection:
#
#     """
#     """
#
#
#     def __init__(self):
#
#         self._inputs = {}
#         self._outputs = {}
#         self._probes = {}
#
#         self._feeds = {}
#
#         ioc = container()
#         ioc.register(self)
#
#     def register_input(self, op, name):
#         _register(op, name, self._inputs)
#
#     def register_output(self, op, name):
#         _register(op, name, self._outputs)
#
#     def register_probe(self, op, name):
#         _register(op, name, self._probes)
#
#     def feed(self, input_name, values):
#         if input_name not in self._inputs:
#             raise ValueError('No such input: {}.'.format(input_name))
#
#         self._feeds[input_name] = values
#
#
#
#     def close(self):
#         ioc = container()
#         ioc.unregister(self)


class ActiveSection:

    def __init__(self, backend=None):
        self._backend = backend

    def erm(self, feed_dict):
        """
        Optimizes the parameters of the current model using the active optimizer and loss on the provided data.

        :param feed_dict:   Dictionary specifying inputs: Operation -> float array
        """

        # Get graph.
        comp_graph = request_graph()

        # Get initialization policy.
        init_policy = request_init_policy()

        # Get optimizer.
        optimizer = request_optimizer()

        # Get the ids of the output(loss)/inputs.
        loss_id = request_loss_id()
        probe_ids = request_probe_ids()
        id_feed_map = [request_id(op): feed_dict[op] for op in feed_dict.keys()]

        # Request computation. TODO: call by param name, not position.
        results = self._backend.erm(comp_graph, optimizer, init_policy, loss_id, probe_ids, id_feed_map)

        # Handle results: set optimizer state, update variables on graph, set probes values, intermediate loss values,
        update_optimizer_states(results['optimizer'])
        update_variables(results['variables'])
        update_probes(results['probes'])
        update_loss_states(results['loss'])



    def evaluate(self, op, feed_dict):
        """
        Evaluates the specified operation using the specified inputs.

        :param op:          Operation object.
        :param feed_dict:   Dictionary specifying inputs: Operation -> float array
        :return:            Float array of results.
        """

        # Get the graph.
        comp_graph = request_graph()

        # Get the ids of the output/inputs.
        op_id = request_id(op)
        id_feed_map = [request_id(op): feed_dict[op] for op in feed_dict.keys()]

        # Request computation.
        return self._backend.real_eval(comp_graph, op_id, id_feed_map)
