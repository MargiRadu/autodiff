
class IterativeOptimizer:

    def __init__(self, gradients_f):
        self.gradients_f = gradients_f
        self.variable_info = {}
        self.feed_dict = None
        self.feed_length = 0
        self.constant_feed_dict = None
        self.loss_id = None
        self.variable_ids = None
        self.graph = None

    def get_variable_values(self):
        return {k: v['current_value'] for k, v in self.variable_info.items()}

    def update_rule(self, vid, gradients):
        raise NotImplementedError()

    def init_variable(self, vid, init_value):
        raise NotImplementedError()

    def has_converged(self):
        raise NotImplementedError()

    def gather_gradients(self, graph, variable_ids, loss_id):
        raise NotImplementedError()

    def make_variable_feed_dict(self):
        return {k: v['current_value'] for k, v in self.variable_info.items()}

    def init_optimizer(self, graph, feed_dict, constant_feed_dict, variable_ids, loss_id):
        self.constant_feed_dict = constant_feed_dict
        self.feed_dict = feed_dict
        self.feed_length = min([len(vs) for vs in feed_dict.values()])
        self.loss_id = loss_id
        self.variable_ids = variable_ids
        self.graph = graph

    def optimize_step(self, variable_feed_dict, iterations):
        self.variable_info = {}
        for vid, init_value in variable_feed_dict.items():
            self.init_variable(vid, init_value)

        for _ in range(iterations):
            vid_grad_map = self.gather_gradients(self.graph, self.variable_ids, self.loss_id)
            for vid, grads in vid_grad_map.items():
                self.update_rule(vid, grads)

        return self.get_variable_values()

    def optimize(self, graph, variable_ids, loss_id, feed_dict, variable_init_feed_dict, constant_feed_dict):
        self.constant_feed_dict = constant_feed_dict
        self.feed_dict = feed_dict
        self.feed_length = min([len(vals) for vals in feed_dict.values()])

        # TODO: extend to a initialization policy (maybe upstream, tho)
        self.variable_info = {}
        for vid, init_value in variable_init_feed_dict.items():
            self.init_variable(vid, init_value)

        while not self.has_converged():
            vid_grad_map = self.gather_gradients(graph, variable_ids, loss_id)
            for vid, grads in vid_grad_map.items():
                self.update_rule(vid, grads)


class MiniBatchSGD(IterativeOptimizer):

    def __init__(self, learning_rate, epsilon, batch_size=32, max_iterations=1000, gradients_f=None):
        super().__init__(gradients_f)

        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.current_batch_ix = 0

    def update_rule(self, vid, gradient):
        var_info = self.variable_info[vid]
        if not var_info['converged']:
            var_info['iterations'] += 1

            old_value = var_info['current_value']
            new_value = old_value - (self.learning_rate * gradient)
            var_info['current_value'] = new_value

            if (abs(new_value - old_value) <= self.epsilon) or (var_info['iterations'] >= self.max_iterations):
                var_info['converged'] = True

    def init_variable(self, vid, init_value):
        self.variable_info[vid] = {
            'current_value': init_value,
            'iterations': 0,
            'converged': False
        }

    def has_converged(self):
        return all([info['converged'] for info in self.variable_info.values()])

    def next_batch(self):

        start_ix = self.current_batch_ix
        end_ix = start_ix + self.batch_size

        if end_ix >= self.feed_length:
            self.current_batch_ix = 0
        else:
            self.current_batch_ix = end_ix

        return {fid: vals[start_ix:end_ix]
                for fid, vals in self.feed_dict.items()}

    def gather_gradients(self, graph, variable_ids, loss_id):
        next_batch = self.next_batch()
        variable_feed_dict = self.make_variable_feed_dict()

        return self.gradients_f(graph=graph,
                                target_node_id=loss_id,
                                feed_dict=next_batch,
                                variable_feed_dict=variable_feed_dict,
                                constant_feed_dict=self.constant_feed_dict,
                                reduce_strategy='avg')
