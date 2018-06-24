class Optimizer:
    def __init__(self):
        self.backend = None

    def get_variable_values(self):
        raise NotImplementedError()

    def optimize(self, graph, variable_ids, loss_id, feed_dict, variable_init_feed_dict, constant_feed_dict):
        raise NotImplementedError()
