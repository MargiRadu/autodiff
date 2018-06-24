import time
import sys
import dill as pickle

from math import ceil

from twisted.python import log
from twisted.internet import reactor
from twisted.internet.protocol import Protocol
from twisted.internet.endpoints import TCP4ClientEndpoint, connectProtocol

from threading import Lock

from optimization.optimizer import Optimizer
from optimization.local_iterative_optimizer import MiniBatchSGD


class DownpourMasterProtocol(Protocol):

    def __init__(self, protocol_id, grad_callback, init_callback, release_callback, no_steps):
        super().__init__()

        # Callbacks
        self.gradient_update_callback = grad_callback
        self.init_callback = init_callback
        self.release_callback = release_callback

        # State
        self.pid = protocol_id
        self.steps = no_steps

    def dataReceived(self, data):
        msg_type, msg_data = pickle.loads(data)

        if msg_type == 'set:model-ack':
            log.msg('Model successfully set.')
            self.init_callback(self.pid)

        elif msg_type == 'run:optimize-ack':
            #log.msg('Optimization round over.')
            self.gradient_update_callback(self.pid, msg_data)

        elif msg_type == 'set:release-ack':
            log.msg('Worker sucessfully released.')
            #self.transport.loseConnection()
            self.release_callback(self.pid)

        else:
            raise NotImplementedError('Unrecognized message type: {}'.format(msg_type))

    def set_model(self, optimizer, graph, variable_ids, loss_id, feed_dict_part, variable_init_feed_dict,
                  constant_feed_dict):
        msg_type = 'set:model'
        msg_data = optimizer, graph, variable_ids, loss_id, feed_dict_part, variable_init_feed_dict, constant_feed_dict

        self.send_message(msg_type, msg_data)

    def run_optimizer(self, global_variables):
        msg_type = 'run:optimize'
        msg_data = (global_variables, self.steps)

        self.send_message(msg_type, msg_data)

    def release_worker(self):
        msg_type = 'set:release'
        msg_data = None

        self.send_message(msg_type, msg_data)

    def send_message(self, msg_type, msg_data):
        self.transport.write(pickle.dumps((msg_type, msg_data)))


class DownpourSGD(Optimizer):
    def __init__(self, worker_addresses, learning_rate, epsilon, batch_size=32, max_iterations=1000, steps=5,
                 backend=None):
        super().__init__()

        # Application info.
        self.optimizer_lr = learning_rate
        self.optimizer_eps = epsilon
        self.optimizer_batch = batch_size
        self.optimizer_max_it = max_iterations
        self.backend = backend
        self.steps = steps

        # Distributed state
        self.iterations = 0
        self.protocol_initialized = [False] * len(worker_addresses)
        self.converged = [False] * len(worker_addresses)
        self.released = [False] * len(worker_addresses)
        self.variable_state = None

        self.worker_addresses = worker_addresses

        # TODO: use different locks...
        self.master_lock = Lock()
        self.protocols = []

        # Log
        log.startLogging(sys.stdout)
        log.msg('Starting Downpour SGD Master...')

    def _register_protocol(self, protocol, graph, variable_ids, loss_id, feed_dict_part, variable_init_feed_dict,
                           constant_feed_dict):
        """
        Registers the protocol and initializes the worker optimizer.
        :param protocol:    The protocol to the worker.
        """

        # Register the protocol.
        with self.master_lock:
            self.protocols.append(protocol)

        # Init the optimizer.
        # TODO: would be cool to have a local optimizer factory for this use case.
        # The factory would be sent as parameter to the master constructor
        optimizer = MiniBatchSGD(self.optimizer_lr, self.optimizer_eps, self.optimizer_batch, self.optimizer_max_it,
                                 self.backend)
        protocol.set_model(optimizer, graph, variable_ids, loss_id, feed_dict_part, variable_init_feed_dict,
                           constant_feed_dict)

    def grad_update_successful_callback(self, pid, optimized_params):
        """
        A callback that will be called by the protocol when optimized parameter values are received.
        :param pid:                 Id of the protocol. Required for successive optimization request.
        :param optimized_params:    The parameters obtained from the optimizer.
        """

        # Update global variable state under lock.
        global_values = {}
        with self.master_lock:
            self.iterations += self.steps
            for var, new_value in optimized_params.items():
                old_value = self.variable_state[var]
                global_values[var] = old_value + ((new_value - old_value) / len(self.protocols))
                self.variable_state[var] = global_values[var]

        # Check end of optimization todo: or convergence.
        if self.iterations <= self.optimizer_max_it:
            # Keep going by requesting a new round of optimization
            self.protocols[pid].run_optimizer(global_values)
        else:
            with self.master_lock:
                self.converged[pid] = True

            self.protocols[pid].release_worker()

    def worker_init_successful_callback(self, pid):
        with self.master_lock:
            self.protocol_initialized[pid] = True

        # Just make the first call to optimize
        self.protocols[pid].run_optimizer(self.variable_state)

    def worker_released_callback(self, pid):
        with self.master_lock:
            self.released[pid] = True
            if all(self.released):
                reactor.stop()

    def optimize(self, graph, variable_ids, loss_id, feed_dict, variable_init_feed_dict, constant_feed_dict):
        """
        Runs the distributed optimizer.
        """

        with self.master_lock:
            self.variable_state = variable_init_feed_dict

        # Establish connections to workers
        endpoints = [TCP4ClientEndpoint(reactor, w[0], w[1]) for w in self.worker_addresses]

        for pid, endpoint in enumerate(endpoints):
            protocol = DownpourMasterProtocol(protocol_id=pid,
                                              grad_callback=self.grad_update_successful_callback,
                                              init_callback=self.worker_init_successful_callback,
                                              release_callback=self.worker_released_callback,
                                              no_steps=self.steps)
            d = connectProtocol(endpoint, protocol)

            # Partition the data:
            feed_dict_part = _extract_feed_dict_partition(feed_dict, pid, len(self.worker_addresses))
            d.addCallback(lambda p: self._register_protocol(p, graph, variable_ids, loss_id, feed_dict_part,
                                                            variable_init_feed_dict, constant_feed_dict))
        reactor.run()

    def get_variable_values(self):
        return self.variable_state

    def has_converged(self):
        with self.master_lock:
            did_converge = all(self.converged)

        return did_converge


def _extract_feed_dict_partition(feed_dict, i, p):
    # Find the minimum of the feed dict
    n = min([len(data) for data in feed_dict.values()])

    # Compute partition length
    partition_size = ceil(n / p)

    # Compute the range (inclusive, exclusive)
    r_low = partition_size * i
    r_high = partition_size * (i + 1)

    # Extract the data
    return {k: v[r_low:r_high] for k, v in feed_dict.items()}
