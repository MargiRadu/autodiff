import sys
import dill as pickle

from twisted.python import log
from twisted.internet import reactor
from twisted.internet.protocol import ServerFactory, Protocol


class DownpourWorkerServerProtocol(Protocol):

    def __init__(self, set_model_callback, optimize_callback, release_callback):
        super().__init__()
        self.set_model_callback = set_model_callback
        self.optimize_callback = optimize_callback
        self.release_callback = release_callback

    def dataReceived(self, data):
        msg_type, msg_data = pickle.loads(data)

        if msg_type == 'set:model':
            self.set_model_callback(msg_data)
            log.msg('Successfully set model and optimizer.')
            self.transport.write(pickle.dumps(('set:model-ack', None)))

        elif msg_type == 'run:optimize':
            # Data is a tuple (model parameters, iterations). Run the set variables callback
            # Run the optimizer callback and send the result.
            new_variable_feed_dict = self.optimize_callback(msg_data[0], msg_data[1])
            self.transport.write(pickle.dumps(('run:optimize-ack', new_variable_feed_dict)))

        elif msg_type == 'set:release':
            self.release()
            self.transport.write(pickle.dumps(('set:release-ack', None)))

        else:
            raise NotImplementedError('Unrecognized message type: {}'.format(msg_type))

    def connectionMade(self):
        log.msg('Client connection from {}'.format(self.transport.getPeer()))

    def connectionLost(self, reason):
        log.msg('Lost connection because {}'.format(reason))

    def run_optimizer(self, iterations):
        return self.optimizer.optimize_step(self.variable_feed_dict, iterations)

    def release(self):
        self.release_callback()
        self.transport.loseConnection()


class DownpourWorkerServerFactory(ServerFactory):

    def __init__(self, set_model_callback, optimize_callback, release_callback):
        super().__init__()
        self.set_model_callback = set_model_callback
        self.optimize_callback = optimize_callback
        self.release_callback = release_callback

    def buildProtocol(self, addr):
        return DownpourWorkerServerProtocol(self.set_model_callback, self.optimize_callback, self.release_callback)


class DownpourWorkerServer:
    def __init__(self, port):
        # Model info.
        self.graph = None
        self.variable_ids = None
        self.loss_id = None
        self.feed_dict = None
        self.variable_feed_dict = None
        self.constant_feed_dict = None
        self.optimizer = None

        log.startLogging(sys.stdout)
        log.msg('Starting Downpour SGD worker...')

        factory = DownpourWorkerServerFactory(self.set_model, self.optimize, self.release)
        reactor.listenTCP(port, factory)
        reactor.run()

    def set_model(self, packed_data):
        """
        Sets the model parameters.

        :param packed_data:  A tuple of (opt, graph, var_ids, loss_id, feed_dict, variable_feed_dict, constant_feed_dict)
        """
        self.optimizer = packed_data[0]
        graph = packed_data[1]
        variable_ids = packed_data[2]
        loss_id = packed_data[3]
        feed_dict = packed_data[4]
        variable_feed_dict = packed_data[5]
        constant_feed_dict = packed_data[6]

        self.optimizer.init_optimizer(graph, feed_dict, constant_feed_dict, variable_ids, loss_id)

    def optimize(self, variable_feed, iterations):
        return self.optimizer.optimize_step(variable_feed, iterations)

    def release(self):
        self.optimizer = None
