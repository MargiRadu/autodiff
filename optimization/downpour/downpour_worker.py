import sys
import dill as pickle

from twisted.python import log
from twisted.internet import reactor
from twisted.internet.protocol import ServerFactory, ClientFactory, Protocol


""" 
    TODO: 
    This is the actual optimizer object. 
    Super important - the optimizer should have the gradients_f function set to the one of the backend 
        by the Master, which in turn receives it from the active section.
"""
class DownpourWorkerServerProtocol(Protocol):

    def __init__(self, set_optimizer_callback, set_variables_callback, optimize_callback):
        super().__init__()
        self.set_optimizer_callback = set_optimizer_callback
        self.set_variables_callback = set_variables_callback
        self.optimize_callback = optimize_callback

    def dataReceived(self, data):
        msg_type, msg_data = pickle.loads(data)

        if msg_type == 'set:variables':
            self.set_variables_callback(msg_data)
            self.transport.write(pickle.dumps(('set:variables-ack', None)))
        elif msg_type == 'set:optimizer':
            self.set_optimizer_callback(msg_data)
            self.transport.write(pickle.dumps(('set:optimizer-ack', None)))
        elif msg_type == 'run:optimize':
            # Data is a tuple (model parameters, iterations). Run the set variables callback
            self.set_variables_callback(msg_data[0])

            # Run the optimizer callback and send the result.
            new_variable_feed_dict = self.optimize_callback(msg_data[1])
            self.transport.write(pickle.dumps(('run:optimize-ack', new_variable_feed_dict)))
        else:
            raise NotImplementedError('Unrecognized message type: {}'.format(msg_type))

    def connectionMade(self):
        log.msg('Client connection from {}'.format(self.transport.getPeer()))

    def connectionLost(self, reason):
        log.msg('Lost connection because {}'.format(reason))

    def run_optimizer(self, iterations):
        return self.optimizer.optimize_step(self.variable_feed_dict, iterations)


class DownpourWorkerServerFactory(ServerFactory):

    def __init__(self, set_optimizer_callback, set_variables_callback, optimize_callback):
        super().__init__()
        self.set_optimizer_callback = set_optimizer_callback
        self.set_variables_callback = set_variables_callback
        self.optimize_callback = optimize_callback

    def buildProtocol(self, addr):
        return DownpourWorkerServerProtocol(self.set_optimizer_callback, self.set_variables_callback, self.optimize_callback)


class DownpourWorkerServer:
    def __init__(self, port):
        # Model info.
        self.variable_feed_dict = None
        self.optimizer = None

        log.startLogging(sys.stdout)
        log.msg('Starting Downpour SGD worker...')

        reactor.listenTCP(port, DownpourWorkerServerFactory(self.init_optimizer, self.set_variables, self.optimize))
        reactor.run()

    def init_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_variables(self, variable_feed_dict):
        self.variable_feed_dict = variable_feed_dict

    def optimize(self, iterations):
        return self.optimizer.optimize_step(self.variable_feed_dict, iterations)