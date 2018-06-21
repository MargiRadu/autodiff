import socket


class DownpourWorkerClient:
    """
    Represents a client for a worker node in the Downpour SGD optimizer.
    """

    def __init__(self, ip, port):
        self.sock = socket.socket()
        self.sock.connect((ip, port))
        # TODO: Send graph, data, etc.

    def terminate(self):
        self.sock.close()

    def _send_graph(self, graph):
        pass

    def _send_variable_feed(self, variable_feed):
        pass

    def _send_constant_feed(self, variable_feed):
        pass

    def _send_feeder_feed(self, variable_feed):
        pass

    def _send_loss_id(self, loss_id):
        pass

    def run_optimizer(self, variable_feed, iterations=1):
        pass


class DownpourWorkerServer:
    """
    Represents a worker node (server) in the Downpour SGD framework.
    """

    def __init__(self, ip, port):
        self.sock = socket.socket()
        self.sock.bind((ip, port))

    def start(self):
        pass

    def terminate(self):
        self.sock.close()

    def parse_message(self, m):
        pass

    def run_optimizer(self, variable_feed):
        pass
