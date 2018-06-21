from autodiff.active_section import active_section


class GraphNode:
    def __init__(self, op_id, incoming, extra_info=None):
        self.op_id = op_id
        self.incoming = incoming
        self.extra_info = extra_info

        section = active_section()
        self.identifier = section.next_id()
        section.register_node(self)

    def _handle_two_input_op(self, other, op_id):
        if type(other) != GraphNode:
            other = constant(other)

        return GraphNode(op_id, [self, other])

    def __add__(self, other):
        return self._handle_two_input_op(other, 'add')

    def __mul__(self, other):
        return self._handle_two_input_op(other, 'mul')

    def __sub__(self, other):
        return self._handle_two_input_op(other, 'sub')

    def __truediv__(self, other):
        return self._handle_two_input_op(other, 'div')

    # TODO: more ops!!


def constant(value):
    return GraphNode('constant', [], extra_info=value)


def variable(init_value):
    return GraphNode('variable', [], extra_info=init_value)


def feeder():
    return GraphNode('feeder', [])


def loss(node):
    return GraphNode('loss', [node])
