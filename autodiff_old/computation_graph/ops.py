from autodiff_old.dual_number.structure import DualNumber
from autodiff_old.dual_number.ops import *

from autodiff_old.computation_graph.graph import Graph


class NodeNameGenerator:
    """
    Simple name generator that generates unique names for provided items.
    """

    def __init__(self):
        self.name_counter_map = {}

    @staticmethod
    def make_string(item_name, counter):
        return '{}_{}'.format(item_name, counter)

    def get_name(self, item_name):
        if item_name not in self.name_counter_map:
            self.name_counter_map[item_name] = 0

        # Register and increment.
        counter = self.name_counter_map[item_name]
        self.name_counter_map[item_name] += 1

        return NodeNameGenerator.make_string(item_name, counter)

    def __getitem__(self, item):
        return self.get_name(item)


class Operation:

    def __init__(self):
        self._incoming = []
        self._outgoing = []

    def eval(self):
        raise NotImplementedError

    def local_eval(self):
        return self.eval().real

    def register_child(self, op):
        self._outgoing.append(op)


class OneInputOperation(Operation):

    def __init__(self, a, op):
        super().__init__()
        self.incoming = [a]
        self._op = op

        a.register_child(self)

    def eval(self):
        return self._op(self.incoming[0].eval())


class TwoInputOperation(Operation):

    def __init__(self, a, b, op):
        super().__init__()
        self.incoming = [a, b]
        self._op = op

        a.register_child(self)
        b.register_child(self)

    def eval(self):
        return self._op(self.incoming[0].eval(), self.incoming[1].eval())


class FeederNew(Operation):

    def __init__(self):
        super().__init__()
        self._active = False

    def feed(self, values):
        # TODO: Register as feed to AS
        pass

    # These 2 should be package protected and available only to the AS.
    # The user is not supposed to use these methods.
    def set_active_state(self, active_state):
        self._active = active_state

    def set_values(self, values):
        # TODO: we need to retrieve the values from the AS somehow. The AS should be responsible for
        # what kind of data we receive. Ideally the same data, but it might run several sweeps over the
        # graph so that we can't afford to lose the values through iteration.
        # In other words, the control responsibility is passed over to the AS. This class only needs
        # to be able to compute the dual value.

        # On the other hand, we don't want to request every single value to the AS due to the overhead.
        # Also, the AS might be remote. What we want is a way to receive data from the AS and iterate freely through it.

        self._values = values

    def eval(self):
        # TODO: what happens if we dont have any values set (as is the case for calling eval on the graph nodes)?
        # One idea: have a mechanism that enables the AS to take control over the feeding mechanism.
        # Two idea: have two functions: _dual_eval - called by AS, and local_eval - called by the user.
        #           The local stands for local evaluation, while the other option would be to call eval on the AS whch can run remotely.
        if self._active:
            return dual_variable(next(self._values))
        else:
            return dual_constant(next(self._values))


class Variable(Operation):

    def __init__(self):
        super().__init__()

        self._value = dual_constant(0)

    def initialize(self, init_value):
        self._value = init_value

    def eval(self):
        if self._value is None:
            self._value = 2# TODO: this lel request_initial_value()
        return self._value

    def update(self, new_dual_value):
        # TODO: 1. should we update using a dual value?
        # TODO: 2. should we overload the assignment operator?
        self._value = new_dual_value


class Constant(Operation):

    def __init__(self, value):
        super().__init__()
        self._constant = dual_constant(value)

    def eval(self):
        return self._constant


class Probe(OneInputOperation):
    """
    A Probe is an abstraction that guarantees the recording of the values for the attached node during computation.
    """

    def __init__(self, a):


class Add(TwoInputOperation):

    def __init__(self, a, b):
        super().__init__(a, b, dual_add)


class Sub(TwoInputOperation):

    def __init__(self, a, b):
        super().__init__(a, b, dual_sub)


class Mul(TwoInputOperation):

    def __init__(self, a, b):
        super().__init__(a, b, dual_mul)
