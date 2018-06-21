class DualNumberEngine:
    def __init__(self, ops_set=None):
        self.ops = {
            'constant': _dual_constant,
            'variable': _dual_variable,
            'feeder': _dual_constant,
            'loss': _dual_identity,
            'add': _dual_add,
            'sub': _dual_sub,
            'mul': _dual_mul,
            'sqrt': _dual_sqrt,
            'pow': _dual_pow,
            'div': _dual_div,
            'logistic': _dual_logistic
        }

        if ops_set is not None:
            set_diff = ops_set.difference(self.ops.keys())
            if len(set_diff) > 0:
                raise ValueError('Operations not implemented in number engine: {}'.format(set_diff))

    def do(self, op, *args):
        return self.ops[op](*args)

class DualNumber:

    def __init__(self, real, dual):
        self.real = real
        self.dual = dual

    def __add__(self, other):
        if isinstance(other, DualNumber):
            real = self.real + other.real
            dual = self.dual + other.dual
            return DualNumber(real, dual)
        else:
            return DualNumber(self.real + other, self.dual)

    def __sub__(self, other):
        if isinstance(other, DualNumber):
            real = self.real - other.real
            dual = self.dual - other.dual
            return DualNumber(real, dual)
        else:
            return DualNumber(self.real - other, self.dual)

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            real = self.real * other.real
            dual = (self.real * other.dual) + (other.real * self.dual)
            return DualNumber(real, dual)
        else:
            return DualNumber(self.real * other, self.dual)

    def __truediv__(self, other):
        if isinstance(other, DualNumber):
            real = self.real / other.real
            dual = ((-1) * self.real * (self.dual + other.dual)) / (other.real ** 2)
            return DualNumber(real, dual)
        else:
            return DualNumber(self.real / other, self.dual)


def _print_dual(a):
    print('Real: {}. Dual: {}'.format(a.real, a.dual))


def _dual_constant(c):
    return DualNumber(c, 0)


def _dual_variable(c, seed):
    return DualNumber(c, seed)

def _dual_identity(input):
    return DualNumber(_real(input), _dual(input))

def _real(a):
    return a.real


def _dual(a):
    return a.dual


def _dual_add(a, b):
    return a + b


def _dual_sub(a, b):
    return a - b


def _dual_mul(a, b):
    return a * b


def _dual_sqrt(a):
    pass


def _dual_pow(a, power):
    return a ** power


def _dual_div(a, b):
    return a / b


def _dual_abs(a):
    pass


def _dual_sin(a):
    pass


def _dual_cos(a):
    pass


def _dual_tan(a):
    pass


def _dual_tanh(a):
    pass


def _dual_arctan(a):
    pass


def _dual_logistic(a):
    pass


def _dual_exp(a):
    pass


def _dual_log(a, base=10):
    pass


def _dual_ln(a):
    pass


def _dual_erf(a):
    pass


def _dual_gudermann(a):
    pass


def _dual_softplus(a):
    pass


def _dual_rectifier(a):
    pass
