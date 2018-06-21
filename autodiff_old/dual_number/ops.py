from autodiff_old.dual_number.structure import DualNumber


def print_dual(a):
    print('Real: {}. Dual: {}'.format(a.real, a.dual))


def dual_constant(c):
    return DualNumber(c, 0)


def dual_variable(c):
    return DualNumber(c, 1)

def real(a):
    return a.real


def dual(a):
    return a.dual


def dual_add(a, b):
    return a + b


def dual_sub(a, b):
    return a - b


def dual_mul(a, b):
    return a * b


def dual_sqrt(a):
    pass


def dual_pow(a, power):
    return a ** power


def dual_div(a, b):
    return a / b


def dual_abs(a):
    pass


def dual_sin(a):
    pass


def dual_cos(a):
    pass


def dual_tan(a):
    pass


def dual_tanh(a):
    pass


def dual_arctan(a):
    pass


def dual_logistic(a):
    pass


def dual_exp(a):
    pass


def dual_log(a, base=10):
    pass


def dual_ln(a):
    pass


def dual_erf(a):
    pass


def dual_gudermann(a):
    pass


def dual_softplus(a):
    pass


def dual_rectifier(a):
    pass
