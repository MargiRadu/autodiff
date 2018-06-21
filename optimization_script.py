from autodiff_old.computation_graph.active_section import ActiveSection

from autodiff_old.computation_graph.ops import Feeder, Variable, Mul

from autodiff_old.computation_graph.losses import QuadraticLoss

from autodiff_old.computation_graph.optimizers import StochasticGradientDescent


train_xs = [4, 7, 2]
train_ys = [20, 35, 10]

test_xs = [1, 2, 3]
test_ys = [5, 10, 15]

# An active section should receive a backend parameter that specifies the computational backend used to
# carry out all the computation. A backend can be Local, Remote, Tensorflow, Theano, etc.
with ActiveSection(backend=RemoteBackend(ip='192.168.0.2')) as model:
    # How can I define a model somewhere else?
    # Ideally, if the model is a function, it should be able to register itself at the time of the function call.
    x = Feeder()
    y = Feeder()
    theta = Variable()  # We can set the init value through a separate object, through the as

    f = Mul(theta, x)  # Implement a way to overload operators and create the nodes implicitly

    # A loss is a operation that also register itself as 'the loss'
    loss = QuadraticLoss(f, y)
    # loss = Square(
    #         Sub(y,
    #             Mul(theta,
    #                 x)))

    # Will hold data relevant to optimizer state and history, e.g. intermediate points, momentums used, etc.
    # TODO: All optimizers/losses/etc that need to be registered with the AS must implement a protocol that enables
    #       them to run remotely (e.g. use a predefined string identifier or send the code over to the server).
    #       ideally, each server will have a directory of strings -> optimizers/losses.
    optimizer = StochasticGradientDescent()

    """
    Checks performed: Optimizer is defined, loss is defined. 
    Issue overriding warning when trying to set a second optimizer/loss.
    Raise warning if no variable initializer is set and default initializer is used.
    ERM stands for empirical risk minimization.
    """
    model.erm({x: train_xs,
               y: train_ys})

    print('Optimized value is {}'.format(theta.value()))

    #
    model.evaluate(f, {x: test_xs})

    model.evaluate(loss, {x: test_xs,
                          y: test_ys})
