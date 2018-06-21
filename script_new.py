from autodiff.graph_nodes import variable, constant, feeder, loss
from autodiff.active_section import ActiveSection
from optimization.optimizer import MiniBatchSGD
from autodiff.backend.forward_accumulation_backend import LocalForwardAccumulationBackend


with ActiveSection() as section:
    x = feeder()
    y = feeder()
    theta = variable(3)

    # Define model.
    model = (x * theta) + (x * 2) + 1
    loss((model - y) * (model - y))

    section.register_backend(LocalForwardAccumulationBackend())
    section.register_optimizer(MiniBatchSGD(0.01, 0.001, 32, 5))

    # Run optimizer
    section.optimize_model({x: [1, 2, 3], y: [4, 5, 6, 7]})

    # Evaluate optimized model
    results = section.eval(model, {x: [1, 2, 3, 4], y: [0, 0, 0, 0]})

    print(results)