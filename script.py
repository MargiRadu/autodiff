from autodiff.graph_nodes import variable, constant, feeder, loss
from autodiff.active_section import ActiveSection
from optimization.local_iterative_optimizer import MiniBatchSGD
from autodiff.backend.forward_accumulation_backend import ForwardAccumulationBackend

with ActiveSection() as section:
    x = feeder()
    y = feeder()
    theta = variable(5)

    # Define model.
    model = x * theta
    loss((model - y) * (model - y))

    section.register_backend(ForwardAccumulationBackend())
    section.register_optimizer(MiniBatchSGD(learning_rate=0.1, epsilon=0.001, batch_size=3, max_iterations=400))

    # Run optimizer
    section.optimize_model({x: [1, 2, 3], y: [2, 4, 6]})

    # Evaluate optimized model
    results = section.eval(model, {x: [10, 20, 30, 40], y: [0, 0, 0, 0]})

    print(results)
