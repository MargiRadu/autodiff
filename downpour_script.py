from autodiff.graph_nodes import variable, feeder, loss
from autodiff.active_section import ActiveSection
from autodiff.backend.forward_accumulation_backend import ForwardAccumulationBackend

from optimization.downpour.downpour_master import DownpourSGD

with ActiveSection() as section:
    x = feeder()
    y = feeder()
    theta = variable(5)

    # Define model.
    model = x * theta
    loss((model - y) * (model - y))

    feed_dict = {x: [1, 2, 3, 4, 5, 6], y: [3, 6, 9, 12, 15, 18]}

    # Initialize backend
    backend = ForwardAccumulationBackend()
    needed_ops = set(section.graph_ops_map.values()).difference({'loss'})
    backend.init_capabilities(needed_ops)
    sgd = DownpourSGD([('127.0.0.1', 16001), ('127.0.0.1', 16002)], 0.001, 0.0001, backend=backend, max_iterations=400, batch_size=1)

    flattened_graph = section.flatten_graph()

    # Call optimizer using selected backend
    sgd.optimize(graph=flattened_graph,
                 variable_ids=[v.identifier for v in section.variables],
                 loss_id=section.loss_id,
                 feed_dict={k.identifier: v for k, v in feed_dict.items()},
                 variable_init_feed_dict={v.identifier: 0.0 for v in section.variables},
                 constant_feed_dict=section.constants)

    variables = sgd.get_variable_values()
    print(variables)
