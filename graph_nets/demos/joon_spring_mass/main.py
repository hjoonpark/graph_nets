from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from graph_nets import blocks
from graph_nets import utils_tf
from graph_nets import utils_np
from graph_nets.demos import models
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import sonnet as snt
import tensorflow as tf

try:
    import seaborn as sns
except ImportError:
    pass
else:
    sns.reset_orig()


def base_graph(n, d):
    """
    n: number of masses
    d: spring rest length
    """
    masses = 1 # kg
    spring_constant = 50.
    gravity = 10.
    
    # nodes
    nodes = np.zeros((n, 5), dtype=np.float32)
    half_width = d*n/2.0
    nodes[:, 0] = np.linspace(-half_width, half_width, num=n, endpoint=False, dtype=np.float32)
    
    # indicate first/last nodes are fixed
    nodes[(0, -1), -1] = 1.
    
    # edges
    edges, senders, receivers = [], [], []
    for i in range(n-1):
        left_node = i
        right_node = i+1
        if right_node < n-1:
            # left incoming edge
            edges.append([spring_constant, d])
            senders.append(left_node)
            receivers.append(right_node)
        if left_node > 0:
            # right incoming edge
            edges.append([spring_constant, d])
            senders.append(right_node)
            receivers.append(left_node)
        
    return {
        "globals": [0., -gravity],
        "nodes": nodes,
        "edges": edges,
        "receivers": receivers,
        "senders": senders
    }

def hookes_law(receiver_nodes, sender_nodes, k, x_rest):
    """
    receiver_nodes/sender_nodes: Ex5, [x, y, vx, vy, is_fixed]
    k: spring constant for each edge
    x_rest: rest length of each edge
    """
    diff = receiver_nodes[..., 0:2] - sender_nodes[..., 0:2] # delta_x (dx, dy) between receiver and sender nodes
    x = tf.norm(diff, axis=-1, keepdims=True)
    print(diff.shape, x.shape)
    force_magnitude = -1 * tf.multiply(k, (x-x_rest) / x) # -k*(|x1-x0|-L0) / |x1-x0|
    force = force_magnitude * diff # -k* (|x1-x0|-L0) / |x1-x0| * (x1-x0)
    return force

def euler_integration(nodes, forces_per_node, step_size):
    """
    nodes: Ex5, [x, y, vx, vy, is_fixed]
    forces_per_node: Ex2, [fx, fy] acting on each edge
    step_size: scalar

    returns:
    nodes with updated velocities
    """
    is_fixed = nodes[..., 4:5]

    # set forces to zero for fixed nodes
    forces_per_node *= 1 - is_fixed
    new_vel = nodes[..., 2:4] + forces_per_node * step_size # v1 = v0 + f*dt
    return new_vel

def prediction_to_next_state(input_graph, predicted_graph, step_size):
    # manually integrate velocities to compute new positions
    new_pos = input_graph.nodes[..., :2] + predicted_graph.nodes * step_size
    new_nodes = tf.concat([new_pos, predicted_graph.nodes, input_graph.nodes[..., 4:5]], axis=-1)
    return input_graph.replace(nodes=new_nodes)


class SpringMassSimulator(snt.AbstractModule):
    def __init__(self, step_size, name="SpringMassSimulator"):
        super(SpringMassSimulator, self).__init__(name=name)
        self._step_size = step_size

        with self._enter_variable_scope():
            self._aggregator = blocks.ReceivedEdgesToNodesAggregator(reducer=tf.unsorted_segment_sum)

    def _build(self, graph):
        """
        graph: graphs.GraphsTuple having, for some integers N, E, G:
            - edges: Nx2 [spring_constant, rest_length] 
            - nodes: Ex5 [x, y, vx, vy, is_fixed]
            - globals: Gx2 [gravitational constant]

        returns: graphs.GraphsTuple with
            - edges: [fx, fy]
            - nodes: positions and velocities after one step of Euler integration

        """
        receiver_nodes = blocks.broadcast_receiver_nodes_to_edges(graph)
        sender_nodes = blocks.broadcast_sender_nodes_to_edges(graph)

        spring_force_per_edge = hookes_law(receiver_nodes, sender_nodes, graph.edges[..., 0:1], graph.edges[..., 1:2])
        graph = graph.replace(edges=spring_force_per_edge)

        spring_force_per_node = self._aggregator(graph)
        gravity = blocks.broadcast_globals_to_nodes(graph)
        updated_velocities = euler_integration(graph.nodes, spring_force_per_node + gravity, self._step_size)
        graph = graph.replace(nodes=updated_velocities)
        return graph

        

def roll_out_physics(simulator, graph, steps, step_size):
    """
    simulator: SpringMassSimulator

    graph: graphs.GraphsTuple having edges, nodes, and globals.
    steps: integer
    step_size: scalar

    returns: a pair of:
        - the graph, updated after steps of simulations
        - (steps+1)xNx5 node features at each step
    """
    
    def body(t, graph, nodes_per_step):
        predicted_graph = simulator(graph)
        if isinstance(predicted_graph, list):
            predicted_graph = predicted_graph[-1]
        graph = prediction_to_next_state(graph, predicted_graph, step_size)
        return t+1, graph, nodes_per_step.write(t, graph.nodes)

    nodes_per_step = tf.TensorArray(dtype=graph.nodes.dtype, size=steps+1, element_shape=graph.nodes.shape)
    nodes_per_step = nodes_per_step.write(0, graph.nodes)

    _, g, nodes_per_step = tf.while_loop(lambda t, *unused_args: t <= steps, body, loop_vars=[1, graph, nodes_per_step])
    return g, nodes_per_step.stack()



def set_rest_lengths(graph):
    """Computes and sets rest lengths for the springs in a physical system.

    The rest length is taken to be the distance between each edge's nodes.

    Args:
    graph: a graphs.GraphsTuple having, for some integers N, E:
        - nodes: Nx5 Tensor of [x, y, _, _, _] for each node.
        - edges: Ex2 Tensor of [spring_constant, _] for each edge.

    Returns:
    The input graph, but with [spring_constant, rest_length] for each edge.
    """
    receiver_nodes = blocks.broadcast_receiver_nodes_to_edges(graph)
    sender_nodes = blocks.broadcast_sender_nodes_to_edges(graph)
    
    rest_length = tf.norm(
        receiver_nodes[..., :2] - sender_nodes[..., :2], axis=-1, keepdims=True)
    return graph.replace(
        edges=tf.concat([graph.edges[..., :1], rest_length], axis=-1))

def generate_trajectory(simulator, graph, steps, step_size, node_noise_level,
                        edge_noise_level, global_noise_level):
    """Applies noise and then simulates a physical system for a number of steps.

    Args:
    simulator: A SpringMassSimulator, or some module or callable with the same
        signature.
    graph: a graphs.GraphsTuple having, for some integers N, E, G:
        - nodes: Nx5 Tensor of [x, y, v_x, v_y, is_fixed] for each node.
        - edges: Ex2 Tensor of [spring_constant, _] for each edge.
        - globals: Gx2 tf.Tensor containing the gravitational constant.
    steps: Integer; the length of trajectory to generate.
    step_size: Scalar.
    node_noise_level: Maximum distance to perturb nodes' x and y coordinates.
    edge_noise_level: Maximum amount to perturb edge spring constants.
    global_noise_level: Maximum amount to perturb the Y component of gravity.

    Returns:
    A pair of:
    - The input graph, but with rest lengths computed and noise applied.
    - A `steps+1`xNx5 tf.Tensor of the node features at each step.
    """
    graph = set_rest_lengths(graph)
    _, n = roll_out_physics(simulator, graph, steps, step_size)
    return graph, n


def create_loss_ops(target_op, output_ops):
    """Create supervised loss operations from targets and outputs.

    Args:
    target_op: The target velocity tf.Tensor.
    output_ops: The list of output graphs from the model.

    Returns:
    A list of loss values (tf.Tensor), one per output op.
    """
    loss_ops = [
        tf.reduce_mean(
            tf.reduce_sum((output_op.nodes - target_op[..., 2:4])**2, axis=-1))
        for output_op in output_ops
    ]
    return loss_ops


def make_all_runnable_in_session(*args):
    """Apply make_runnable_in_session to an iterable of graphs."""
    return [utils_tf.make_runnable_in_session(a) for a in args]

if __name__ == "__main__":
    SEED = 1
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    tf.reset_default_graph()
    rand = np.random.RandomState(SEED)

    # model parameters
    num_processing_steps_tr = 1
    num_processing_steps_ge = 1

    # data / training parameters
    num_training_iterations = 1000
    batch_size_tr = 256
    batch_size_ge = 100
    num_time_steps = 50
    step_size = 0.1
    num_masses_min_max_tr = (3, 5)
    dist_between_masses_min_max_tr = (0.2, 1.0)

    # create the model
    model = models.EncodeProcessDecode(node_output_size=2)

    # base graphs for training
    num_masses_tr = rand.randint(*num_masses_min_max_tr, size=batch_size_tr)
    dist_between_masses_tr = rand.uniform(*dist_between_masses_min_max_tr, size=batch_size_tr)
    print("num_masses_tr:", num_masses_tr.shape, "\t dist_between_masses_tr:", dist_between_masses_tr.shape)
    static_graph_tr = []
    i = 0
    for n, d in zip(num_masses_tr, dist_between_masses_tr):
        # print("{}\tn={}, d={}".format(i, n, d))
        bg = base_graph(n, d)
        static_graph_tr.append(bg)
        # print("{} --------------------------------------".format(i))
        # for k, v in bg.items():
        #     print("\t{}: {}".format(k, v))
        # i += 1

    base_graph_tr = utils_tf.data_dicts_to_graphs_tuple(static_graph_tr)

    save_test_image = False
    if save_test_image:
        base_graph_tr_np = utils_np.data_dicts_to_graphs_tuple(static_graph_tr)
        graphs_nx = utils_np.graphs_tuple_to_networkxs(base_graph_tr_np)
        _, axs = plt.subplots(ncols=2, figsize=(6, 3))
        for iax, (graph_nx, ax) in enumerate(zip(graphs_nx, axs)):
            nx.draw(graph_nx, ax=ax)
        ax.set_title("Graph {}".format(iax))
        plt.savefig("./test.png", dpi=150)

    # base graphs for testing
    base_graph_4_ge = utils_tf.data_dicts_to_graphs_tuple([base_graph(4, 0.5)] * batch_size_ge)

    # physics simulator for data generation
    simulator = SpringMassSimulator(step_size=step_size)

    # training
    initial_conditions_tr, true_trajectory_tr = generate_trajectory(
        simulator,
        base_graph_tr,
        num_time_steps,
        step_size,
        node_noise_level=0.0,
        edge_noise_level=0.0,
        global_noise_level=0.0
    )
    print("true_trajectory_tr:", true_trajectory_tr.shape)

    # random start
    t = tf.random_uniform([], minval=0, maxval=num_time_steps - 1, dtype=tf.int32)
    input_graph_tr = initial_conditions_tr.replace(nodes=true_trajectory_tr[t])
    target_nodes_tr = true_trajectory_tr[t+1]
    
    output_ops_tr = model(input_graph_tr, num_processing_steps_tr)

    # test
    initial_conditions_4_ge, true_trajectory_4_ge = generate_trajectory(
        lambda x: model(x, num_processing_steps_ge),
        base_graph_4_ge,
        num_time_steps,
        step_size,
        node_noise_level=0.0,
        edge_noise_level=0.0,
        global_noise_level=0.0
    )

    _, true_nodes_rollout_4_ge = roll_out_physics(simulator, initial_conditions_4_ge, num_time_steps, step_size)
    _, predicted_nodes_rollout_4_ge = roll_out_physics(lambda x: model(x, num_processing_steps_ge), initial_conditions_4_ge, num_time_steps, step_size)

    # training loss
    loss_ops_tr = create_loss_ops(target_nodes_tr, output_ops_tr)

    # trianing loss across processing steps
    loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr

    # test
    loss_op_4_ge = tf.reduce_mean(tf.reduce_sum((predicted_nodes_rollout_4_ge[..., 2:4] - true_nodes_rollout_4_ge[..., 2:4])**2, axis=-1))

    # optimizer
    learning_rate = 1e-3
    optimizer = tf.train.AdamOptimizer(learning_rate)
    step_op = optimizer.minimize(loss_op_tr)

    input_graph_tr = make_all_runnable_in_session(input_graph_tr)
    initial_condition_4_ge = make_all_runnable_in_session(initial_conditions_4_ge)

    # train
    try:
        sess.close()
    except NameError:
        pass
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    last_iteration = 0
    logged_iterations = []
    losses_tr = []
    losses_4_ge = []

    log_every_seconds = 20
    print("# (iteration number), T (elapsed seconds), "
        "Ltr (training 1-step loss), "
        "Lge4 (test/generalization rollout loss for 4-mass strings), ")

    start_time = time.time()
    last_log_time = start_time
    for iteration in range(last_iteration, num_training_iterations):
        last_iteration = iteration
        train_values = sess.run({
            "step": step_op,
            "loss": loss_op_tr,
            "input_graph": input_graph_tr,
            "target_nodes": target_nodes_tr,
            "outputs": output_ops_tr
        })
        the_time = time.time()
        elapsed_since_last_log = the_time - last_log_time
        if elapsed_since_last_log > log_every_seconds:
            last_log_time = the_time
            test_values = sess.run({
                "loss_4": loss_op_4_ge,
                "true_rollout_4": true_nodes_rollout_4_ge,
                "predicted_rollout_4": predicted_nodes_rollout_4_ge
            })
            elapsed = time.time() - start_time
            losses_tr.append(train_values["loss"])
            losses_4_ge.append(test_values["loss_4"])
            logged_iterations.append(iteration)
            print("# {:05d}, T {:.1f}, Ltr {:.4f}, Lge4 {:.4f}".format(
                iteration, elapsed, train_values["loss"], test_values["loss_4"]))



    # # n = 5 
    # # d = 1
    # # g = base_graph(n, d)
    # # for k, v in g.items():
    # #     print(k)
    # #     print(v)
    # #     print()
