"""
Double integrator example

"""

import torch
import torch.nn.functional as F
import slim
import psl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from neuromancer.activations import activations
from neuromancer import blocks, estimators, dynamics
from neuromancer.trainer import Trainer
from neuromancer.simulators import ClosedLoopSimulator
from neuromancer.visuals import VisualizerClosedLoop
from neuromancer.problem import Problem, Objective
from neuromancer import policies
import neuromancer.arg as arg
from neuromancer.datasets import Dataset
from neuromancer.loggers import BasicLogger, MLFlowLogger



def arg_dpc_problem(prefix=''):
    """
    Command line parser for DPC problem definition arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers
                         are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = arg.ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("DPC")
    gp.add("-nsteps", type=int, default=1,
           help="prediction horizon.")
    gp.add("-Qx", type=float, default=1.0,
           help="state weight.")
    gp.add("-Qu", type=float, default=10.0,
           help="control action weight.")
    gp.add("-Q_sub", type=float, default=0.0,
           help="regularization weight.")
    gp.add("-Q_con_x", type=float, default=10.0,
           help="state constraints penalty weight.")
    gp.add("-Q_con_u", type=float, default=100.0,
           help="Input constraints penalty weight.")
    gp.add("-nx_hidden", type=int, default=10,
           help="Number of hidden states")
    gp.add("-n_layers", type=int, default=4,
           help="Number of hidden layers")
    gp.add("-bias", action="store_true",
           help="Whether to use bias in the neural network block component models.")
    gp.add("-norm", nargs="+", default=[], choices=["U", "D", "Y", "X"],
               help="List of sequences to max-min normalize")
    gp.add("-data_seed", type=int, default=408,
           help="Random seed used for simulated data")
    gp.add("-epochs", type=int, default=1000,
           help='Number of training epochs')
    gp.add("-lr", type=float, default=0.001,
           help="Step size for gradient descent.")
    gp.add("-patience", type=int, default=100,
           help="How many epochs to allow for no improvement in eval metric before early stopping.")
    gp.add("-warmup", type=int, default=10,
           help="Number of epochs to wait before enacting early stopping policy.")
    # gp.add("-verbosity", type=int, default=1,
    #        help="How many epochs in between status updates")
    return parser


def plot_policy(net, xmin=-5, xmax=5):
    x = torch.arange(xmin, xmax, 0.1)
    y = torch.arange(xmin, xmax, 0.1)
    xx, yy = torch.meshgrid(x, y)
    features = torch.stack([xx, yy]).transpose(0, 2)
    uu = net(features)
    plot_u = uu.detach().numpy()[:,:,0]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx.detach().numpy(), yy.detach().numpy(), plot_u,
                           cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    return


def lpv_batched(fx, x):
    """ LPV refactor of DNN
    :param fx:
    :param x:
    :return:
    """
    x_layer = x

    Aprime_mats = []
    activation_mats = []
    bprimes = []

    for nlin, lin in zip(fx.nonlin, fx.linear):
        A = lin.effective_W()  # layer weight

        b = lin.bias if lin.bias is not None else torch.zeros(A.shape[-1])
        Ax = torch.matmul(x_layer, A) + b  # affine transform

        zeros = Ax == 0
        lambda_h = nlin(Ax) / Ax  # activation scaling
        lambda_h[zeros] = 0.

        lambda_h_mats = [torch.diag(v) for v in lambda_h]
        activation_mats += lambda_h_mats
        lambda_h_mats = torch.stack(lambda_h_mats)

        x_layer = Ax * lambda_h

        Aprime = torch.matmul(A, lambda_h_mats)
        # Aprime = A * lambda_h
        Aprime_mats += [Aprime]

        bprime = lambda_h * b
        bprimes += [bprime]

    # network-wise parameter varying linear map:  A* = A'_L ... A'_1
    Astar = Aprime_mats[0]
    bstar = bprimes[0] # b x nx
    for Aprime, bprime in zip(Aprime_mats[1:], bprimes[1:]):
        Astar = torch.bmm(Astar, Aprime)
        bstar = torch.bmm(bstar.unsqueeze(-2), Aprime).squeeze(-2) + bprime

    return Astar, bstar, Aprime_mats, bprimes, activation_mats




def check_stability(A, B, net):
    return




if __name__ == "__main__":

    """
    # # #  arguments
    """
    parser = arg.ArgParser(parents=[arg.log(),
                                    arg_dpc_problem()])
    args, grps = parser.parse_arg_groups()

    # problem dimensions
    nx = 2
    ny = 2
    nu = 1
    # number of datapoints
    nsim = 10000
    # problem constraints
    umin = -1
    umax = 1
    xmin = -10
    xmax = 10

    """
    # # #  dataset 
    """
    #  randomly sampled constraints and initial conditions
    #  we treat states as observables, i.e. Y = X
    sequences = {
        "Y_max": xmax*np.ones([nsim, nx]),
        "Y_min": xmin*np.ones([nsim, nx]),
        "U_max": umax*np.ones([nsim, nu]),
        "U_min": umin*np.ones([nsim, nu]),
        "Y": 10*np.random.randn(nsim, nx),
        "U": np.random.randn(nsim, nu),
    }
    dataset = Dataset(nsim=nsim, ninit=0, norm=args.norm, nsteps=args.nsteps,
                      device='cpu', sequences=sequences, name='closedloop')
    # dataset.dims['U'] = (nsim, nu)

    """
    # # #  System model
    """
    fu = slim.maps['linear'](nu, nx)
    fx = slim.maps['linear'](nx, nx)
    fy = slim.maps['linear'](nx, ny)
    # Identity
    estimator = estimators.FullyObservable({**dataset.dims, "x0": (nx,)},
                                           nsteps=1,  # future window Nf
                                           window_size=1,  # past window Np <= Nf
                                           input_keys=["Yp"],
                                           name='estimator')
    # LTI SSM
    dynamics_model = dynamics.BlockSSM(fx, fy, fu=fu, name='dynamics',
                                       input_keys={'x0': f'x0_{estimator.name}'})
    dynamics_model.input_keys[dynamics_model.input_keys.index('Uf')] = 'U_pred_policy'

    # model matrices
    A = torch.tensor([[1.2, 1.0],
                      [0.0, 1.0]])
    B = torch.tensor([[1.0],
                      [0.5]])
    C = torch.tensor([[1.0, 1.0]])
    dynamics_model.fx.linear.weight = torch.nn.Parameter(A)
    dynamics_model.fu.linear.weight = torch.nn.Parameter(B)
    dynamics_model.fy.linear.weight = torch.nn.Parameter(C)
    # fix model parameters
    dynamics_model.requires_grad_(False)

    """
    # # #  control policy
    """
    activation = activations['relu']
    linmap = slim.maps['linear']
    block = blocks.MLP

    policy = policies.MLPPolicy(
        {"x0_estim": (dynamics_model.nx,), **dataset.dims},
        nsteps=args.nsteps,
        bias=args.bias,
        linear_map=linmap,
        nonlin=activation,
        hsizes=[args.nx_hidden] * args.n_layers,
        input_keys=['Yp'],
        name='policy',
    )

    """
    # # #  DPC objectives and constraints
    """
    regulation_loss = Objective(
        [f'Y_pred_{dynamics_model.name}'],
        lambda x: F.mse_loss(x, x),
        weight=args.Qx,
        name="x^T*Qx*x loss",
    )
    action_loss = Objective(
        [f"U_pred_{policy.name}"],
        lambda x: F.mse_loss(x, x),
        weight=args.Qu,
        name="u^T*Qu*u loss",
    )
    regularization = Objective(
        [f"reg_error_{policy.name}"], lambda reg: reg, weight=args.Q_sub, name="reg_loss",
    )
    state_lower_bound_penalty = Objective(
        [f'Y_pred_{dynamics_model.name}', "Y_minf"],
        lambda x, xmin: torch.mean(F.relu(-x + xmin)),
        weight=args.Q_con_x,
        name="state_lower_bound",
    )
    state_upper_bound_penalty = Objective(
        [f'Y_pred_{dynamics_model.name}', "Y_maxf"],
        lambda x, xmax: torch.mean(F.relu(x - xmax)),
        weight=args.Q_con_x,
        name="state_upper_bound",
    )
    inputs_lower_bound_penalty = Objective(
        [f"U_pred_{policy.name}", "U_minf"],
        lambda u, umin: torch.mean(F.relu(-u + umin)),
        weight=args.Q_con_u,
        name="input_lower_bound",
    )
    inputs_upper_bound_penalty = Objective(
        [f"U_pred_{policy.name}", "U_maxf"],
        lambda u, umax: torch.mean(F.relu(u - umax)),
        weight=args.Q_con_u,
        name="input_upper_bound",
    )

    objectives = [regularization, regulation_loss, action_loss]
    constraints = [
        state_lower_bound_penalty,
        state_upper_bound_penalty,
        inputs_lower_bound_penalty,
        inputs_upper_bound_penalty,
    ]

    """
    # # #  DPC problem 
    """
    model = Problem(
        objectives,
        constraints,
        [estimator, policy, dynamics_model],
    )

    """
    # # #  DPC trainer 
    """
    # loger and metrics
    args.savedir = 'test_control'
    args.verbosity = 1
    log_constructor = MLFlowLogger if args.logger == 'mlflow' else BasicLogger

    metrics = ["nstep_dev_loss", "loop_dev_loss", "best_loop_dev_loss",
               "nstep_dev_ref_loss", "loop_dev_ref_loss"]
    logger = log_constructor(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)
    logger.args.system = 'integrator'
    # device and optimizer
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # plot visualizer
    plot_keys = ["Y_pred", "U_pred"]  # variables to be plotted
    visualizer = VisualizerClosedLoop(
        dataset, policy, plot_keys, args.verbosity, savedir=args.savedir
    )
    # simulator
    simulator = ClosedLoopSimulator(
        model=model, dataset=dataset, emulator=dynamics_model, policy=policy
    )
    # trainer
    trainer = Trainer(
        model,
        dataset,
        optimizer,
        logger=logger,
        visualizer=visualizer,
        simulator=simulator,
        epochs=args.epochs,
        patience=args.patience,
        warmup=args.warmup,
    )
    # Train control policy
    best_model = trainer.train()

    # TODO: generalize to plot N-step ahead policy
    plot_policy(policy.net)

    # TODO: eigenvalue plots + closed loop trajectories
    # simple scripts or check simulator

    # best_outputs = trainer.evaluate(best_model)
    # plots = visualizer.eval(best_outputs)
    # # Logger
    # logger.log_artifacts(plots)
    # logger.clean_up()