"""
Neural State Space Models - N-step ahead System ID 
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
import mlflow
import argparse
from scipy.io import loadmat
from copy import deepcopy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None,
                        help="Gpu to use")
    # OPTIMIZATION PARAMETERS
    opt_group = parser.add_argument_group('OPTIMIZATION PARAMETERS')
    opt_group.add_argument('-yonly', action='store_true',
                           help='Use only y prediction for loss update.')
    opt_group.add_argument('-batchsize', type=int, default=-1)
    opt_group.add_argument('-dropout', type=float, default=0.0)
    opt_group.add_argument('-epochs', type=int, default=15000)
    opt_group.add_argument('-lr', type=float, default=0.001,
                        help='Step size for gradient descent.')

    #################
    # DATA PARAMETERS
    data_group = parser.add_argument_group('DATA PARAMETERS')
    data_group.add_argument('-nsteps', type=int, default=16,
                            help='Number of steps for open loop during training.')
    data_group.add_argument('-norm', action='store_true',
                            help='Whether to normalize U and D input')

    ##################
    # MODEL PARAMETERS
    model_group = parser.add_argument_group('MODEL PARAMETERS')
    model_group.add_argument('-cell_type', type=str, choices=['ssm', 'rnn', 'gru', 'lin'], default='ssm')
    model_group.add_argument('-num_layers', type=int, default=1)
    model_group.add_argument('-cknown', action='store_true', help='Whether to learn the C matrix which derives y from x')
    model_group.add_argument('-bias', action='store_true', help='Whether to use bias in the neural network models.')

    ####################
    # LOGGING PARAMETERS
    log_group = parser.add_argument_group('LOGGING PARAMETERS')
    log_group.add_argument('-savedir', type=str, default='weights/garbage/',
                           help="Where should your trained model be saved")
    log_group.add_argument('-verbosity', type=int, default=100,
                           help="How many epochs in between status updates")
    log_group.add_argument('-exp', default='test',
                           help='Will group all run under this experiment name.')
    log_group.add_argument('-location', default='mlruns',
                           help='Where to write mlflow experiment tracking stuff')
    log_group.add_argument('-run', default='test',
                           help='Some name to tell what the experiment run was about.')
    return parser.parse_args()


def plot_matrices(matrices, labels, figname):
    rows = len(matrices)
    cols = len(matrices[0])
    fig, axes = plt.subplots(nrows=rows, ncols=cols, squeeze=False)
    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(matrices[i][j])
            axes[i, j].title.set_text(labels[i][j])
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(figname)
    mlflow.log_artifact(figname)


def plot_trajectories(traj1, traj2, labels, figname):
    fig, ax = plt.subplots(len(traj1), 1)
    for row, (t1, t2, label) in enumerate(zip(traj1, traj2, labels)):
        if t2 is not None:
            ax[row].plot(t1, label=f'True')
            ax[row].plot(t2, '--', label=f'Pred')
        else:
            ax[row].plot(t1)
        steps = range(0, t1.shape[0] + 1, 288)
        days = np.array(list(range(len(steps))))+7
        ax[row].set(xticks=steps,
                    xticklabels=days,
                    ylabel=label,
                    xlim=(0, len(t1)))
        ax[row].tick_params(labelbottom=False)
        ax[row].axvspan(2016, 4032, facecolor='grey', alpha=0.25, zorder=-100)
        ax[row].axvspan(4032, 6048, facecolor='grey', alpha=0.5, zorder=-100)
    ax[row].tick_params(labelbottom=True)
    ax[row].set_xlabel('Day')
    ax[0].text(64, 30, '             Train                ',
            bbox={'facecolor': 'white', 'alpha': 0.5})
    ax[0].text(2064, 30, '           Validation           ',
            bbox={'facecolor': 'grey', 'alpha': 0.25})
    ax[0].text(4116, 30, '              Test                ',
               bbox={'facecolor': 'grey', 'alpha': 0.5})
    plt.tight_layout()
    plt.savefig(figname)
    mlflow.log_artifact(figname)


class ConstrainedLinear(nn.Module):
    def __init__(self, nx, nu):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(nx, nu))
        self.scalar = nn.Parameter(torch.rand(nx, nx))  # matrix scaling to allow for different row sums

    def effective_W(self):
        s_clapmed = 1 - 0.1 * torch.sigmoid(self.scalar)  # constrain sum of rows to be in between 0.9 and 1
        w_sofmax = s_clapmed * F.softmax(self.weight, dim=1)
        return w_sofmax.T

    def forward(self, x):
        return torch.mm(x, self.effective_W())


class GRU(nn.Module):
    def __init__(self, nx, nu, nd, bias=False):
        super().__init__()
        self.nx, self.nu, self.dn = nx, nu, nd
        self.cell = nn.GRUCell(nu+nd, nx, bias=bias)

    def forward(self, x, U, D):
        X = []
        Y = []
        for u, d in zip(U, D):
            ins = torch.cat([u, d], dim=1)
            x = self.cell(ins, x)
            y = x[:, -1]
            X.append(x)
            Y.append(y)
        return torch.stack(X), torch.stack(Y)


class RNN(nn.Module):
    def __init__(self, nx, nu, nd, bias=False):
        super().__init__()
        self.nx, self.nu, self.dn = nx, nu, nd
        self.cell = nn.RNNCell(nu+nd, nx, bias=bias, nonlinearity='relu')

    def forward(self, x, U, D):
        X = []
        Y = []
        for u, d in zip(U, D):
            ins = torch.cat([u, d], dim=1)
            x = self.cell(ins, x)
            y = x[:, -1]
            X.append(x)
            Y.append(y)
        return torch.stack(X), torch.stack(Y)


class SSM(nn.Module):
    def __init__(self, nx, nu, nd, bias=False):
        super().__init__()
        self.A = ConstrainedLinear(nx, nx)
        self.B = nn.Linear(nu, nx, bias=bias)
        self.E = nn.Linear(nd, nx, bias=bias)

    def forward(self, x, U, D):
        """
        """
        X = []
        Y = []
        for u, d in zip(U, D):
            x = self.A(x) + self.B(u) + self.E(d)
            X.append(x)
            Y.append(x[:, -1])
        return torch.stack(X), torch.stack(Y)


class LIN(nn.Module):
    def __init__(self, nx, nu, nd, bias=False):
        super().__init__()
        self.A = nn.Linear(nx, nx, bias=bias)
        self.B = nn.Linear(nu, nx, bias=bias)
        self.E = nn.Linear(nd, nx, bias=bias)

    def forward(self, x, U, D):
        """
        """
        X = []
        Y = []
        for u, d in zip(U, D):
            x = self.A(x) + self.B(u) + self.E(d)
            X.append(x)
            Y.append(x[:, -1])
        return torch.stack(X), torch.stack(Y)


def min_max_norm(M):
    return (M - M.min(axis=0).reshape(1, -1))/(M.max(axis=0) - M.min(axis=0)).reshape(1, -1)


def control_profile(max_input=4e3, samples_day=288, sim_days=7):
    """
    """
    U_day = max_input*np.sin(np.arange(0, 2*np.pi, 2*np.pi/samples_day)) #  samples_day
    U = np.tile(U_day, sim_days).reshape(-1, 1) # samples_day*sim_days
    return U


def disturbance(file='../TimeSeries/disturb.mat', n_sim=2016):
    return loadmat(file)['D'][:, :n_sim].T # n_sim X 3


class Building:
    def __init__(self):
        self.A = np.matrix([[0.9950, 0.0017, 0.0000, 0.0031], [0.0007, 0.9957, 0.0003, 0.0031],
                       [0.0000, 0.0003, 0.9834, 0.0000], [0.2015, 0.4877, 0.0100, 0.2571]])
        self.B = np.matrix([[1.7586e-06], [1.7584e-06],
                       [1.8390e-10], [5.0563e-04]])
        self.E = np.matrix([[0.0002, 0.0000, 0.0000], [0.0002, 0.0000, 0.0000],
                       [0.0163, 0.0000, 0.0000], [0.0536, 0.0005, 0.0001]])
        self.C = np.matrix([[0.0, 0.0, 0.0, 1.0]])
        self.x = 20*np.ones(4, dtype=np.float32)

    def loop(self, nsim, U, D):
        """

        :param nsim: (int) Number of steps for open loop response
        :param U: (ndarray, shape=(nsim, self.nu)) Control profile matrix
        :param D: (ndarray, shape=(nsim, self.nd)) Disturbance matrix
        :param x: (ndarray, shape=(self.nx)) Initial state. If not give will use internal state.
        :return: The response matrices are aligned, i.e. X[k] is the state of the system that Y[k] is indicating
        """
        Y = np.zeros((nsim, 1))  # output trajectory placeholders
        X = np.zeros((nsim+1, 4))
        X[0] = 20*np.ones(4, dtype=np.float32)
        for k in range(nsim):
            Y[k] = self.C*np.asmatrix(X[k]).T
            d = np.asmatrix(D[k]).T
            u = np.asmatrix(U[k]).T
            x = self.A*np.asmatrix(X[k]).T + self.B*u + self.E*d
            X[k+1] = x.flatten()
        return X, Y


def make_dataset(nsteps, device):
    U = control_profile(samples_day=288, sim_days=28)

    nsim = U.shape[0]
    D = disturbance(n_sim=nsim)
    building = Building()
    D_scale, U_scale = min_max_norm(D), min_max_norm(U)
    X, Y = building.loop(nsim, U, D)

    target_response = X[1:]
    initial_states = X[:-1]
    data = np.concatenate([initial_states, U_scale, D_scale, target_response, Y], axis=1)[2016:]
    nsplits = (data.shape[0]) // nsteps
    leftover = (data.shape[0]) % nsteps
    data = np.stack(np.split(data[:data.shape[0] - leftover], nsplits))  # nchunks X nsteps X 14
    data = torch.tensor(data, dtype=torch.float32).transpose(0, 1).to(device)  # nsteps X nsamples X nfeatures
    train_idx = (data.shape[1] // 3)
    dev_idx = train_idx * 2
    train_data = data[:, :train_idx, :]
    dev_data = data[:, train_idx:dev_idx, :]
    test_data = data[:, dev_idx:, :]

    return train_data, dev_data, test_data


if __name__ == '__main__':
    args = parse_args()

    ####################################
    ###### LOGGING SETUP
    ####################################
    mlflow.set_tracking_uri(args.location)
    mlflow.set_experiment(args.exp)
    mlflow.start_run(run_name=args.run)
    params = {k: str(getattr(args, k)) for k in vars(args) if getattr(args, k)}
    mlflow.log_params(params)
    device = 'cpu'
    if args.gpu is not None:
        device = f'cuda:{args.gpu}'
    ####################################
    ###### DATA SETUP
    ####################################
    train_data, dev_data, test_data = make_dataset(args.nsteps, device)
    x0_in, U_in, D_in, x_response, Y_target = (train_data[0, :, :4],
                                               train_data[:, :, 4:5],
                                               train_data[:, :, 5:8],
                                               train_data[:, :, 8:-1],
                                               train_data[:, :, -1])

    x0_dev, U_dev, D_dev, x_response_dev, Y_target_dev = (dev_data[0, :, :4],
                                                          dev_data[:, :, 4:5],
                                                          dev_data[:, :, 5:8],
                                                          dev_data[:, :, 8:-1],
                                                          dev_data[:, :, -1])

    x0_tst, U_tst, D_tst, x_response_tst, Y_target_tst = (test_data[0, :, :4],
                                                          test_data[:, :, 4:5],
                                                          test_data[:, :, 5:8],
                                                          test_data[:, :, 8:-1],
                                                          test_data[:, :, -1])
    ####################################
    ######MODEL SETUP
    ####################################
    nx, nu, nd, ny = 4, 1, 3, 1
    models = {'rnn': RNN, 'ssm': SSM, 'gru': GRU, 'lin': LIN}
    model = models[args.cell_type](nx, nu, nd, bias=args.bias).float().to(device)
    nweights = sum([i.numel() for i in list(model.parameters())])
    print(nweights, "parameters in the neural net.")
    mlflow.log_param('Parameters', nweights)

    ####################################
    ######OPTIMIZATION SETUP
    ####################################
    criterion = nn.MSELoss()  # we'll convert this to RMSE later
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    #######################################
    ### N-STEP AHEAD TRAINING
    #######################################
    best_dev = np.finfo(np.float32).max
    for i in range(args.epochs):
        model.train()
        X_pred, Y_pred = model(x0_in, U_in, D_in)
        loss = criterion(Y_pred, Y_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ##################################
        # DEVELOPMENT SET EVALUATION
        ###################################
        with torch.no_grad():
            model.eval()
            X_pred, Y_pred = model(x0_dev, U_dev, D_dev)
            dev_loss = criterion(Y_pred, Y_target_dev)
            if dev_loss < best_dev:
                best_model = deepcopy(model.state_dict())
                best_dev = dev_loss
        mlflow.log_metrics({'trainloss': loss.item(), 'devloss': dev_loss.item(), 'bestdev': best_dev.item()}, step=i)
        if i % args.verbosity == 0:
            print(
                f'epoch: {i:2}  loss: {loss.item():10.8f}\tdevloss: {dev_loss.item():10.8f}\tbestdev: {best_dev.item()}')

    model.load_state_dict(best_model)
    ########################################
    ########## NSTEP TRAIN RESPONSE ########
    ########################################
    X_out, Y_out = model(x0_in, U_in, D_in)
    mlflow.log_metric('nstep_train_loss', criterion(Y_out, Y_target).item())
    xpred = X_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, 4)
    xtrue = x_response.transpose(0, 1).detach().cpu().numpy().reshape(-1, 4)

    X_out, Y_out = model(x0_dev, U_dev, D_dev)
    mlflow.log_metric('nstep_dev_loss', criterion(Y_out, Y_target_dev).item())
    devxpred = X_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, 4)
    devxtrue = x_response_dev.transpose(0, 1).detach().cpu().numpy().reshape(-1, 4)

    X_out, Y_out = model(x0_tst, U_tst, D_tst)
    mlflow.log_metric('nstep_test_loss', criterion(Y_out, Y_target_tst).item())
    testxpred = X_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, 4)
    testxtrue = x_response_tst.transpose(0, 1).detach().cpu().numpy().reshape(-1, 4)
    plot_trajectories([np.concatenate([xtrue[:, k], devxtrue[:, k], testxtrue[:, k]])
                       for k in range(xtrue.shape[1])],
                      [np.concatenate([xpred[:, k], devxpred[:, k], testxpred[:, k]])
                       for k in range(xpred.shape[1])],
                      ['$X_1$', '$X_2$', '$X_3$', '$X_4$'], 'nstep.png')

    ########################################
    ########## OPEN LOOP RESPONSE ####
    ########################################
    def open_loop(model, data):
        data = torch.cat([data[:, k, :] for k in range(data.shape[1])]).unsqueeze(1)
        x0_in, U_in, D_in, x_response, Y_target = (data[0, :, :4],
                                                   data[:, :, 4:5],
                                                   data[:, :, 5:8],
                                                   data[:, :, 8:-1],
                                                   data[:, :, -1:])
        X_pred, Y_pred = model(x0_in, U_in, D_in)
        open_loss = criterion(Y_pred.squeeze(), Y_target.squeeze())
        return (open_loss.item(),
                X_pred.squeeze().detach().cpu().numpy(),
                x_response.squeeze().detach().cpu().numpy())

    openloss, xpred, xtrue = open_loop(model, train_data)
    print(f'Train_open_loss: {openloss}')
    mlflow.log_metric('train_openloss', openloss)

    devopenloss, devxpred, devxtrue = open_loop(model, dev_data)
    print(f'Dev_open_loss: {devopenloss}')
    mlflow.log_metric('dev_openloss', devopenloss)

    testopenloss, testxpred, testxtrue = open_loop(model, test_data)
    print(f'Test_open_loss: {testopenloss}')
    mlflow.log_metric('Test_openloss', testopenloss)
    plot_trajectories([np.concatenate([xtrue[:, k], devxtrue[:, k], testxtrue[:, k]])
                       for k in range(xtrue.shape[1])],
                   [np.concatenate([xpred[:, k], devxpred[:, k], testxpred[:, k]])
                    for k in range(xpred.shape[1])],
                   ['$X_1$', '$X_2$', '$X_3$', '$X_4$'], 'open_test.png')

    torch.save(best_model, 'bestmodel.pth')
    mlflow.log_artifact('bestmodel.pth')
