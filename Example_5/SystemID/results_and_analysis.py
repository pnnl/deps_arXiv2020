import pandas
import glob
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from scipy import stats

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from system_id import make_dataset, RNN, SSM, plot_matrices, plot_trajectories, Building, LIN
import pandas as pd
import scipy.linalg as LA

steps = [8, 16, 32, 64, 128]
N = 5

files = glob.glob('csvs/*')

gru = [f for f in files if 'gru' in f]
rnn = [f for f in files if 'rnn' in f]
lin = [f for f in files if 'lin' in f]
ssm = [f for f in files if 'ssm' in f]

stdopen, stdnstep, meanopen, meannstep, minopen, minnstep = [pandas.DataFrame(index=['LIN', 'SSM', 'RNN', 'GRU'],
                                                                              columns=[8, 16, 32, 64, 128, 256])
                                                             for i in range(6)]
print(stdopen)
for l, n in zip([lin, ssm, rnn, gru], ['LIN', 'SSM', 'RNN', 'GRU']):
    for step in ['8', '16', '32', '64', '128', '256']:
        res = pandas.read_csv([k for k in l if f'{n.lower()}_{step}.csv' in k][0])
        if 'bias' in res.columns:
            res = res.loc[res['bias'].isnull()]
        best = res.loc[res['dev_openloss'].idxmin()]
        minnstep.loc[n][int(step)] = best['nstep_test_loss']
        minopen.loc[n][int(step)] = best['Test_openloss']
        res = res.loc[res['lr'] == best['lr']]
        res = res.loc[res['Test_openloss'].notnull()]
        nsteploss = res['nstep_test_loss']
        openloss = res['Test_openloss']
        mean_openloss = openloss.mean()
        mean_nsteploss = nsteploss.mean()
        std_openloss = openloss.std()
        std_nsteploss = nsteploss.std()
        stdopen.loc[n][int(step)] = std_openloss
        stdnstep.loc[n][int(step)] = std_nsteploss
        meanopen.loc[n][int(step)] = mean_openloss
        meannstep.loc[n][int(step)] = mean_nsteploss


for k in [stdopen, stdnstep, meanopen, meannstep, minopen, minnstep]:
    print(k.to_latex(float_format=lambda x: '%.3f' % x))

fig = plt.figure()
width = 0.25
ind = np.arange(6)

for i, n in enumerate(['LIN', 'SSM', 'RNN']):
    plt.bar(ind+i*width, minopen.loc[n], width, label=n, edgecolor='white')

plt.xlabel('Training Prediction Horizon')
plt.ylabel('Open loop MSE')

plt.xticks(ind + width, ('8', '16', '32', '64', '128', '256'))
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('open_loop_min_mse.eps')
plt.savefig('open_loop_min_mse.png')

fig = plt.figure()
width = 0.35
ind = np.arange(6)

for i, n in enumerate(['LIN', 'SSM']):
    plt.bar(ind+i*width, meanopen.loc[n], width, label=n, edgecolor='white', yerr=[(0, 0, 0, 0, 0, 0), stdopen.loc[n]])

plt.xlabel('Training Prediction Horizon')
plt.ylabel('Open loop MSE')

plt.xticks(ind + .5*width, ('8', '16', '32', '64', '128', '256'))
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('lin_ssm_compare_mse.eps')
plt.savefig('lin_ssm_compare_mse.png')

fig = plt.figure()
width = 0.20
ind = np.arange(6)
for i, n in enumerate(['LIN', 'SSM', 'RNN', 'GRU']):
    plt.bar(ind+i*width, minnstep.loc[n], width, label=n, edgecolor='white')

plt.xlabel('Training Prediction Horizon')
plt.ylabel('N-step MSE')
plt.xticks(ind + 1.5*width, ('8', '16', '32', '64', '128', '256'))
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('nstep_mse.eps')
plt.savefig('nstep_mse.png')


ind = np.arange(6)

markers = ['x', '*', '+', 'o']
lines = ['--', '-.', '-', ':']
fig, ax = plt.subplots() # create a new figure with a default 111 subplot
for n, m, l in zip(['GRU', 'LIN', 'RNN', 'SSM'], markers, lines):
    ax.plot(np.arange(6), minnstep.loc[n], label=n, marker=m, linestyle=l)

plt.xlabel('Training Prediction Horizon')
plt.ylabel('N-step MSE')
plt.xticks(range(6), ('8', '16', '32', '64', '128', '256'))
plt.legend(loc='center left', bbox_to_anchor=(0, .35))

axins = zoomed_inset_axes(ax, 2.6, loc=2) # zoom-factor: 2.5, location: upper-left
for n, m, l in zip(['GRU', 'LIN', 'RNN', 'SSM'], markers, lines):
    axins.plot(np.arange(6), minnstep.loc[n], label=n, marker=m, linestyle=l)
plt.xticks(range(6), ('8', '16', '32', '64', '128', '256'))

x1, x2, y1, y2 = 3.75, 5.25, 0.00, 1.05 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits
ax.yaxis.tick_right()
mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
plt.tight_layout()
plt.savefig('nstep_mse_line.eps')
plt.savefig('nstep_mse_line.png')


def plot_matrices(matrices, figname):
    rows = len(matrices)
    cols = len(matrices[0])
    fig, axes = plt.subplots(nrows=rows, ncols=cols, gridspec_kw={'wspace':0, 'hspace':0.05})

    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(matrices[i][j])
            axes[i, j].set(xticklabels= [],
                            xticks = [],
                            yticks=[],
                            yticklabels=[])
    axes[3,0].set_ylabel('RNN')
    axes[0,0].set_ylabel('True')
    axes[1,0].set_ylabel('SSM')
    axes[2,0].set_ylabel('LIN')
    axes[3,0].set_xlabel('$A$')
    axes[3,1].set_xlabel('$E$')
    axes[3,2].set_xlabel('$B$')
    plt.tight_layout()
    plt.savefig(figname+'.eps')
    plt.savefig(figname+'.png')


def plot_trajectories(traj1, traj2, traj3, traj4, labels, figname):

    fig, ax = plt.subplots(len(traj1), 1, gridspec_kw={'wspace':0, 'hspace':0})
    for row, (t1, t2, t3, t4, label) in enumerate(zip(traj1, traj2, traj3, traj4, labels)):
        ax[row].plot(t1, label=f'True')
        ax[row].plot(t2, '--', label='RNN')
        ax[row].plot(t3, '-.', label='LIN')
        ax[row].plot(t4, ':', label='SSM')
        steps = range(0, t1.shape[0] + 1, 288)
        days = np.array(list(range(len(steps))))+7
        ax[row].set(xticks=steps,
                    xticklabels=days,
                    xlim=(0, len(t1)))
        ax[row].set_ylabel(label)
        ax[row].tick_params(labelbottom=False)
        ax[row].axvspan(2016, 4032, facecolor='grey', alpha=0.25, zorder=-100)
        ax[row].axvspan(4032, 6048, facecolor='grey', alpha=0.5, zorder=-100)
    plt.text(1500, 8, "Train", fontsize=8)
    plt.text(3200, 8, "Validation", fontsize=8)
    plt.text(5600, 8, "Test", fontsize=8)
    fig.subplots_adjust(wspace=0, hspace=0)

    ax[row].tick_params(labelbottom=True)
    plt.legend(fontsize=7, labelspacing=0.2, ncol=2, loc='upper left', facecolor='white')
    plt.tight_layout()
    plt.savefig(f'{figname}.pdf')
    plt.savefig(f'{figname}.png')

####################################
###### DATA SETUP
####################################
train_data, dev_data, test_data = make_dataset(16, 'cpu')
print(train_data.shape[0]*train_data.shape[1])
print(test_data.shape[0]*test_data.shape[1])
print(test_data.shape[0]*test_data.shape[1])

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
rnn = RNN(nx, nu, nd)
rnn.load_state_dict(torch.load('models/rnn.pth', map_location=torch.device('cpu')))
ssm = SSM(nx, nu, nd)
ssm.load_state_dict(torch.load('models/ssm_256.pth', map_location=torch.device('cpu')))

lin = LIN(nx, nu, nd)
lin.load_state_dict(torch.load('models/lin.pth', map_location=torch.device('cpu')))
building = Building()
matrices = [[np.asarray(building.A), np.asarray(building.E), np.asarray(building.B)],
            [ssm.A.effective_W().T.detach().cpu().numpy(),
             ssm.E.weight.cpu().data.numpy(),
             ssm.B.weight.cpu().data.numpy()],
            [lin.A.weight.cpu().data.numpy(), lin.E.weight.cpu().data.numpy(), lin.B.weight.cpu().data.numpy()],
                [rnn.cell.weight_hh.data.numpy(), rnn.cell.weight_ih.data[:, 1:].numpy(), rnn.cell.weight_ih.data[:, :1].numpy()]]

fig = plt.figure()

plot_matrices(matrices, 'parameters')
df = pd.DataFrame(index=['SSM', 'LIN', 'RNN'], columns=['A', 'E', 'B'])
for model, mats in zip(['SSM', 'LIN', 'RNN'], matrices[1:]):
    for i, (m, name) in enumerate(zip(mats, ['A', 'E', 'B'])):
        df.loc[model][name] = np.sqrt(np.sum((m - matrices[0][i])**2))
        np.save(f'models/{name}_{model}.npy', m)
print(df.to_latex(float_format=lambda x: '%.3f' % x))


df = pd.DataFrame(index=['True', 'SSM', 'LIN', 'RNN'], columns=['$\lambda_1$', '$\lambda_2$', '$\lambda_3$', '$\lambda_4$'])
for model, mats, name in zip(['True', 'SSM', 'LIN', 'RNN'], matrices, ['A', 'E', 'B']):
    w, v = LA.eig(mats[0])
    df.loc[model] = w
print(df.to_latex(float_format=lambda x: '%.3f' % x))


criterion = nn.MSELoss()  # we'll convert this to RMSE later

#######################################
######### NSTEP TRAIN RESPONSE ########
#######################################
X_out, Y_out = rnn(x0_in, U_in, D_in)
print(f'RNN nstep_train_loss: {criterion(Y_out, Y_target).item()}')
xpred = X_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, 4)
xtrue = x_response.transpose(0, 1).detach().cpu().numpy().reshape(-1, 4)

X_out, Y_out = rnn(x0_dev, U_dev, D_dev)
print(f'RNN nstep_validation_loss: {criterion(Y_out, Y_target_dev).item()}')
devxpred = X_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, 4)
devxtrue = x_response_dev.transpose(0, 1).detach().cpu().numpy().reshape(-1, 4)

X_out, Y_out = rnn(x0_tst, U_tst, D_tst)
print(f'RNN nstep test loss: {criterion(Y_out, Y_target_tst).item()}')
testxpred = X_out.transpose(0, 1).detach().cpu().numpy().reshape(-1, 4)
testxtrue = x_response_tst.transpose(0, 1).detach().cpu().numpy().reshape(-1, 4)



#######################################
######### OPEN LOOP RESPONSE ####
#######################################
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


trajs = []
for m in [rnn, lin, ssm]:
    openloss, xpred, xtrue = open_loop(m, train_data)
    print(f' Train_open_loss: {openloss}')

    devopenloss, devxpred, devxtrue = open_loop(m, dev_data)
    print(f' Dev_open_loss: {devopenloss}')

    testopenloss, testxpred, testxtrue = open_loop(m, test_data)
    print(f' Test_open_loss: {testopenloss}')
    trajs.append(  [np.concatenate([xpred[:, k], devxpred[:, k], testxpred[:, k]])
                for k in range(xpred.shape[1])])

plot_trajectories([np.concatenate([xtrue[:, k], devxtrue[:, k], testxtrue[:, k]])
                   for k in range(xtrue.shape[1])],
             *trajs,
               ['$x_1$', '$x_2$', '$x_3$', '$x_4$'], 'open_test.png')

linerr = (testxtrue[:, 3] - trajs[1][3][-2016:])**2
ssmerr = (testxtrue[:, 3] - trajs[2][3][-2016:])**2

print(linerr.mean(), ssmerr.mean())
t2, p2 = stats.ttest_ind(linerr, ssmerr)
print("t = " + str(t2))
print("p = " + str(p2))