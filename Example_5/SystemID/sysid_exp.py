import os

steps = [8, 16, 32, 64, 128, 256]
for lr in [0.001, 0.03, 0.01]:
    for k in range(10):
        for model in ['ssm', 'lin', 'rnn', 'gru']:
            for step in steps:
                os.system(f'python system_id.py -lr {lr} -nsteps {step} '
                          f'-cell_type {model} -exp sysid_{model}_{step} -run {k}_{lr}_{model}')
