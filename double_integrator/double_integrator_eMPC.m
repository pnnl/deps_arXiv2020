
%% Double integrator model
% system from slide 72
% https://engineering.utsa.edu/ataha/wp-content/uploads/sites/38/2017/10/MPC_Intro.pdf

yalmip('clear')
clear all
% Model data
A = [1.2 1;0 1];
B = [1;0.5];  
% C = [1 1];
C = [1 0; 0 1];
nx = 2; % Number of states
nu = 1; % Number of inputs
ny = 2; % Number of outputs
% Prediction horizon
N = 10;

% constraints
umin = -1;
umax = 1;
ymin = 5;
ymax = 5;
xmin = -10;
xmax = 10;

% objective weights
Qy = 1;
Qu = 1;

% variables
x = sdpvar(nx, N+1, 'full');        % MPC parameter
u = sdpvar(nu, N, 'full');          % decision variables
y = sdpvar(ny, N, 'full');          % internal variable

%% Optimization problem

% explicit MPC policy
run_eMPC = 1;

con = [];
obj = 0;
for k = 1:N
    con = con + [x(:, k+1) == A*x(:, k) + B*u(:, k)];       % state update model
%     con = con + [y(:, k) == C*x(:, k+1) ];         % output model
    con = con + [ xmin <= x(:, k) <= xmax ];
%     con = con + [ ymin <= y(:, k) <= ymax ];
    con = con + [ umin <= u(:, k) <= umax ];                    % input constraints
    
        % objective function
    obj = obj + x(:, k)'*Qy*x(:, k) + u(:, k)'*Qu*u(:, k);
end

options = sdpsettings('verbose', 0,'solver','QUADPROG','QUADPROG.maxit',1e6);


if run_eMPC
    % eMPC optimizer
    plp = Opt(con, obj, [x(:, 1)], u(:, 1));
    solution = plp.solve();
    filename = strcat('eMPC_di');
    solution.xopt.toMatlab(filename, 'primal', 'obj')  %  generate standalone m-file with the eMPC policy
    
%     figure;
%     solution.xopt.plot()
    figure;
    solution.xopt.fplot('primal');
    figure;
    solution.xopt.fplot('obj');
    xlabel('t');
    ylabel('J(t)');
    
%     [sol,diagn,Z,Valuefcn,Optimizer] = solvemp(con,obj ,[],[x(:, 1)], u(:, 1));
%     figure;plot(Valuefcn)
%     figure;plot(Optimizer)
else
    % % iMPC optimizer
    opt = optimizer(con, obj, options, x(:, 1), u);
end


%% Simulation Setup


% initial conditions
x0 = 1.5*ones(nx,1);
Nsim = 50;
Xsim = x0;
Usim = [];
Ysim = [];
for k = 1:Nsim
    
    x_k = Xsim(:, end);
%     uopt = eMPC_policy(x_k);
    uopt = solution.xopt.feval(x_k, 'primal');
    xn = A*x_k + B*uopt;
    yn = C*x_k;
    Usim = [Usim, uopt];
    Xsim = [Xsim, xn];
    Ysim = [Ysim, yn];
end

%% Plots
% close all
t = 0:Nsim-1;

figure
subplot(2,1,1)
plot(t,Xsim(:,1:end-1),'LineWidth',2)
legend('x_1', 'x_2')
xlabel('time')
subplot(2,1,2)
stairs(t,Usim,'LineWidth',2)
title('Input')
legend('u')
xlabel('time')

% TODO: plots with multiple initial conditions
figure
% solution.xopt.plot()
% hold on
plot(Xsim(1,:),Xsim(2,:),'LineWidth',2)
grid on
xlim([-1 7])
ylim([-4 4])

[X1,X2] = meshgrid(-5:.1:5);
U = nan(length(X1),length(X1));
for i=1:length(X1)
    for j=1:length(X1)
        U(i,j) = solution.xopt.feval([X1(i,j); X2(i,j)], 'primal'); 
    end
end
figure
s = surf(X1,X2,U)
xlabel('x_2')
ylabel('x_1')
zlabel('u')
s.EdgeColor = 'none';



