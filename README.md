# Differentiable Predictive Control 
Examples of the differentiable predictive control (DPC) policy optimization algorithm presented in the paper "Learning Constrained Adaptive Differentiable Predictive Control Policies With Guarantees"
https://arxiv.org/abs/2004.11184

DPC combines the principles of model predictive control, reinforcement learning, and differentiable programming to offer a systematic way for offline unsupervised model-based policy optimization with goal-parametrized domain-aware intrinsic rewards.


## Method and Examples

![methodology.](Example_1/test_control/DPC_abstract.png)  
*Conceptual methodology. Simulation of the differentiable closed-loop system dynamics in the forward pass is followed by backward pass computing direct policy gradients for policy optimization *


![methodology_2.](Example_1/test_control/deep_MPC_var2.png)  
*Structural equivalence of DPC architecture with MPC constraints.*


![cl_trajectories.](Example_1/test_control/cl_animation.gif)  
*Example 1: Closed-loop trajectories of learned stabilizing neural control policy using DPC policy optimization.*

![cl_trajectories_2.](Example_1/test_control/closed loop policy training.gif)  
*Example 1: Evolution of the closed-loop trajectories and DPC neural policy during training.*


![dpc_policy.](Example_1/test_control/policies_surfaces.png)  
*Example 1: Landscapes of the learned neural policy via DPC policy optimization algorithm (right) and explicit MPC policy computed using parametric programming solver (left).*


![example_2_dpc.](Example_2/figs/pvtol_dpc_cl1.png)  
*Example 2: Closed-loop control trajectories for the PVTOL aircraft model controlled by DPC neural policy.*


![example_2_ampc.](Example_2/figs/ampc_1.png)  
*Example 2: Closed-loop control trajectories for the PVTOL aircraft model controlled by approximate MPC neural policy.*


![example_3.](Example_3/figs/quadcopter_dpc_1.png)  
*Example 3: Closed-loop reference tracking control trajectories for the quadcopter model controlled by DPC neural policy.*


![example_4.](Example_4/figs/obstacle_avoidance_DPC.png)  
*Example 4: Obstacle avoidance with nonlinear constraints via learned DPC neural policy compared to online IPOPT solution.*



![example_5.](Example_5/figs/DeepMPC_simSysID_on_paper.png)  
*Example 5: Adaptive DPC of unknown linear system subject to disturbances.*



## Dependencies
For examples 1, 2, 3, and 4 have been implemented using our newly developed Neuromancer library for learning-based constrained optimization in Pytorch: 
[neuromancer](https://github.com/pnnl/neuromancer/).

See environment.yml to reproduce the Conda environment for running example 5. 


## Files for Running the Examples

### Control Example 1 
- double_integrator_DPC.py - DPC double integrator example using the Neuromancer package
- double_integrator_eMPC.m - explicit MPC benchmark using MPT3 toolbox

### Control Example 2
- vtol_aircraft_DPC_stabilize.py - Unsupervised DPC policy optimization for VTOL aircraft model using the Neuromancer package
- vtol_aircraft_aMPC.py - Approximate MPC supervised by online MPC solver
- pvtol_aircraft_iMPC.m - Online MPC solved in Matlab using Yalmip toolbox and quadprog solver

### Control Example 3
- quad_3D_linearDPC.py - Reference tracking for a quadcopter model via DPC using the Neuromancer package
- CVXPY_linearMPC_quadcopter.py - Reference tracking for a quadcopter model online MPC using CVXPY and OSQP solver

### Control Example 4
- 2D_obstacle_avoidance_DPC.py - Parametric obstacle avoidance with nonlinear constraints via DPC using the Neuromancer package
- 2D_obstacle_avoidance_csadi.py - Online obstacle avoidance using CasADi and IPOPT solver

### Control Example 5
- DeepMPC_sysID_ctrl_sec_2_4.py - policy optimization with ground truth model 
- DeepMPC_sysID_ctrl_sec_2_5.py - adaptive policy optimization via online simultaneous system ID and policy updates 
- DeepMPC_sysID_ctrl_sec_3_7 	- computational aspects and scalability analysis

## Cite as

```yaml
@misc{drgona2022learning,
      title={Learning Constrained Adaptive Differentiable Predictive Control Policies With Guarantees}, 
      author={Jan Drgona and Aaron Tuor and Draguna Vrabie},
      year={2022},
      eprint={2004.11184},
      archivePrefix={arXiv},
      primaryClass={eess.SY}
}
```