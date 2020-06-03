# deps_arXiv2020
Prototype code for the paper: Constrained Physics-Informed Deep Learning for Stable System Identification and Control of Unknown Linear Systems

## Dependencies: Python Libraries
See environment.yml to reproduce the Conda environment for running experiments. For GPU capabilities 
install gpu version of Pytorch. 

## Files
### Models
- GroundTruthSSM.py: Ground truth system model
### Control
- DeepMPC_sysID_ctrl_sec_2_4.py - policy optimization with ground truth model Section 2.4
- DeepMPC_sysID_ctrl_sec_2_5.py - simultaneous system ID and policy optimization Section 2.5
- DeepMPC_sysID_ctrl_sec_3_7 	- computational aspects and scalability analysis results for Section 3.7

### SystemID
- results_and_analysis.py: Generate tables and figures related to system identification experiments
- sysid_exp.py: System ID experiments described in paper
- system_id.py: Class definitions for RNN, GRU, LIN, and SSM models, optimization code for system identification training

### Control Benchmarks
- LQR
- LQI
- nominal, stochastic, and robust MPC - formulations defined in the paper: https://ieeexplore.ieee.org/document/6760908
