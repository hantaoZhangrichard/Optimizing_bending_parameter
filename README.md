# Optimizing_bending_parameter

## User guide
This repository includes almost everything you need for optimizing the bending parameter to reduce the springback deviation in the process of alluminum bending production process. \\
- How to run bending process simulation on ABAQUS:
    1. Run gen_curve_and_mould.py. This will create a mould and the feature lines.
    2. Run calc_init_param.py. This will generate a set of bending parameter using a gready algorithm.
    3. Run gen_abaqus_model.py. This will generate a bending job inp file for ABAQUS software and then automatically submit the job to run finite element analysis.
    4. Run gen_spring_back_model.py. This will generate a springback job inp file for ABAQUS and automatically submit it to get springback.
    5. Or you can run automation.py to run entire simulation in one shot.

- Details for calculating bending parameters from the involute feature lines are in the core/param_util folder. 
- Data are collected in model and mould_output folder. Model folder contains data from ABAQUS, mould_output folder contains mould file and bending parameter csv. ABAQUS outputs are collected in the .obd file.
- Mould are generated using the same method. Test0 is the reference used in the interactive environment rl_env.py.
- Don't forget to check that mould_static exists. Otherwise can't run gen_curve_and_mould.py.

## Things already been done
- Extract stress information from ODB for each step of each test and build a dataset to store them
- Calculate next step bending parameter based on chosen next step size and previous parameter
- Extraction magnitude of springback from ODB
- Given the stress information and nodes' coordinates, recover the geometric shape of the alluminium part
- Build a naive surrogate model with multi-input (stress distribution and bending parameter) to predict next state stress distribution. Best validation MSE: 30~
- Reorganized data file system
- Implement Genetic Algorithm in the environment, 5 generations with population size 10: from 1.42 to 1.36
- Build another surrogate model to predict springback from final state stress distribution

## Things to be done
- Finish the reinforcement learning framework with environment being the naive surrogate model
- Write script to let Abaqus run per-step analysis
- Build dataset: from last step stress distribution to spring back
- Consider the design of reward function. Now just use max springback.