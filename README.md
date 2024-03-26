# Optimizing_bending_parameter

## User guide
- Data are collected in model and mould_output folder. Model folder contains data related to ABAQUS, mould_output folder contains mould file and bending parameter csv
- Mould are generated using the same method. Test0 is the reference used in the interactive environment
- Don't forget to check that mould_static exists. Otherwise can't gen_curve_and_mould

## Things already been done
- Extract stress information from ODB for each step of each test and build a dataset to store them
- Calculate next step bending parameter based on chosen next step size and previous parameter
- Extraction magnitude of springback from ODB
- Given the stress information and nodes' coordinates, recover the geometric shape of the alluminium part
- Build a naive surrogate model with multi-input (stress distribution and bending parameter) to predict next state stress distribution. Best validation MSE: 30~
- Reorganized data file system
- Implement Genetic Algorithm in the environment, 5 generations with population size 10: from 1.42 to 1.36

## Things to be done
- Finish the reinforcement learning framework with environment being the naive surrogate model
- Write script to let Abaqus run per-step analysis
- Build dataset: from last step stress distribution to spring back
- Build another surrogate model to predict springback from final state stress distribution