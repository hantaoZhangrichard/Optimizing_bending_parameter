# Optimizing_bending_parameter

## Things already been done
- Extract stress information from ODB for each step of each test and build a dataset to store them
- Calculate next step bending parameter based on chosen next step size and previous parameter
- Extraction magnitude of springback from ODB
- Given the stress information and nodes' coordinates, recover the geometric shape of the alluminium part
- Build a naive surrogate model with multi-input (stress distribution and bending parameter) to predict next state stress distribution. Best validation MSE: 30~

## Things to be done
- 