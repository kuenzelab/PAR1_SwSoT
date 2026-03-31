# PAR1 - Energy Landscape Sampling Reveals Ligand-Dependent Structural Dynamics

This repository contains all scripts and exemplary data regarding the study "Energy Landscape Sampling Reveals Ligand-Dependent Structural Dynamics of the Protease-Activated Receptor 1". It is based on the [String Method with Swarms of Trajectories](https://github.com/delemottelab/string-method-swarms-trajectories) implementation by the Delemottelab.

## Adjustments in this study

* Instead of steered MD, targeted MD was used to create the first string
* The number of beads was increased stepwise (+1) every eigth iteration from 20 to 40
* Instead of the output structure of the previous iterations restrained simulation, any output conformation of the last iteration that is nearest to the new string point was used for the next iterations restrained simulation
* Further postprocessing scripts are added    

All this implementations are part of the modified code in this repository.

## Exemplary data

For each of the simulations done in this study, the restrained output conformations of the last iteration (300) are available under simulations. The PAR1_apo folder aditionally contains the starting conformations taken from targeted simulations (0).