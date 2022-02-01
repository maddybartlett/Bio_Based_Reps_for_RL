# Biologically-Based Neural Representations Enable Fast Online Shallow Reinforcement Learning

Contributors: Dr. M. Bartlett, Dr. T. Stewart & Dr. J. Orchard <br>
Affiliation: University of Waterloo, ON, Canada

Repository to accompany Bartlett, Stewart & Orchard (2022) "Biologically-Based Neural Representations Enable Fast Online Shallow Reinforcement Learning" CogSci Paper (LINK).

## Requirements:

* Python 3.5+
* OpenAI Gym
* Gym MiniGrid
* PyTry
* Nengo
* Numpy
* Pandas
* Pickle
* Matplotlib
* SciPy
* Pathlib
* Math

## Overview:

This study explored the effect of using different methods of representing the state on the performance of a Temporal Difference-based Actor-Critic network on solving the MiniGrid Reinforcement Learning task. 
Specifically, it examines whether performance when using biologically-inspired representations (Spatial Semantic Pointers and grid cells) differs from performance when using one-hot representations. 

A total of 4 representations were tested and compared:
1) baseline - one-hot without neurons
2) one-hot 
3) Spatial Semantic Pointers (SSPs)
4) grid cells

Additionally, we implemented two Temporal-Difference (TD) learning rules:
1) TD(0)
2) TD(&lambda;) <br>
*Note: these scripts can also be used to implement TD(n), but this rule was not tested in the published study.*

## Experiment:

Before running any experiments we first identified a working parameter set for each configuration -- a set of parameters that resulted in the network solving the task in at least 2 out of 5 runs.
The chosen parameter sets that came out of this were:

| Environment | Rule          | rep                                       | runs | steps | alpha | beta | gamma | n_neurons| sparsity | lambda | sample_encoders | dims |
| ----------- | ------------- | ---------------------                     | ---- | ----- | ----- | ---- | ----- | -------- | -------- | ------ | --------------- | ---- |
| MiniGrid    | TD(0)         | OneHotRep((8,8,4))                        | 500  | 200   | 0.5   | 0.9  | 0.95  | None     | None     | None   | False           | None |
| MiniGrid    | TD(0)         | OneHotRep((8,8,4))                        | 1000 | 200   | 0.5   | 0.8  | 0.8   | 3000     | 0.1      | None   | False           | None |
| MiniGrid    | TD(0)         | SSPRep(N=3, D=256, scale=[0.75,0.75,1.0]) | 500  | 200   | 0.5   | 0.6  | 0.7   | None     | None     | None   | False           | 256  |
| MiniGrid    | TD(0)         | SSPRep(N=3, D=128, scale=[0.75,0.75,1.0]) | 300  | 200   | 0.5   | 0.6  | 0.8   | 3000     | 0.25     | None   | False           | 128  |
| MiniGrid    | TD(0)         | GridSSPRep(3)                             | 300  | 200   | 0.1   | 0.85 | 0.95  | None     | None     | None   | False           | None |
| MiniGrid    | TD(0)         | GridSSPRep(3)                             | 300  | 200   | 0.1   | 0.85 | 0.95  | 1000     | 0.1      | None   | False           | None |
| ----------- | ------------- | ---------------------                     | ---- | ----- | ----- | ---- | ----- | -------- | -------- | ------ | --------------- | ---- |
| MiniGrid    | TD($\lambda$) | OneHotRep((8,8,4))                        | 300  | 200   | 0.1   | 0.9  | 0.95  | None     | None     | 0.9    | False           | None |
| MiniGrid    | TD($\lambda$) | OneHotRep((8,8,4))                        | 300  | 200   | 0.1   | 0.85 | 0.85  | 2000     | 0.005    | 0.8    | False           | None |
| MiniGrid    | TD($\lambda$) | SSPRep(N=3, D=256, scale=[0.75,0.75,1.0]) | 500  | 200   | 0.1   | 0.9  | 0.7   | None     | None     | 0.5    | False           | 256  |
| MiniGrid    | TD($\lambda$) | SSPRep(N=3, D=256, scale=[0.75,0.75,1.0]) | 500  | 200   | 0.1   | 0.9  | 0.7   | 5000     | 0.2      | 0.5    | False           | 256  |
| MiniGrid    | TD($\lambda$) | GridSSPRep(3)                             | 50   | 200   | 0.1   | 0.85 | 0.95  | None     | None     | 0.9    | False           | None |
| MiniGrid    | TD($\lambda$) | GridSSPRep(3)                             | 50   | 200   | 0.1   | 0.85 | 0.95  | 2000     | 0.2      | 0.9    | False           | None |
| ----------- | ------------- | ---------------------                     | ---- | ----- | ----- | ---- | ----- | -------- | -------- | ------ | --------------- | ---- |

We then conducted a parameter survey where we varied these parameters over a wide range of values, varying one parameter at a time. 

The parameters that were varied, and the values tested, were:

| Parameter                           | Testing Values                                                                          |
| ----------------------------------- | --------------------------------------------------------------------------------------- |
| Alpha (learning rate)               | 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0                         |
| Beta (action-value discount)        | 0.01, 0.1, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99                        |
| Gamma (state-value discount)        | 0.01, 0.1, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99                        |
| Number of Neurons                   | 10, 100, 500, 1000, 1500, 2000, 2500, 3000, 5000                                        |
| Sparsity                            | 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99 |
| Lambda (eligibility trace discount) | 0.01, 0.1, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99                        |
| Number of Dimensions                | 64, 128, 256, 532                                                                       |

## Data Preprocessing:

Running a single experiment will produce a large number of files which takes a while to load up. We therefore created the Data_PreProcessing script to collect all the data from each folder, and create smaller, more manageable dataframes containing the information necessary for the analysis. 
*Note : preprocessing each folder of data may take a while*

## Analysis:

The analysis for this study was descriptive in nature. <br>
The Analysis script creates plots showing the mean number of trials needed to reach the target rolling average reward of 0.95. <br>
This mean number of trials was calculated for each testing value of each parameter that was varied. <br>
The plots show the distribution of mean number of trials across all testing values for each of the 4 representations. 

We also retrieved the parameter sets that resulted in the smallest mean number of trials to reach the target rolling average reward and present this information in a table along with the mean number of trials, and 95% confidence intervals for that mean. <br>
*Note: the dimensions data has been excluded because there was little to no effect of varying this parameter (at least across the values we tested).*

## Citation:

Please use this bibtex to reference the paper: 

<pre>
<!-- @inproceedings{bartlett2022_RL,
  author = {Bartlett, Madeleine and Stewart, Terrence C and Orchard, Jeff},
  title = {Biologically-Based Neural Representations Enable Fast Online Shallow Reinforcement Learning},
  year = {2022},
  booktitle={44th Annual Conference of the Cognitive Science Society (CogSci 2022)},
 } -->
</pre>

