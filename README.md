# Biologically-Based Neural Representations Enable Fast Online Shallow Reinforcement Learning

Contributors: Dr. M. Bartlett, Dr. T. Stewart & Dr. J. Orchard <br>
Affiliation: University of Waterloo, ON, Canada

Repository to accompany Bartlett, Stewart & Orchard (2022) "Biologically-Based Neural Representations Enable Fast Online Shallow Reinforcement Learning" CogSci Paper (LINK).

## Requirements:

You will need to have Jupyter Notebook installed in order to run these scripts. Recommended to install [Anaconda](https://www.anaconda.com/products/individual). 

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

## Scripts and How-To Guide:

Download repository as a zip file or clone repository by clicking the "Code" button on the top right. <br>

### Run the Network

You can use the runme.py script to run the network from the command line. 
When running this script, the Actor-Critic Network will try to solve the MiniGrid task using the TD(0) learning rule. 
This learning task will be repeated 5 times.
Plots showing the total reward received in each learning trial of an experiment will be saved as run{i}plot.pdf. 
Plots comparing the actual value of each state visited in the last learning trial of an experiment with the ideal value will be saved as ideal_value_plot{i}.pdf. 

### Experiment:

In order to replicate the experiment, open the "Experiment.ipynb" file in Jupyter Notebook. <br>
Things to change: <br>
* Path to the data folder where you want the data to be saved (call 2)

To run an experiment:
1) Set the path to the data folder where you want the data to be saved (cell 2) -- we recommend using the same forumlaic folder-names as used in the original study to avoid complications with loading the data in the other scripts. Folder-name formula is outlined below.
2) Set the parameters in cell 3 -- these can be copied from the table above 
3) Set 'variable' to the list of values relevant to the parameter to-be-tested
4) Replace the numerical value for the to-be-tested parameter with 'v' (e.g. alpha = v)3
5) Run the notebook
6) Repeat for each parameter

Path to Data:

We recommend setting the data_folder path to: Path('../DATA_FOLDER/'task_rule_rep_hidden_param)

where:
* DATA_FOLDER = the parent folder where you want everything to be saved. Set this to whatever you want
* task = the task being learned:
    * 'MG' for MiniGrid
* rule = the rule used:
    * 'TD0' for TD(0)
    * 'TDLam' for TD($\lambda$)
* rep = the representation used:
    * '1H' for one hot
    * 'SSP' for SSP
    * 'Grid' for grid cells
* param = the parameter that was tested/varied:
    * 'alpha'
    * 'beta'
    * 'gamma'
    * 'lambda'
    * 'neurons'
    * 'sparsity'
    * 'dims'
* hidden = whether or not the hidden layer contained neurons:
    * 'nn' for no neurons
    * 'n' for with neurons

### Data Preprocessing:

After replicating the experiment you will need to trim the data and save it into a dataframe before performing analyses. <br>
Go into the get_data function in cell 4 and change the data_folder to your path (i.e. change the parent folder 'WAT002_RL_Data' to your parent folder). <br>
Also change the path where the new data files will be saved ('data.to_pickle ... '). <br>

Beyond these first 4 cells, there is one cell for each experiment. Run the relevant cells for trimmming the data that you have. 

*Note: the trimming process can take a while so we recommend trimming the data for one experiment at a time, rather than running the full script.
    
### Analysis:

If you have run your own experiments, change data_folder path if required (cell 8).

Run script. <br>
If you want to save the figures, uncomment the relevant lines and change the location and filename (optional).


## Study Overview:

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

| Environment | Rule                     | rep                                          | runs | steps | alpha | beta | gamma | n_neurons| sparsity | lambda | sample_encoders | dims |
| ----------- | -------------            | ---------------------                        | ---- | ----- | ----- | ---- | ----- | -------- | -------- | ------ | --------------- | ---- |
| MiniGrid    | rule.ActorCriticTD(0)    | rp.OneHotRep((8,8,4))                        | 500  | 200   | 0.5   | 0.9  | 0.95  | None     | None     | None   | False           | None |
| MiniGrid    | rule.ActorCriticTD(0)    | rp.OneHotRep((8,8,4))                        | 1000 | 200   | 0.5   | 0.8  | 0.8   | 3000     | 0.1      | None   | False           | None |
| MiniGrid    | rule.ActorCriticTD(0)    | rp.SSPRep(N=3, D=128, scale=[0.75,0.75,1.0]) | 300  | 200   | 0.5   | 0.6  | 0.8   | 3000     | 0.25     | None   | False           | 128  |
| MiniGrid    | rule.ActorCriticTD(0)    | rp.GridSSPRep(3)                             | 300  | 200   | 0.1   | 0.85 | 0.95  | 1000     | 0.1      | None   | False           | None |
| ----------- | -------------            | ---------------------                        | ---- | ----- | ----- | ---- | ----- | -------- | -------- | ------ | --------------- | ---- |
| MiniGrid    | rule.ActorCriticTDLambda | rp.OneHotRep((8,8,4))                        | 300  | 200   | 0.1   | 0.9  | 0.95  | None     | None     | 0.9    | False           | None |
| MiniGrid    | rule.ActorCriticTDLambda | rp.OneHotRep((8,8,4))                        | 300  | 200   | 0.1   | 0.85 | 0.85  | 2000     | 0.005    | 0.8    | False           | None |
| MiniGrid    | rule.ActorCriticTDLambda | rp.SSPRep(N=3, D=256, scale=[0.75,0.75,1.0]) | 500  | 200   | 0.1   | 0.9  | 0.7   | 5000     | 0.2      | 0.5    | False           | 256  |
| MiniGrid    | rule.ActorCriticTDLambda | rp.GridSSPRep(3)                             | 50   | 200   | 0.1   | 0.85 | 0.95  | 2000     | 0.2      | 0.9    | False           | None |
| ----------- | -------------            | ---------------------                        | ---- | ----- | ----- | ---- | ----- | -------- | -------- | ------ | --------------- | ---- |

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

Running a single experiment will produce a large number of files which takes a while to load up. We therefore created the Data_PreProcessing script to collect all the data from each folder, and create smaller, more manageable dataframes containing the information necessary for the analysis. <br>
*Note : preprocessing each folder of data may take a while*

## Analysis:

The analysis for this study was descriptive in nature. 
The Analysis script creates plots showing the mean number of trials needed to reach the target rolling average reward of 0.95. 
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

