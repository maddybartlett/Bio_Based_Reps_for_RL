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
-- Note: these scripts can also be used to implement TD(n), but this rule was not tested in the published study.

## Experiment:

## Data Preprocessing:

## Analysis:

## Citation:

Paper:

<pre>
citation
</pre>

Repository:

<pre>
citation
</pre>
