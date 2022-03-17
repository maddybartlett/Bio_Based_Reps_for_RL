import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special
from pathlib import Path

import nengo
import learnrules as rules
import representations as rp
import minigrid_wrap
from ac_learn import ActorCriticLearn

#Set path for data 
data_folder = Path('./data/debug')

#Set number of runs
runs = 5
    
ep_results = []
ep_rewards=[]
ep_values=[]
    
#Run experiment
for run in range(runs):
    for attempt in range(5):
        try:
            out = ActorCriticLearn().run(env='MiniGrid',
                                         rep=rp.OneHotRep((8,8,4)),
                                         trials = 10000,
                                         steps = 200,
                                         rule=rules.ActorCriticTD0,
                                         alpha = 0.5, 
                                         beta = 0.9, 
                                         gamma = 0.95, 
                                         n_neurons = None,
                                         sparsity = None,
                                         sample_encoders = 'False',
                                         lambd = None,
                                         verbose = False,
                                         seed = run,
                                         dims = None,
                                         data_dir = data_folder, 
                                         data_format = "npz")
            if run == 0:
                Results_df = pd.DataFrame([out])
            else:
                Results_df.loc[len(Results_df.index)] = out
            ep_results.append(out["episodes"])
            ep_rewards.append(out["rewards"])
            ep_values.append(out["values"])
            print("Finished test number ", run+1)
        except (FloatingPointError, ValueError):
            print('NaNs found. Starting again')
            continue
        else: break
    else: 
        print('Could not do it. Value we could not test: ', v)
        break
            
for i in range(len(ep_results)):
    plt.figure()
    plt.plot(ep_results[i])
    plt.xlabel('Trial Number')
    plt.ylabel('Total Trial Reward')
    plt.savefig('./run{i}plot.pdf'.format(i=i+1))
    
#Plot ideal value given discount value
for i in range(len(ep_rewards)):
    def value(discount):
        d = discount**t
        return np.convolve(d[::-1], ep_rewards[i][-1])[-len(t):]
    T = len(ep_rewards[i][-1])#[:-n])
    t = np.arange(0, int(T))

    #Plot ideal value against actual value in final run
    plt.figure(figsize=(15, 3))
    plt.plot(t, value(0.95), label='ideal value')
    plt.plot(ep_values[i][-1], label='actual value')
    plt.legend()
    plt.xlabel('Timestep')
    plt.ylabel('State Value')
    plt.savefig('./ideal_value_plot{i}.pdf'.format(i=i+1))