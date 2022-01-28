# Create State Representations
'''Here we define a few functions for representing the state in different ways: 
    (1) Using normalized representation
    (2) Using one-hot representation - acts like look-up tables
    (3) Using nengo Spatial Semantic Pointers (SSP) '''

import numpy as np
import nengo_spa
import grid_cells
import math
import gym
import minigrid_wrap

class NormalRep(object):
    '''Create look-up tables of size: 
    (size_state_dim_1, size_state_dim_2, ..., size_state_dim_n, n_actions).
    
    Inputs: agent's state
    Params: environment
    Outputs: representation of agent's state
    
    Examples of Usage:
    Initialize the representation:
        >> representation = NormalRep('MiniGrid')
                    
    Translate agent state into representation:
        >> state = (0,7,3)  
        >> representation.map(state) 
        ans = array([-1.  ,  0.75,  0.5 ])
    '''
    def __init__(self, env):
        ##Set environment
        if env == 'MiniGrid':
            self.env = minigrid_wrap.MiniGrid_wrapper()
        elif env == 'MountainCar':
            self.env = gym.make("MountainCar-v0")
        elif env == 'CartPole':
            self.env = gym.make('CartPole-v1')
        elif env == 'LunarLander':
            self.env = gym.make('LunarLander-v2')
            
        self.upper = self.env.observation_space.high
        self.lower = self.env.observation_space.low
        
        self.ranges = self.upper - self.lower
        self.size_out = len(self.ranges) 
        
    ##Normalize the state into values between -1 and 1   
    def map(self, state):
        state = state + abs(self.lower)
        norm_state = (state/self.ranges-0.5)*2
        
        #Check that the state has been normalised properly
        if np.any(norm_state > 1) or np.any(norm_state < -1) :
            print ("Representation Error: State values outside of normalisation bounds (-1, 1): ", state)
        
        return norm_state
    
    def get_state(self, state, env):
        discrete_state = state            
        return discrete_state
    
class OneHotRep(object):
    '''Create one-hot representation. I.e. the state is represented as a list of 0's and a 1. 
    The position of the 1 in the list is the state that the agent is in.
    To translate the state into this list:
    (1) Calculate an array of the products of each pair of elements in ranges (excluding the first value) (self.factors)
    (2) Create an empty array whose size = the product of all elements in ranges (self.result)
    (3) Multiply the state values for the agent's state by the factor array and sum the result to get a single value
    which falls between 0 and len(result) 
    (4) This value is the index for the position of the '1' in the array of 0's which acts as the state representation  
    
    Inputs: agent's state
    Params: size of state space
    Outputs: representation of agent's state
    
    Examples of Usage:
    Initialize the representation:
        >> representation = OneHotRep((8,8,4))
                    
    Translate agent state into representation:
        >> state = (0,7,3)  
        >> representation.map(state)   
        ans = array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   0.])
    '''
    def __init__(self, ranges):
        ##step 1
        self.ranges = ranges
        self.factors = np.array([np.prod(ranges[x+1:]) for x in range(len(ranges))], dtype=int)
        ##step 2
        self.result = np.zeros(np.prod(ranges))
        self.size_out = len(self.result)
    def map(self, state):
        index = 0
        ##step 3
        for i,v in enumerate(state):
            index += v*self.factors[i]
        self.result[:] = 0
        ##step 4
        self.result[index] = 1
        return self.result    
    
    def get_state(self, state, env):
        ''' First find the edges of the state space 
        (i.e. the maximum and minimum values for each dimension describing the agent's state).

        Second, define the number of bins (Discrete_obs_size) and then define the size of
        each bin (Discrete_obs_win_size).

        Having defined the shape of our state space we can now take the current state of the agent and 
        translate the state values into values which will map on to our discrete space. The resultant 
        value can then be used as an index for accessing the right values in the look-up tables.'''
        
        #MiniGrid
        if len(state) == 3:
            discrete_state = state
            
        #MountainCar
        elif len(state) == 2:
            Obs_state_space_high = env.observation_space.high
            Obs_state_space_low = env.observation_space.low
            Discrete_obs_size = np.array(self.ranges)

            Discrete_obs_size = np.array(Discrete_obs_size, dtype='int')
            Discrete_obs_win_size = (Obs_state_space_high - Obs_state_space_low)/Discrete_obs_size

            discrete_state = (state - Obs_state_space_low)/Discrete_obs_win_size
            discrete_state = tuple(discrete_state.astype(np.int))
        
        #CartPole
        elif len(state) == 4:
            #extract the upper and lower limits of the observation space
            Obs_state_space_high = np.array([env.observation_space.high[0], 0.5, env.observation_space.high[2], 0.9])
            Obs_state_space_low = np.array([env.observation_space.low[0], -0.5, env.observation_space.low[2], -0.9])

            #get the size of each dimension space
            Discrete_obs_win_size = Obs_state_space_high - Obs_state_space_low

            #Add the absolute lower bound values to the state values 
            #this has the effect of treating the lower bounds as 0.
            #Divide by the size of the dimensions to work out the 
            #position of the state values in the dimension windows 
            Pos_in_win = state + abs(Obs_state_space_low)/Discrete_obs_win_size

            #Multiply the position values by the number of buckets (minus 1)
            #to work out which bucket the position belongs in
            discrete_state = [round(Pos_in_win[i] * (self.ranges[i] - 1)) for i in range(len(state))]

            #Force the values within the range of available buckets
            discrete_state = [min(self.ranges[i] - 1, max(0, discrete_state[i])) for i in range(len(state))]

        return discrete_state
        
        
class SSPRep(object):
    '''Create state representation using nengo's Spatial Semantic Pointers (SSP).
    Semantic pointers are neural representations that carry partial semantic content.
    These representations capture relations in a semantic vector space in virtue of 
    their distances to one another.
    
    Inputs: agent's state
    Params: N = number of dimensions in state space, 
            D = number of dimensions for representing the state space,
            scale
    Outputs: representation of agent's state as ssp
    
    
    Examples of Usage:
    Initialize the representation:
        >> representation = SSPRep(N=3, D=256, scale=[0.5,0.5,1.0])
                    
    Translate agent state into representation:
        >> state = (0,7,3)  
        >> representation.map(state)   
        ans = array([-5.62878534e-02, -4.07633639e-02,  7.21655501e-02, -7.44510892e-02,
                    ..., 2.19691553e-02, -1.60385311e-02, -8.02381664e-02, -4.76827969e-02]) 
    (Note: len(ans) = 256)
    '''
    def __init__(self, N, D=256, scale=None):
        ##create collection of semantic pointers, each with D dimensions
        vocab = nengo_spa.Vocabulary(D)
        ##create pointer for each dimension of the state space
        self.Vs = [vocab.create_pointer().unitary() for i in range(N)]
        self.size_out = D
        
        ##Scaling factor for adjusting generalisation
        ##i.e. how wide a range of values is considered similar vs completely different
        if scale is None:
            scale = np.ones(N)
        self.scale = scale
        
    def power(self, s, e):
        ''' s = semantic pointer for one state dimension
            e = state value (for same state dimension) * scale value (for same state dimension)
        Below equation is effectively the circular convolution'''
        x = np.fft.ifft(np.fft.fft(s.v) ** e).real
        return nengo_spa.SemanticPointer(data=x)
    
    def map(self, state):
        '''Translate state into SSP using circular convolution'''
        r = self.power(self.Vs[0], state[0]*self.scale[0])
        for i in range(1, len(state)):
            r = r * self.power(self.Vs[i], state[i]*self.scale[i])
        return r.v
    
    def get_state(self, state, env):
        return state
    
class GridSSPRep(object):
    '''Create state representation with grid-cells'''
    
    def __init__(self, N, scales=np.linspace(0.5, 2, 8), n_rotates=8, hex=True):
        self.basis = grid_cells.GridBasis(dimensions=N, scales=scales, n_rotates=n_rotates, hex=hex)
        self.size_out = self.basis.axes.shape[1]
    
    def map(self, state):
        return self.basis.encode(state)
    
    def make_encoders(self, n_neurons):
        return self.basis.make_encoders(n_neurons)
    
    def get_state(self, state, env):
        return state