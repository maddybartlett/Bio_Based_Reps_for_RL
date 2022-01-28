import gym_minigrid
from gym_minigrid.wrappers import *

class MiniGrid_wrapper(gym.Env):
    '''Wrapper for the mini-grid environment.
    This allows us to access the agent's state in the same way as we would with 
    other gym environments.
    '''
    def __init__(self):
        self.orig_env = gym.make('MiniGrid-Empty-8x8-v0') #call minigrid environment
        self.action_space = spaces.Discrete(3) #self.orig_env.action_space #call minigrid actionspace
        #self.observation_space = [8,8,4] #set observation space
        self.orig_env.reset() #reset environment
        
        self.min_x_pos = 0
        self.max_x_pos = 7
        self.min_y_pos = 0
        self.max_y_pos = 7
        self.min_dir = 0
        self.max_dir = 3

        self.low = np.array([self.min_x_pos, self.min_y_pos, self.min_dir], dtype=np.float32)
        self.high = np.array([self.max_x_pos, self.max_y_pos, self.max_dir], dtype=np.float32)

        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
    
    def reset(self):
        self.orig_env.reset() #reset environment
        #get agent's state (x position, y position, direction)
        obs = (self.orig_env.agent_pos[0], self.orig_env.agent_pos[1], self.orig_env.agent_dir)
        return obs #return state
        
    def step(self, a):
        blah, reward, done, info = self.orig_env.step(a) #do step in environment
        #get agent's state (x position, y position, direction)
        obs = (self.orig_env.agent_pos[0], self.orig_env.agent_pos[1], self.orig_env.agent_dir)
        #return state, reward, done and info
        return obs, reward, done, info