import nengo
import numpy as np

## TD(0) Learning Rule

class ActorCriticTD0(nengo.processes.Process):
    '''Create nengo Node with input, output and state.
    This node is where the TD(0) learning rule is applied.
    
    Inputs: state, reward, chosen action, whether or not the environment was reset
    Outputs: updated state value, updated action values
    '''
    def __init__(self, n_actions, n=None, gamma = 0.95, alpha = 0.1, beta = 0.9, lambd=None):
        self.n_actions = n_actions ##number of possible actions for action-value space
        self.gamma = gamma ##discount factor for state values
        self.alpha = alpha ##learning rate
        self.beta = beta ##discount factor for action values
        
        ##Input = reward + state representation + action values for current state
        ##Output = action values for current state + state value
        super().__init__(default_size_in=n_actions + 2, default_size_out=n_actions + 1) 
        
    def make_state(self, shape_in, shape_out, dt, dtype=None):
        '''Get a dictionary of signals to represent the state of this process.
        This will include: the representation of the state being updated (update_state_rep),
        the initial value of the state (0) (update_value), and the weight matrix/look-up table (w).
        Weight matrix shape = [(n_actions + state) * size of state representation output]'''
        
        ##set dim = length of each row in matrix/look-up table 
        ##where a nengo ensemble is used to store the state representation, 
        ##this is equal to the number of neurons in the ensemble
        dim = shape_in[0]-2-self.n_actions
        ##return the state dictionary
        return dict(update_state_rep=np.zeros(dim),
                    update_value=0,
                    w=np.zeros((self.n_actions+1, dim)))
    
    def make_step(self, shape_in, shape_out, dt, rng, state):
        '''Create function that advances the process forward one time step.'''
        ##set dim = length of each row in matrix/look-up table 
        ##where a nengo ensemble is used to store the state representation, 
        ##this is equal to the number of neurons in the ensemble
        dim = shape_in[0]-2-self.n_actions
        
        def step_TD0(t, x, state=state):
            ''' Function for performing TD(0) update at each timestep.
            Inputs: state representation, chosen action, reward
                and whether or not the env was reset
            Params:
                t = time
                x = inputs 
                state = the dictionary created in make_state.
            Outputs: updated state value, updated action values'''
            
            current_state_rep = x[:dim] ##get the state representation of current state
            update_state_rep = state['update_state_rep'] ##get representation of state being updated
            update_action = x[dim:-2] ##get the action being updated (i.e. the action that was taken)
            reward = x[-2] ##get reward received following action
            reset = x[-1] ##get whether or not env was reset
            
            ##get the dot product of the weight matrix and state representation
            ##results in 1 value for the state, and 1 value for each action
            result_values = state['w'].dot(current_state_rep)
            
            ##get the value of the current state
            current_state_value = result_values[0]

            ##Do the TD(0) update
            if not reset: #skip this if the env has just been reset
                ##calculate td error term
                td_error = reward + (self.gamma*current_state_value) - state['update_value']
        
                ##calculate a scaling factor
                ##this scaling factor allows us to switch from updating
                ##a look-up table to updating weights 
                ##Note: add description of scaling factor somewhere
                scale = np.sum(update_state_rep**2)
                if scale != 0:
                    scale = 1.0 / scale
                    
                ##update the state value
                state['w'][0] += self.alpha*td_error*update_state_rep*scale
                ##update the action values
                ##multiply the entire state represention by (action value * beta * tderror)
                ##scale these values and then update the weight matrix/look-up table
                dw = np.outer(update_action*self.beta*td_error, update_state_rep)
                state['w'][1:] += dw*scale
                
            ##calculate the updated value for update state and add it to the result_values array
            result_values[0] = state['w'].dot(state['update_state_rep'][:])[0]
            ##change the state to be updated to the current state in this step    
            state['update_state_rep'][:] = current_state_rep
            ##change the value being updated to the value of the current state in this step
            state['update_value'] = current_state_value

            ##return updated state value for update state and action values for current state
            return result_values 
        return step_TD0

## TD(n) Learning Rule

class ActorCriticTDn(nengo.processes.Process):
    '''Create nengo Node with input, output and state.
    This node is where the TD(n) learning rule is applied.
    
    Inputs: state, reward, chosen action, whether or not the environment was reset
    Outputs: updated state value, updated action values
    '''
    def __init__(self, n_actions, n, gamma = 0.95, alpha = 0.1, beta = 0.9, lambd=None):
        self.n_actions = n_actions ##number of possible actions for action-value space
        self.n = n ##value of n - the number of steps between the current state and the state being updated
        self.gamma = gamma ##discount factor for state values
        self.alpha = alpha ##learning rate
        self.beta = beta ##discount factor for action values
        
        ##Input = reward + state representation + action values for current state
        ##Output = action values for current state + state value
        super().__init__(default_size_in=n_actions + 2, default_size_out=n_actions + 1) 
        
    def make_state(self, shape_in, shape_out, dt, dtype=None):
        '''Get a dictionary of signals to represent the state of this process.
        This will include: the representation of the state being updated (update_state_rep),
        the initial value of the state (0) (update_value), and the weight matrix/look-up table (w).
        Weight matrix shape = [(n_actions + state) * size of state representation output]'''
        
        ##set dim = length of each row in matrix/look-up table 
        ##where a nengo ensemble is used to store the state representation, 
        ##this is equal to the number of neurons in the ensemble
        dim = shape_in[0]-2-self.n_actions
        ##return the state dictionary
        return dict(update_state_rep=np.zeros(dim),
                    update_value=0,
                    w=np.zeros((self.n_actions+1, dim)))
    
    def make_step(self, shape_in, shape_out, dt, rng, state):
        '''Create function that advances the process forward one time step.'''
        ##set dim = length of each row in matrix/look-up table 
        ##where a nengo ensemble is used to store the state representation, 
        ##this is equal to the number of neurons in the ensemble
        dim = shape_in[0]-2-self.n_actions
        
        state_memory = [] ##list for storing the last n states
        value_memory = [] ##list for storing the last n values
        reward_memory = [] ##list for storing the last n rewards
        action_memory = [] ##list for storing the last n chosen actions
        
        def step_TDn(t, x, state=state):
            ''' Function for performing TD(n) update at each timestep.
            Inputs: state representation, chosen action, reward
                and whether or not the env was reset
            Params:
                t = time
                x = inputs 
                state = the dictionary created in make_state.
            Outputs: updated state value, updated action values'''
            
            n = int(self.n)
                
            current_state_rep = x[:dim] ##get the state representation of current state
            update_state_rep = state['update_state_rep'] ##get representation of the state being updated
            last_action = x[dim:-2] ##get the action that was taken
            reward = x[-2] ##get reward received following action
            reset = x[-1] ##get whether or not env was reset
            
            ##get the dot product of the weight matrix and state representation
            ##results in 1 value for the state, and 1 value for each action
            result_values = state['w'].dot(current_state_rep[:])
            
            ##get the value of the current state
            current_state_value = result_values[0]
            
            global count

            ##Do the TD(n) update
            ##if the environemt has just been reset, set the count to 0 and empty the memory lists
            if reset:
                count = 0
                state_memory.clear()
                value_memory.clear()
                reward_memory.clear()
                action_memory.clear()
                
            elif not reset: ##skip this if the env has just been reset
                ##add the most recent reward and action to the memories
                reward_memory.append(reward)
                action_memory.append(last_action)
                
                ##Start updating after n steps
                if count>=n:
                    ##calculate td error term
                    Rs = self.gamma**np.arange(n)*reward_memory[:]
                    target = np.sum(Rs) + ((self.gamma**n)*current_state_value)
                    td_error = target - state['update_value']

                    ##calculate a scaling factor
                    ##this scaling factor allows us to switch from updating
                    ##a look-up table to updating weights 
                    scale = np.sum(update_state_rep**2)
                    if scale != 0:
                        scale = 1.0 / scale

                    ##update the state value
                    state['w'][0] += self.alpha*td_error*update_state_rep*scale

                    ##update the action values
                    ##multiply the entire state represention by (action value * beta * tderror)
                    ##scale these values and then update the weight matrix/look-up table
                    dw = np.outer(action_memory[0]*self.beta*td_error, update_state_rep)
                    state['w'][1:] += dw*scale
                    
                    ##delete the first value in each memory list 
                    state_memory.pop(0)
                    value_memory.pop(0)
                    reward_memory.pop(0)
                    action_memory.pop(0)
            
            ##increase count by 1 
            count+=1
            ##add the most recent state and value to the memories
            state_memory.append(current_state_rep.copy())
            value_memory.append(current_state_value.copy())
            
            ##calculate the updated value for update state and add it to the result_values array
            result_values[0] = state['w'].dot(state['update_state_rep'][:])[0]
            ##change the state to be updated to the new first value in the state memory   
            state['update_state_rep'][:] = state_memory[0][:]
            ##change the value being updated to the new first value in the value memory   
            state['update_value'] = value_memory[0]
        
            ##return updated state value for update state and action values for current state
            return result_values 
        return step_TDn


## TD($\lambda$) Learning Rule

class ActorCriticTDLambda(nengo.processes.Process):
    '''Create nengo Node with input, output and state.
    This node is where the TD(lambda) learning rule is applied.
    
    Inputs: state, reward, chosen action, whether or not the environment was reset
    Outputs: updated state value, updated action values
    '''
    def __init__(self, n_actions, n=None, alpha=0.1, beta=0.85, gamma=0.9, lambd=0.95):
        self.n_actions = n_actions ##number of possible actions for action-value space
        self.alpha = alpha ##learning rate
        self.beta = beta ##discount factor for action values
        self.gamma = gamma ##discount factor for state values
        self.lambd = lambd ##discount factor for eligibility traces
        
        super().__init__(default_size_in=n_actions + 2, default_size_out=n_actions + 1)
        
    def make_state(self, shape_in, shape_out, dt, dtype=None, y0=None):
        '''Get a dictionary of signals to represent the state of this process.
        This will include: the representation of the previous state (prev_state_rep),
        the eligibility traces for the state (trace) and the actions (action_trace),
        the initial value of the previous state (0) (prev_value), and the weight matrix/look-up table (w).
        Weight matrix shape = [(n_actions + state) * size of state representation output]'''
        
        ##set dim = length of each row in matrix/look-up table 
        ##where a nengo ensemble is used to store the state representation, 
        ##this is equal to the number of neurons in the ensemble
        dim = shape_in[0]-2-self.n_actions
        ##return the state dictionary
        return dict(prev_state_rep=np.zeros(dim),
                    trace=np.zeros(dim),
                    action_trace=np.zeros((self.n_actions, dim)),
                    prev_value=0,
                    w=np.zeros((self.n_actions+1, dim)))
    
    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        '''Function that advances the process forward one time step.'''
        ##set dim = length of each row in matrix/look-up table 
        ##where a nengo ensemble is used to store the state representation, 
        ##this is equal to the number of neurons in the ensemble
        
        dim = shape_in[0]-2-self.n_actions
        
        def step_TDlambda(t, x, state=state):
            ''' Function for performing TD(lambda) update at each timestep.
            Inputs: state representation, chosen action, reward
                and whether or not the env was reset
            Params:
                t = time
                x = inputs 
                state = the dictionary created in make_state.
            Outputs: updated state values, updated action values'''
            
            current_state_rep = x[:dim] ##get the state representation of current state
            prev_state_rep = state['prev_state_rep'] ##get representation of the previous state
            prev_action = x[dim:-2] ##get the action that was taken
            reward = x[-2] ##get reward received following action
            reset = x[-1] ##get whether or not env was reset
            
            ##get the dot product of the weight matrix and state representation
            ##results in 1 value for the state, and 1 value for each action
            result_values = state['w'].dot(current_state_rep)
            
            ##get the value of the current state
            current_state_value = result_values[0]
            
            ##Do the TD(lambda) update
            ##skip if the environment has just been reset
            if not reset:
                ##calculate the td error term
                td_error = reward + (self.gamma*current_state_value) - state['prev_value']
                
                ##calculate a scaling factor
                ##this scaling factor allows us to switch from updating
                ##a look-up table to updating weights 
                scale = np.sum(prev_state_rep**2)
                if scale != 0:
                    scale = 1.0 / scale
                
                ##update the state and action eligibility traces
                state['trace'] *= self.gamma * self.lambd
                state['action_trace'] *= self.gamma * self.lambd
                ##Accummulative trace - increment the state eligibility trace by the previous state representation
                state['trace'] += prev_state_rep * scale
                ##increment the action eligibility trace for the previous action by the previous state representation
                state['action_trace'] += np.outer(prev_action, prev_state_rep * scale)
                #state['action_trace'][:,prev_action] += prev_state_rep * scale
                
                ##update the weights for the state value
                state['w'][0] += self.alpha*td_error*state['trace']
                
                ##update the weights for the action values
                ##multiply the entire state represention by (action value * beta * tderror)
                ##scale these values and then update the weight matrix/look-up table
                dw = state['action_trace'] * self.beta * td_error
                state['w'][1:] += dw
                
                ##Uncomment these to compare TD(0) result to TD(lambda) where lambda = 0 (should be the same)
                #dw2 = np.outer(prev_action*self.beta*td_error, prev_state_rep)*scale
                #assert np.allclose(dw, dw2)

            ##change the 'prev_state_rep' to the representation of the current state in this trial
            state['prev_state_rep'][:] = current_state_rep
            ##change the 'prev_value' to the value of the current state in this trial
            state['prev_value'] = result_values[0]

            ##return updated state value for update state and action values for current state
            return result_values
        return step_TDlambda