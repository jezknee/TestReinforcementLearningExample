import random
#import gym
import numpy as np
from collections import deque
import keras
from keras.models import Sequential, load_model  # Added load_model import
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import os


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.input_shape = input_shape
        self.discrete = discrete # whether to represent as a vector or as one-hot encoding
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32 # if one hot we don not want to save as integers
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros((self.mem_size)) # keeping track of rewards agent receives
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32) # keeps track of terminal flags from the environment - once the episode is over, you don't want to take account of reward from next state

    def store_transition(self, state, action, reward, state_, done):
        # this stores every transition we do from the environment
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_  # Fixed spacing
        self.reward_memory[index] = reward 
        self.terminal_memory[index] = 1 - int(done)
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.mem_cntr += 1
        
    def sample_buffer(self, batch_size):
        # function to sample a subset of the memory
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
    
def build_dpqn(lr, n_actions, input_dims, fcl_dims, fc2_dims):
    # function to build deep-q network
    # we can use the Sequential object to construct a sequence of layers, and it takes a list of inputs
    # first is a dense layer
    model = Sequential([
                Dense(fcl_dims, input_shape=(input_dims, )), # input_dims allows us to pass in a batch or a single memory, important for learning or choosing an action respectively
                Activation('relu'),
                Dense(fc2_dims), # we do not need to specify the input shape in keras, makes it much easier
                Activation('relu'),
                Dense(n_actions)])
    
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')  # Fixed: lr -> learning_rate

    return model
    # this will allow us to call model.fit and model.predict to allow training and choosing actions
    # this is very simple, even more so than PyTorch

class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=0.996, epsilon_end=0.01, mem_size=10000000, fname='dqn_model.keras'):
        # gamma is our discount factor, epsilon for explore / exploit
        # need to look up why you want epsilon to gradually decrease (maybe less exploring over time?)
        # epsilon doesn't become 0 in testing, as you always want some exploring
        # but when actually using it that cn become 0
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)

        # Load model if it exists, else create a new one
        if os.path.exists(self.model_file):
            print(f"Loading existing model from {self.model_file}...")
            self.q_eval = keras.models.load_model(self.model_file)
        else:
            print("No saved model found. Building new model...")
            self.q_eval = build_dpqn(alpha, n_actions, input_dims, 256, 256)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis,:]
        rand = np.random.random()
        # pass the state through the network
        # get all the value of all the actions for that particular state
        # select the action that has the maximum value
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state, verbose=0)  # Added verbose=0 to reduce output
            action = np.argmax(actions)
        
        return action
    
    def learn(self):
        # learns on every step - temporal difference
        # we have created our memory with zeroes - do we pick random numbers, or just start learning? Latter here, but we have to wait until we've filled up a batch before we start learning
        if self.memory.mem_cntr < self.batch_size:
            return  
            
        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)
        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)  # Fixed typo: was "avtion_indices"

        q_eval = self.q_eval.predict(state, verbose=0)  # Added verbose=0
        q_next = self.q_eval.predict(new_state, verbose=0)  # Added verbose=0

        q_target = q_eval.copy()
        # address all states in your array, but can't just use array slicing, because the shape will be different for each batch size
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # these \ are line breaks, by the way, just continues on next line
        q_target[batch_index, action_indices] = reward + self.gamma*np.max(q_next, axis=1)*done # best possible reward you could have received in the next state * done

        _ = self.q_eval.fit(state, q_target, verbose=0)
        # this passes batch of states through network, calculates, then compares to q_target (delta between where we are and where we want to be)

        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min 
        # and that is how we learn
        # sample your buffer (non-sequential memories, as correlations will slow down learning process)
        # back from one hot to integer
        # calc value of next states
        # update q_targets
        # then use q_target as target for loss function of q network

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = keras.models.load_model(self.model_file)