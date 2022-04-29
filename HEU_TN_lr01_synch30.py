#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gym
import tensorflow as tf
from collections import deque
import numpy as np
from random import sample, randint
from tensorflow import keras
# from Helper import argmax, softmax
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical



def softmax(x, temp):
    ''' Computes the softmax of vector x with temperature parameter 'temp' '''
    x = x / temp # scale by temperature
    z = x - max(x) # substract max to prevent overflow of softmax 
    return np.exp(z)/np.sum(np.exp(z)) # compute softmax

def argmax(x):
    ''' Own variant of np.argmax with random tie breaking '''
    try:
        return np.random.choice(np.where(x == np.max(x))[0])
    except:
        return np.argmax(x)




def get_model(ip_shape,lr,op_shape,summary = True):
    '''
    get_model(ip_shape,lr,op_shape,summary = True):
    creates and returns a model and prints it's summary based on summary flag
    '''
    
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=ip_shape))
    model.add(keras.layers.Dense(24, kernel_initializer = tf.keras.initializers.HeUniform(seed=None), activation="relu"))
    model.add(keras.layers.Dense(24, kernel_initializer = tf.keras.initializers.HeUniform(seed=None), activation="relu"))
    model.add(keras.layers.Dense(24, kernel_initializer = tf.keras.initializers.HeUniform(seed=None), activation="relu"))
    model.add(keras.layers.Dense(op_shape, activation="linear"))

    # compile model
    model.compile(loss="mean_squared_error",optimizer=keras.optimizers.Adam(learning_rate=lr),metrics=["accuracy"])
    if summary == True:
        print(model.summary())
    return(model)

class experience_deque:
    '''
    __init__(self, max_len):
    
    # methods:
    add_experience(self,s,a,r,s_next,done):
    get_batch(self, batch_size):
    '''
    def __init__(self, max_len):
        '''initialisation'''
        #initialise max buffer length
        self.deque_size = max_len
        
        # initialise buffer for live deque length
        self.live_ds = 0
        
        # initialise experience buffers
        self.s_experience = deque(maxlen = self.deque_size)
        self.s_next_experience = deque(maxlen = self.deque_size)
        self.a_experience = deque(maxlen = self.deque_size)
        self.r_experience = deque(maxlen = self.deque_size)
        self.d_experience = deque(maxlen = self.deque_size)
        
        
        
        
    def add_experience(self,s,a,r,s_next,done):
        '''
        add_experience(self,s,a,r,s_next,done)
        add an experience to the deques
        '''
        self.s_experience.append(s)
        self.s_next_experience.append(s_next)
        self.a_experience.append(a)
        self.r_experience.append(r)
        self.d_experience.append(done)
        
        #update live deque size
        self.live_ds = len(self.s_experience)
        
        

        
        
    def get_batch(self, batch_size):
        '''
        get_batch(self, batch_size):
        generate random samples from experiences
        returns them
        '''
        # warn that deque is not full
        if self.live_ds < self.deque_size:
            if self.live_ds%1000 == 0:
                print(self.live_ds)
                
#             print("deque is not full, current size is : ", self.live_ds)
#             if batch_size > self.live_ds:
#                 print("batch size bigger than live deque size (bs,lds): ", batch_size, self.live_ds)
#             else:
#                 print("sampling from incomplete deque (bs,lds): ", batch_size, self.live_ds)
        
        # get random indices
        ind = sample(range(self.live_ds), batch_size)
        
        # sample from all deques
        s_sampled = np.asarray(self.s_experience)[ind]
        s_next_sampled = np.asarray(self.s_next_experience)[ind]
        a_sampled = np.asarray(self.a_experience)[ind]
        r_sampled = np.asarray(self.r_experience)[ind]
        d_sampled = np.asarray(self.d_experience)[ind]
        
        return (s_sampled,s_next_sampled,a_sampled,r_sampled,d_sampled)
        

class DQNagent:
    '''
    __init__(self, n_states, n_actions, learning_rate, gamma, max_len,
    er = True, tn = True, conv = False, summary = True, verbose = 0):
    Sets up a model and provides handy methods to interact with it
    '''
    
    def __init__(self, env, buffer, live_model, gamma, target_model, er = True,
                 TN = True, summary = True, verbose = 2, ):
        '''Iniitialization function for class DQNagent, read the __docs__'''
        
        # used for 
        self.TN = TN
        self.buffer = buffer
        self.live_model = live_model
        if TN:
            self.target_model = target_model
        self.n_states = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.gamma = gamma
#         self.deque_size = max_len
        self.verbose = verbose


        
        
    def select_action(self, state, policy='egreedy', epsilon=None, temp=None):
        '''
        select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        selects action based on policy specified
        returns action
        '''
        state = state.reshape(1,4)
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
                
            # TO DO: Add own code
            exploit = np.random.choice([0,1],p = [epsilon,1-epsilon])
            if exploit:
#                 print(q[0])
                a = argmax((self.live_model.predict(state))[0])
#                 print("exploiting: ",a)
            else:
                a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
#                 print("exploring: ",a)
                
#         elif policy == 'softmax':
#             if temp is None:
#                 raise KeyError("Provide a temperature")
                
#             # TO DO: Add own code
#             a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
#             print("action selected :", a)
        return a
                

    '''
    Drawing inspiration from the snippet in the comment below, citation in the report
    
    def update(self,s,a,r,s_next,done):
        #perform a Q-learning update

        # TO DO: Add own code
        G = r + ( self.gamma * np.max(self.Q_sa[s_next,:]) )
        self.Q_sa[s,a] = self.Q_sa[s,a] + ( self.learning_rate * ( G - self.Q_sa[s,a] ) )
        pass
    
#                     def fit_batch(env, model, target_model, batch):
#                        observations, actions, rewards, next_observations, dones = batch
#                        # Predict the Q values of the next states. Passing ones as the action mask.
#                        next_q_values = predict(env, target_model, next_observations)
#                        # The Q values of terminal states is 0 by definition.
#                        next_q_values[dones] = 0.0
#                        # The Q values of each start state is the reward + gamma * the max next state Q value
#                        q_values = rewards + DISCOUNT_FACTOR_GAMMA * np.max(next_q_values, axis=1)
#                        one_hot_actions = np.array([one_hot_encode(env.action_space.n, action) for action in actions])
#                        history = model.fit(
#                            x=[observations, one_hot_actions],
#                            y=one_hot_actions * q_values[:, None],
#                            batch_size=BATCH_SIZE,
#                            verbose=0,
#                        )
#                        return history.history['loss'][0]

    '''
    
    def update_er_tn(self,batch_of_ss, batch_of_as, batch_of_sns, batch_of_rs,batch_of_ds, batch_size):
        '''
        update_er_tn(self,batch):
        perform a Q-learning update
        '''
        if not self.TN:
            print("wrong call")

        # get target Q values for the batch
        q_next_batch = self.target_model.predict(batch_of_sns)
        
        # calculate targets and assign done rewards
        G1 = self.live_model.predict(batch_of_ss)
        cat_boa = to_categorical(batch_of_as,num_classes = 2)
        cat_inv_boa = to_categorical(np.invert(batch_of_as),num_classes = 2)
        G1 = G1 * cat_inv_boa
        G2 = batch_of_rs + ( self.gamma * np.max(q_next_batch, axis = 1) )
        G2 = np.where(batch_of_ds == True, batch_of_rs, G2)
        G2 = G2.reshape(G2.shape[0],1)
        G2 = G2 * cat_boa
        G_batch = G1+G2

        
        # update live_network
        history = self.live_model.fit(batch_of_ss,G_batch, batch_size = batch_size, verbose = 0)
        
        return history
    
    def update_er(self,batch_of_ss, batch_of_as, batch_of_sns, batch_of_rs,batch_of_ds, batch_size):
        '''
        update_er(self,batch):
        perform a Q-learning update
        '''
        if self.TN:
            print("wrong call")
        # get target Q values for the batch
        q_next_batch = self.live_model.predict(batch_of_sns)
        
        # calculate targets and assign done rewards
        G1 = self.live_model.predict(batch_of_ss)
        cat_boa = to_categorical(batch_of_as,num_classes = 2)
        cat_inv_boa = to_categorical(np.invert(batch_of_as),num_classes = 2)
        G1 = G1 * cat_inv_boa
        G2 = batch_of_rs + ( self.gamma * np.max(q_next_batch, axis = 1) )
        G2 = np.where(batch_of_ds == True, batch_of_rs, G2)
        G2 = G2.reshape(G2.shape[0],1)
        G2 = G2 * cat_boa
        G_batch = G1+G2

        
        # update live_network
        history = self.live_model.fit(batch_of_ss,G_batch, batch_size = batch_size, verbose = 0)
        
        return history
    
    
    
    
    def synch_weights(self):
        '''
        synch_weights(self):
        synchronises target model weights
        '''
        self.target_model.set_weights(self.live_model.get_weights())
        print("weights synched")

def decay_eps(epsilon = 1, min_epsilon = 0.01, decay_rate = 0.95):
    '''
    decay_eps(max_epsilon = 1, min_epsilon = 0.01, decay_rate = 0.995):
    decay the epsilon value
    '''
    epsilon *= decay_rate
    return max(epsilon, min_epsilon)

# main loop for a single run, averaging loop is not included
def Qlearn(learning_rate, epsilon, buffer_size, n_eps, max_timesteps,min_batch_size, synch_weight_freq, decay_epsilon = True, TN = True):
    
    # initialise environment
    env = gym.make('CartPole-v1')
    
    # create buffers for kpi
    cum_reward_per_ep = [] # list of final rewards [n_eps]

    # initialise networks
    live_net = get_model(env.observation_space.shape,learning_rate,env.action_space.n)
    
    if TN:
        target_net = get_model(env.observation_space.shape,learning_rate,env.action_space.n)
    
    # initialise buffers
    buffer = experience_deque(max_len = buffer_size)
    
    # initialise agent
#         def __init__(self, env, buffer, live_model, target_model,  gamma, batch_size, er = True,
#                  tn = True, summary = True, verbose = 2):
    if TN:
        agent = DQNagent(env, buffer, live_net, gamma, target_model = target_net, TN = TN)
    else:
        agent = DQNagent(env, buffer, live_net, gamma, target_model = None, TN = TN)
    
    # count for target net update
    step_count = 0
    
    # loop over eps
    for ep_num in range(n_eps):
        print("staring ep: ", ep_num)
        s = env.reset()
        rewards = []
        cum_reward = 0
        ep_list = []
        
#         if ep_num == 24:
#                 print ("\n\n\n\n\n\n training interval changed to 100 steps \n\n\n\n\n\n")
#                 synch_weight_freq = 100
#         if ep_num == 50:
#                 print ("\n\n\n\n\n\n training interval changed to 1000 steps \n\n\n\n\n\n")
#                 synch_weight_freq = 1000
        
        # loop over timesteps
        for step in range(max_timesteps):
            
                
            step_count += 1
#             env.render()
            
            # select action
            a = agent.select_action(state = s,policy = 'egreedy', epsilon = epsilon)

            
            # play a step
            s_next,reward,done,_ = env.step(a)
            cum_reward += reward
            
            # save experience
            buffer.add_experience(s,a,reward,s_next,done)
            
            calc_batch_size = min(int(0.7*buffer.live_ds),500)
        
            if buffer.live_ds >= min_batch_size:
#                 print("buffer min reached")
#             if step = 500
                
                # get batch from memory
                s_exp_batch,s_next_exp_batch,a_exp_batch,r_exp_batch,d_exp_batch = buffer.get_batch(calc_batch_size)

                # run a DQN training loop
                if TN:
#                     print("TN training")
                    history = agent.update_er_tn(s_exp_batch,a_exp_batch,s_next_exp_batch,r_exp_batch,d_exp_batch,calc_batch_size)
#                     print("loss = ",history.history["loss"][0])
                else:
#                     print("training")
                    history = agent.update_er(s_exp_batch,a_exp_batch,s_next_exp_batch,r_exp_batch,d_exp_batch,calc_batch_size)
                    #                 print("loss = ",history.history["loss"][0])
                if TN:
                    # copy weights to target_network
                    if step_count%synch_weight_freq == 0:
                        agent.synch_weights()



            # check if done
            if done:
#                 print("done = ",done)
                print("ep = ", ep_num, "    epsilon = ", epsilon, "    steps = ", step,"    ep reward = ",cum_reward)
                break
                
                
        
            # set new state
            s = s_next
#             print(step)
            
            
            # timestep loop ends
                        
            
        # end of episode calculations
        
        # collect kpi
        cum_reward_per_ep.append(cum_reward)
        
        
        # Decay probability of taking random action
        if decay_epsilon:
            epsilon = decay_eps(epsilon,eps_min)
            
        
        # check if done
        if ep_num>benchmark_averaging_eps:
            avg_10ep_cum_rewards = np.mean(cum_reward_per_ep[-benchmark_averaging_eps:])
            print("avg 20 ep rewards : ",avg_10ep_cum_rewards)
            if avg_10ep_cum_rewards >= benchmark:
                print("\n\n\n Rewards have converged to a value above the benchmark score (",benchmark,") \n\n\n")
                break
                
#                 extend = input('another episode? (y/n)')
#                 if extend == 'y':
#                     continue
#                 else:
#                     break
                    
        ep_list.append(ep_num)
                
    return (cum_reward_per_ep, avg_10ep_cum_rewards, ep_list)
    



def safe(fname = "TN_heu_lr001_synch10.csv"):
    try:
        fig = plt.figure(figsize = (14,7))
        list_y = []
        for i,y in enumerate(cum_rewards):
            y = np.array(y)
            y = np.pad(y,(0,300-y.shape[0]),mode = 'constant',constant_values = (np.nan))
            list_y.append(y)
            np.savetxt(fname, list_y, delimiter = "," )
    #         print(y.shape)
            plt.plot(range(y.shape[0]),y,label = r"run "+str(i+1))
        plt.title("TN individual")
        plt.xlabel("Rewards")
        plt.ylabel("Episodes")
        list_y = np.array(list_y)
        print(list_y.shape)
        fig = plt.figure(figsize = (14,7))
        plt.plot(range(300), np.nanmean(list_y, axis = 0))
        plt.xlabel("Rewards")
        plt.ylabel("Episodes")
        plt.title("TN individual")
    except Exception as e:
        print(e)
    

n_eps = 150                  # max number of eps
buffer_size = 10000
synch_weight_freq = 30
learning_rate = 0.01
gamma = 0.995                 # discount factor
epsilon = 1
min_batch_size = 32              # size of the replay sample
# min_buff = 1000
max_timesteps = 500           # max steps an episode can last
eps_min = 0.01
num_runs = 5
benchmark_averaging_eps = 20  # how many eps to average over to compare benchmark
benchmark = 350 

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# print(physical_devices)

# buffers for kpis
cum_rewards = []
averages_at_end = []
list_of_ep_nums = []

# loop for averaging over 8 runs
for run in range(num_runs):
#     learning_rate, epsilon, buffer_size, n_eps, max_timesteps,min_batch_size
    cum_rewards_run, averages_at_end_run, list_of_ep_nums = Qlearn(learning_rate, epsilon, buffer_size,
                                                                   n_eps, max_timesteps, min_batch_size, synch_weight_freq,
                                                                   TN = True)
    cum_rewards.append(cum_rewards_run)
    averages_at_end.append(averages_at_end_run)
    list_of_ep_nums.append(run)
    safe()

# summarise results
print(np.shape(cum_rewards))
print(np.shape(averages_at_end))

np.savetxt("TN_heu_lr01_synch30.csv", list_y, delimiter = "," )

