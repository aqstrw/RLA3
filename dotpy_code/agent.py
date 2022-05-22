#!/usr/bin/env python3
"""
Assignment Submission for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Ambar Qadeer
"""


import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_probability as tfp
from policy_net import policy_network


class reinforce_agent():

    def __init__(self, n_actions, n_states, lr=0.003, gamma=0.99, ):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.states = []
        self.actions = []
        self.rewards = []
        self.pi = policy_network(n_states=self.n_states, n_actions=n_actions)  # check
        #         self.pi = policy_network(n_actions = n_actions)
        self.pi.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr))  # check
        print(self.pi.summary())

    # get action from policy
    def get_act(self, state, ):
        '''takes a state and returns a tensor of action categorical probabilities'''
        state_t = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.pi(state_t)
        cat_probs = tfp.distributions.Categorical(probs=probs)
        action = cat_probs.sample()
        return action.numpy()[0]

    # record state, action, and rewards
    def remember(self, state, action, reward, ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    # learn from histories
    def learn(self):
        actions_t = tf.convert_to_tensor(self.actions, dtype=tf.float32)
        rewards_arr = np.array(self.rewards)
        returns = np.zeros_like(rewards_arr)

        # loop over all states
        for state_id in range(returns.shape[-1]):
            returns_ds = 0
            discount_factor = 1

            # loop over downstream states
            for state_id_ds in range(state_id, returns.shape[-1]):
                returns_ds += rewards_arr[state_id_ds] * discount_factor
                discount_factor *= self.gamma

            returns[state_id] = returns_ds

        # within scope
        with tf.GradientTape() as tape:
            loss = 0

            # calcule loss
            for state_id, (return_g, state_g) in enumerate(zip(returns, self.states)):
                state_g = tf.convert_to_tensor([state_g], dtype=tf.float32)
                probabilities = self.pi(state_g)
                action_probabilities = tfp.distributions.Categorical(probs=probabilities)
                log_probabilities = action_probabilities.log_prob(actions_t[state_id])
                loss += -return_g * tf.squeeze(log_probabilities)

        # get gradients
        grad = tape.gradient(loss, self.pi.trainable_variables)
        self.pi.optimizer.apply_gradients(zip(grad, self.pi.trainable_variables))

        # empty lists
        self.states = []
        self.actions = []
        self.rewards = []
