#!/usr/bin/env python3
"""
Assignment Submission for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Ambar Qadeer
"""


import tensorflow as tf
from tensorflow import keras

class policy_network(keras.Model):

    def __init__(self, n_states, n_actions, ):
        super().__init__()
        self.n_actions = n_actions
        self.ip = keras.layers.Flatten(input_shape = n_states)
        self.l1 = keras.layers.Dense(128, kernel_initializer = tf.keras.initializers.HeUniform(seed=None), activation="relu")
        self.l2 = keras.layers.Dense(128, kernel_initializer = tf.keras.initializers.HeUniform(seed=None), activation="relu")
        #self.l3 = keras.layers.Dense(24, kernel_initializer = tf.keras.initializers.HeUniform(seed=None), activation="relu")
        self.op = keras.layers.Dense(n_actions, activation="softmax")


    def call(self, state, ):
        fp = self.ip(state)
        fp = self.l1(fp)
        fp = self.l2(fp)
        #fp = self.l3(fp)
        policy = self.op(fp)
        return policy
