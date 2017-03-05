#coding: utf-8

import os
import math
import time
import random
import keras
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from collections import deque
from keras.models import Model
from keras.layers import Input, Lambda, Reshape, merge
from keras.layers.core import Dense, Activation, Flatten, Merge
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.regularizers import l2, l1
from keras.constraints import maxnorm, unitnorm
from keras.layers.convolutional import Convolution2D as Conv2d
from keras.utils.visualize_util import plot



class Agent(object):
    def __init__(self, BUFFER_SIZE, STATE_DIM, ACTION_DIM, BATCH_BOOL, BATCH_SIZE, TAU, GAMMA, LEARNING_RATE,
                 NOISE_SCALE, ITERATION, INITIAL_REPLAY_SIZE, ACTION_BOUND):
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.batch_normalization = BATCH_BOOL
        self.batch_size = BATCH_SIZE
        self.tau = TAU
        self.gamma = GAMMA
        self.learning_rate=LEARNING_RATE
        self.noise_scale = NOISE_SCALE
        self.iteration = ITERATION
        self.initial_replay_size=INITIAL_REPLAY_SIZE
        self.action_bound = ACTION_BOUND
        #self.W_regularizer = l2()
        self.W_regularizer = None
        self.W_constraint = maxnorm(10)
        self.replay_memory = deque()

        self.x, self.u, self.mu, self.v, self.q, self.p, self.q_network = self.build_network()
        self.tx, self.tu, self.tmu, self.tv, self.tq, self.tp, self.target_q_network = self.build_network(t_bool=True)

        self.target_q_network.set_weights(self.q_network.get_weights())
        self.sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()

        plot(self.target_q_network, to_file="model.png")
        
    def _L(self, x):
        """
                :param x: a tensor with shape (batch_size, n*(n+1)/2)
                :return: a tensor of lower-triangular with shape (batch_size, n, n)
        """
        if self.action_dim == 1:
            return tf.exp(x)
        else:
            n = self.action_dim
            x = tf.transpose(x, perm=(1, 0))
            target = tf.Variable(np.zeros((n * n, self.batch_size)), dtype=tf.float32)

            # update diagonal values
            diag_indics = tf.square(tf.range(n))
            #t1 = tf.stop_gradient(tf.scatter_update(target, diag_indics, x[:n, :]))
            t1 = tf.scatter_update(target, diag_indics, x[:n, :])

            # update lower values
            u, v = np.tril_indices(n, -1)
            lower_indics = tf.constant(u * n + v)
            #t2 = tf.stop_gradient(tf.scatter_update(target, lower_indics, x[n:, :]))
            t2 = tf.scatter_update(target, lower_indics, x[n:, :])

            # reshape lower matrix to lower-triangular matrix
            target = tf.add(t1, t2)
            target = tf.transpose(target, (1, 0))
            target = tf.reshape(target, (self.batch_size, n, n))
            return target

    def _P(self, x):
        if self.action_dim == 1:
            return x**2
        else:
            return tf.matmul(x, tf.transpose(x, perm=[0, 2, 1]))

    def _A(self, t):
        if self.action_dim == 1:
            mu, p, u = t
            return -0.5 * (u - mu)**2 * p
        else:
            mu, p, u = t
            #d = tf.expand_dims(u - m, -1)
            d = u - mu
            return -0.5 * tf.matmul(tf.matmul(tf.transpose(d, perm=[0, 2, 1]), p), d)

    def _Q(self,t):
        v, a = t
        return v + a

    def namer(self, name, target=False):
        if target:
            name = "%s_t" % name
        return name

    def build_network(self, t_bool=False):
        x = Input(shape=(4, self.state_dim, self.state_dim), name=self.namer('state', t_bool))
        u = Input(shape=(self.action_dim,), name=self.namer('action', t_bool))
        #x_true = Input(shape=(1, self.state_dim, self.state_dim), name=self.namer('state', t_bool))

        #x = merge([x,x_true], output_shape=(1,self.state_dim*2, self.state_dim))
        if self.batch_normalization:
            h = BatchNormalization()(x)
        else:
            h = x
        h = Conv2d(8, 8, 8, activation='relu', subsample=(4, 4), W_constraint=self.W_constraint, W_regularizer=self.W_regularizer)(h)
        h = BatchNormalization()(h)

        h_m = Conv2d(16, 4, 4, subsample=(2, 2), activation='relu', W_constraint=self.W_constraint,
                     W_regularizer=self.W_regularizer)(h)
        h_m = Flatten()(h_m)
        h_m = Dense(256, activation='relu')(h_m)

        h_v = Conv2d(16, 4, 4, subsample=(2, 2), activation='relu', W_constraint=self.W_constraint,
                     W_regularizer=self.W_regularizer)(h)
        h_v = Flatten()(h_v)
        h_v = Dense(256, activation='relu')(h_v)

        h_l = Conv2d(16, 4, 4, subsample=(2, 2), activation='relu', W_constraint=self.W_constraint,
                     W_regularizer=self.W_regularizer)(h)
        h_l = Flatten()(h_l)
        h_l = Dense(256, activation='relu')(h_l)
        
        mu = Dense(self.action_dim)(h_m)

        v = Dense(1, W_constraint=self.W_constraint, W_regularizer=self.W_regularizer)(h_v)

        l0 = Dense(self.action_dim * (self.action_dim + 1) / 2, name=self.namer('l0', t_bool),
                   W_constraint=self.W_constraint, W_regularizer=self.W_regularizer)(h_l)
        l = Lambda(self._L, output_shape=(self.action_dim, self.action_dim), name=self.namer('l', t_bool))(l0)
        p = Lambda(self._P, output_shape=(self.action_dim, self.action_dim), name=self.namer('p', t_bool))(l)

        a = merge([mu, p, u], mode=self._A, output_shape=(self.action_dim,), name=self.namer("a", t_bool))
        q = merge([v, a], mode=self._Q, output_shape=(self.action_dim,), name=self.namer("q", t_bool))

        model = Model(input=[x, u], output=q)
        model.summary()
        fm = K.function([K.learning_phase(), x], [mu])
        mu_model = lambda x: fm([0, x])
        fp = K.function([K.learning_phase(), x], [p])
        p_model = lambda x: fp([0, x])
        fv = K.function([K.learning_phase(), x], [v])
        v_model = lambda x: fv([0, x])

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return x, u, mu_model, v_model, q, p_model, model

    def update_t(self):
        q_weights = self.q_network.get_weights()
        t_q_weights = self.target_q_network.get_weights()
        for i in xrange(len(q_weights)):
            t_q_weights[i] = self.tau*q_weights[i] + (1-self.tau)*t_q_weights[i]
        self.target_q_network.set_weights(t_q_weights)

    def get_action(self, x):
        mu = self.mu(x)[0]
        p = self.p(x)[0]

        if self.action_dim == 1:
            std = np.minimum(self.noise_scale/p, 1.0)
            action = np.random.normal(mu, std, size=(1,))
        else:
            cov = np.minimum(np.linalg.inv(p) * self.noise_scale, 1.0)
            action = np.random.multivariate_normal(mu, cov)

        action = np.clip(action, self.action_bound[0], self.action_bound[1])

        return action

    def run(self, state, action, reward, terminal, next_state):
        self.replay_memory.append((state, action, reward, terminal, next_state))

        if len(self.replay_memory) >= self.initial_replay_size:
            for i in range(self.iteration):
                self.learn()
                self.update_t()

    def get_initial_state(self):
        pass

    def learn(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        terminal_batch = []
        next_state_batch = []

        minibatch = random.sample(self.replay_memory, self.batch_size)

        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            terminal_batch.append(data[3])
            next_state_batch.append(data[4])

        target_value_batch = self.tv(
            np.float32(np.array(next_state_batch))
            )[0]

        reward_batch = np.array(reward_batch).reshape([self.batch_size, 1])

        y_batch = np.float32(reward_batch + self.gamma * target_value_batch)
        
        loss = self.q_network.train_on_batch([
            np.float32(np.array(state_batch)),
            np.float32(np.array(action_batch))], y_batch
        )
