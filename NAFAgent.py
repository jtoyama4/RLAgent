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
from keras.regularizers import l2, l1
from keras.constraints import maxnorm, unitnorm
from keras.layers.convolutional import Convolution2D as Conv2d


class Agent(object):
    def __init__(self, STATE_DIM, ACTION_DIM, BATCH_BOOL, BATCH_SIZE, TAU, GAMMA, LEARNING_RATE, NOISE_SCALE, ITERATION, INITIAL_REPLAY_SIZE):
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
        self.W_regularizer = l2()
        self.W_constraint = unitnorm()
        self.replay_memory = deque()

        self.x, self.u, self.mu, self.v, self.q, self.p, self.a, self.q_network = self.build_network()
        self.tx, self.tu, self.tmu, self.tv, self.tq, self.tp, self.ta, self.target_q_network = self.build_network()

        self.sess = tf.InteractiveSession()

    def _L(self, x):
        """
                :param x: a tensor with shape (batch_size, n*(n+1)/2)
                :return: a tensor of lower-triangular with shape (batch_size, n, n)
        """
        n = self.action_dim
        x = tf.transpose(x, perm=(1, 0))
        target = tf.Variable(np.zeros((n * n, self.batch_size)), dtype=tf.float32)

        # update diagonal values
        diag_indics = tf.square(tf.range(n))
        t1 = tf.stop_gradient(tf.scatter_update(target, diag_indics, x[:n, :]))

        # update lower values
        u, v = np.tril_indices(n, -1)
        lower_indics = tf.constant(u * n + v)
        t2 = tf.stop_gradient(tf.scatter_update(target, lower_indics, x[n:, :]))

        # reshape lower matrix to lower-triangular matrix
        target = tf.add(t1, t2)
        target = tf.transpose(target, (1, 0))
        target = tf.reshape(target, (self.batch_size, n, n))
        return target

    def _P(self, x):
        return tf.matmul(x, tf.transpose(x, perm=[0, 2, 1]))

    def _A(self, t):
        m, p, u = t
        #d = tf.expand_dims(u - m, -1)
        d = u - m
        return -tf.matmul(tf.matmul(tf.transpose(d, perm=[0, 2, 1]), p), d)

    def _Q(self,t):
        v, a = t
        return v + a

    def build_network(self):
        x = Input(shape=self.state_dim, name='state')
        u = Input(shape=self.action_dim, name='action')
        if self.batch_normalization:
            h = BatchNormalization()(x)
        else:
            h = x
        h = Dense(200, activation='relu', W_constraint=self.W_constraint, W_regularizer=self.W_regularizer)(h)
        h = Dense(200, activation='relu', W_constraint=self.W_constraint, W_regularizer=self.W_regularizer)(h)
        mu = Dense(self.action_dim)(h)

        #h_v = Dense(200, activation='relu', W_constraint=self.W_constraint, W_regularizer=self.W_regularizer)(h)
        #h_v = Dense(200, activation='relu', W_constraint=self.W_constraint, W_regularizer=self.W_regularizer)(h_v)
        v = Dense(1,W_constraint=self.W_constraint, W_regularizer=self.W_regularizer)(h)

        #h_l0 = Dense(200, activation='relu', W_constraint=self.W_constraint, W_regularizer=self.W_regularizer)(h)
        #h_l0 = Dense(200, activation='relu', W_constraint=self.W_constraint, W_regularizer=self.W_regularizer)(h_l0)
        l0 = Dense(self.action_dim * (self.action_dim + 1) / 2, name='l0',
                   W_constraint=self.W_constraint, W_regularizer=self.W_regularizer)(h)

        l = Lambda(self._L, output_shape=(self.action_dim, self.action_dim), name='l')(l0)
        p = Lambda(self._P, output_shape=(self.action_dim, self.action_dim), name='p')(l)
        a = merge([mu, p, u], mode=self._A, output_shape=(None, self.action_dim,), name="a")
        q = merge([v, a], mode=self._Q, output_shape=(None, self.action_dim,), name="q")

        model = Model(input=[x, u], output=q)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return x, u, mu, v, q, p, a, model

    def update_t(self):
        q_weights = self.q_network.get_weights()
        t_q_weights = self.target_q_network()
        for i in xrange(len(q_weights)):
            t_q_weights[i] = self.tau*q_weights[i] + (1-self.tau)*t_q_weights[i]
        self.target_q_network.set_weights(t_q_weights)

    def get_action(self, states):
        mu = self.sess.run(self.mu, feed_dict={
            self.x: states
        })
        p = self.sess.run(self.p, feed_dict={
            self.x: states
        })[0]

        cov = np.minimum(np.linalg.inv(p) * self.noise_scale, 1.0)

        action = np.random.multivariate_normal(mu, cov)

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

        target_value_batch = self.sess.run(self.tv, feed_dict={
            self.tx: np.float32(np.array(next_state_batch))
        })

        y_batch = np.float32(np.array(reward_batch) + self.gamma * target_value_batch)

        loss = self.q_network.train_on_batch([
            np.float32(np.array(state_batch)),
            np.float32(np.array(action_batch))], y_batch
        )
