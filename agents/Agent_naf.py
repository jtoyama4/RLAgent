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

        self.x, self.u, self.mu, self.v, self.q, self.p, self.q_network, self.l = self.build_network()
        self.tx, self.tu, self.tmu, self.tv, self.tq, self.tp, self.target_q_network, self.tl = self.build_network(t_bool=True)

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
            L_flat = x
            nb_elems = (self.action_dim * self.action_dim + self.action_dim) / 2

            diag_indeces = [0]

            for row in range(1, self.action_dim):
                diag_indeces.append(diag_indeces[-1] + (row + 1))

            diag_mask = np.zeros(1 + nb_elems)
            diag_mask[np.array(diag_indeces) + 1] = 1
            diag_mask = K.variable(diag_mask)

            nb_rows = tf.shape(L_flat)[0]
            zeros = tf.expand_dims(tf.tile(K.zeros((1,)), [nb_rows]), 1)
            L_flat = tf.concat(values=[zeros, L_flat], axis=1)

            tril_mask = np.zeros((self.action_dim, self.action_dim), dtype="int32")
            tril_mask[np.tril_indices(self.action_dim)] = range(1, nb_elems + 1)
            init = [
                K.zeros((self.action_dim, self.action_dim)),
                ]

            def ff(a,x):
                x_ = K.exp(x)
                x_ *= diag_mask
                x_ += x * (1. - diag_mask)
                L_ = tf.gather(x_, tril_mask)
                return [L_]

            tmp = tf.scan(ff, L_flat, initializer=init)
            return tmp[0]
            
            """n = self.action_dim
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
            return target"""


    def _P(self, x):
        if self.action_dim == 1:
            return x**2
        else:
            return K.batch_dot(x, K.permute_dimensions(x, (0,2,1)))

    def _A(self, t):
        if self.action_dim == 1:
            mu, p, u = t
            return -0.5 * (u - mu)**2 * p
        else:
            mu, p, u = t
            d = K.expand_dims(u-mu, -1)
            #d = u-mu
            return -0.5 * K.batch_dot(K.batch_dot(K.permute_dimensions(d, (0,2,1)), p), d)
            #return -0.5 * K.batch_dot(K.batch_dot(K.transpose(d), p), d)

    def _Q(self,t):
        v, a = t
        return v + a

    def namer(self, name, target=False):
        if target:
            name = "%s_t" % name
        return name

    def build_network(self, t_bool=False):
        x = Input(shape=(self.state_dim,), name=self.namer('state', t_bool))
        u = Input(shape=(self.action_dim,), name=self.namer('action', t_bool))
        if self.batch_normalization:
            h = BatchNormalization()(x)
        else:
            h = x
        h_m = Dense(100, activation='relu', W_constraint=self.W_constraint, W_regularizer=self.W_regularizer)(h)
        h_m = BatchNormalization()(h_m)
        h_m = Dense(100, activation='relu', W_constraint=self.W_constraint, W_regularizer=self.W_regularizer)(h_m)

        h_v = Dense(100, activation='relu', W_constraint=self.W_constraint, W_regularizer=self.W_regularizer)(h)
        h_v = BatchNormalization()(h_v)
        h_v = Dense(100, activation='relu', W_constraint=self.W_constraint, W_regularizer=self.W_regularizer)(h_v)

        h_l = Dense(100, activation='relu', W_constraint=self.W_constraint, W_regularizer=self.W_regularizer)(h)
        h_l = BatchNormalization()(h_l)
        h_l = Dense(100, activation='relu', W_constraint=self.W_constraint, W_regularizer=self.W_regularizer)(h_l)
        
        mu = Dense(self.action_dim)(h_m)

        v = Dense(1, W_constraint=self.W_constraint, W_regularizer=self.W_regularizer)(h_v)

        l0 = Dense(self.action_dim * (self.action_dim + 1) / 2, name=self.namer('l0', t_bool),
                   W_constraint=self.W_constraint, W_regularizer=self.W_regularizer)(h_l)
        l = Lambda(self._L, output_shape=(self.action_dim, self.action_dim), name=self.namer('l', t_bool))(l0)
        p = Lambda(self._P, output_shape=(self.action_dim, self.action_dim), name=self.namer('p', t_bool))(l)

        a = merge([mu, p, u], mode=self._A, output_shape=(self.action_dim,), name=self.namer("a", t_bool))
        q = merge([v, a], mode=self._Q, output_shape=(self.action_dim,), name=self.namer("q", t_bool))

        model = Model(input=[x, u], output=q)
        #model.summary()
        fl = K.function([K.learning_phase(), x], [l])
        l_model = lambda x: fl([0, x])
        fm = K.function([K.learning_phase(), x], [mu])
        mu_model = lambda x: fm([0, x])
        fp = K.function([K.learning_phase(), x], [p])
        p_model = lambda x: fp([0, x])
        fv = K.function([K.learning_phase(), x], [v])
        v_model = lambda x: fv([0, x])

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return x, u, mu_model, v_model, q, p_model, model, l_model

    def update_t(self):
        q_weights = self.q_network.get_weights()
        t_q_weights = self.target_q_network.get_weights()
        for i in xrange(len(q_weights)):
            t_q_weights[i] = self.tau*q_weights[i] + (1-self.tau)*t_q_weights[i]
        self.target_q_network.set_weights(t_q_weights)

    def get_action(self, x):
        mu = self.mu(x)[0][0]
        p = self.p(x)[0][0]
        l = self.l(x)

        if self.action_dim == 1:
            std = np.minimum(self.noise_scale/p, 1.0)
            action = np.random.normal(mu, std, size=(1,))
        else:
            pp = np.linalg.inv(p)

            cov = np.minimum(np.linalg.inv(p) * self.noise_scale, 1)
            action = np.random.multivariate_normal(mu, cov)

        if any(np.isnan(action)):
            print "nan detected. place mu instead."
            action = mu
        action = np.clip(action, self.action_bound[0], self.action_bound[1])

        return action

    def state_shaping(self, state):
        return state.reshape(1, state.shape[0])

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

        reward_batch = np.array(reward_batch)

        y_batch = np.float32(reward_batch[:, None] + self.gamma * target_value_batch)
        y_batch = np.tile(y_batch, (1, self.action_dim))
        loss = self.q_network.train_on_batch([
            np.float32(np.array(state_batch)),
            np.float32(np.array(action_batch))], y_batch
        )
