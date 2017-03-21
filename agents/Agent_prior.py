#coding: utf-8
import os
import math
import time
import random
import sys
import math

import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Model
from keras.layers import Input, Lambda, Reshape, merge
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.layers import Convolution1D as Conv1d
from keras.regularizers import l2, l1
from keras.constraints import maxnorm, unitnorm
from keras.models import save_model


class Temporal_dynamics_action_prior(object):
    def __init__(self, action_dim, state_dim, z_dim, h_size, batch_size):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.H = h_size
        self.batch_size = batch_size
        self.action_prior = self.build_network()

    def build_network(self):
        u_plus_ph = Input(shape=[self.H, self.action_dim], name="u_plus")
        u_m = Input(shape=[self.H, self.action_dim], name="u_min")

        #encoder

        h_1 = Conv1d(12, 2, activation='relu')(u_plus_ph)
        h_2 = Conv1d(12, 2, subsample_length=2, activation='relu')(h_1)
        h_z = Flatten()(h_2)
        mu = Dense(self.z_dim)(h_z)
        sigma = Dense(self.z_dim, activation="softplus")(h_z)

        def sampling(t):
            z_mean, z_std = t
            eps = K.random_normal(shape=(self.z_dim,), mean=0., stddev=1.0)
            return z_mean + eps * z_std

        z = Lambda(sampling, output_shape=(self.z_dim,))([mu, sigma])

        #layer instantiation
        tan_layer_1 = Conv1d(32, 2, dilation_rate=1, name="Atrous_tanh_1")
        sig_layer_1 = Conv1d(32, 2, dilation_rate=1, name="Atrous_sigmoid_1")
        tan_dense_1 = Dense(32, name="z_dense_tan_1")
        sig_dense_1 = Dense(32, name="z_dense_sigm_1")
        lambda1 = Lambda(self.gated_activation, name='gate_1')

        tan_layer_2 = Conv1d(32, 3, dilation_rate=2, name="Atrous_tanh_2")
        sig_layer_2 = Conv1d(32, 3, dilation_rate=2, name="Atrous_sigmoid_2")
        tan_dense_2 = Dense(32, name="z_dense_tan_2")
        sig_dense_2 = Dense(32, name="z_dense_sigm_2")
        lambda2 = Lambda(self.gated_activation, name='gate_2')

        tan_layer_3 = Conv1d(32, 2, dilation_rate=4, name="Atrous_tanh_3")
        sig_layer_3 = Conv1d(32, 2, dilation_rate=4, name="Atrous_sigmoid_3")
        tan_dense_3 = Dense(32, name="z_dense_tan_3")
        sig_dense_3 = Dense(32, name="z_dense_sigm_3")
        lambda3 = Lambda(self.gated_activation, name='gate_3')

        last_layer = Conv1d(110, 1, name='last_layer')
        #decoder

        tanh_elem = tan_layer_1(u_m)
        sigm_elem = sig_layer_1(u_m)
        tanh_z = tan_dense_1(z)
        sigm_z = sig_dense_1(z)
        px_h = lambda1([tanh_elem, sigm_elem, tanh_z, sigm_z])

        tanh_elem = tan_layer_2(px_h)
        sigm_elem = sig_layer_2(px_h)
        tanh_z = tan_dense_2(z)
        sigm_z = sig_dense_2(z)
        px_h = lambda2([tanh_elem, sigm_elem, tanh_z, sigm_z])

        tanh_elem = tan_layer_3(px_h)
        sigm_elem = sig_layer_3(px_h)
        tanh_z = tan_dense_3(z)
        sigm_z = sig_dense_3(z)
        px_h = lambda3([tanh_elem, sigm_elem, tanh_z, sigm_z])

        u_plus = last_layer(px_h)
        u_plus = Reshape((self.H, self.state_dim))(u_plus)

        def vae_loss(x_original, x_generated):
            square_loss = K.mean((x_original - x_generated)**2)
            kl_loss = K.sum(-K.log(sigma) + (K.square(mu) + K.square(sigma)) / 2 - 0.5)
            return square_loss + kl_loss

        vae = Model([u_plus_ph, u_m], u_plus)

        vae.compile(optimizer='rmsprop', loss=vae_loss)

        sampled_z = Input(shape=(self.z_dim,))

        tanh_elem = tan_layer_1(u_m)
        sigm_elem = sig_layer_1(u_m)
        tanh_z = tan_dense_1(sampled_z)
        sigm_z = sig_dense_1(sampled_z)
        px_h = lambda1([tanh_elem, sigm_elem, tanh_z, sigm_z])

        tanh_elem = tan_layer_2(px_h)
        sigm_elem = sig_layer_2(px_h)
        tanh_z = tan_dense_2(sampled_z)
        sigm_z = sig_dense_2(sampled_z)
        px_h = lambda2([tanh_elem, sigm_elem, tanh_z, sigm_z])

        tanh_elem = tan_layer_3(px_h)
        sigm_elem = sig_layer_3(px_h)
        tanh_z = tan_dense_3(sampled_z)
        sigm_z = sig_dense_3(sampled_z)
        px_h = lambda3([tanh_elem, sigm_elem, tanh_z, sigm_z])

        u_plus = last_layer(px_h)
        u_plus = Reshape((self.H, self.state_dim))(u_plus)

        generator = Model([u_m, sampled_z], u_plus)

        return vae

    def gated_activation(self, t):
        x1, y1, z, zz = t

        x = K.tanh(x1 + z[:, None, :])
        y = K.sigmoid(y1 + zz[:, None, :])

        return x * y

    def learn(self, actions, states, epoch):
        n_traj = len(actions)
        print n_traj

        action_zeros = np.zeros((self.H, self.action_dim))

        train_up = []
        train_um = []

        for n in xrange(n_traj):
            action = np.concatenate([action_zeros, np.array(actions[n])], axis=0)
            for i in xrange(len(action) - 2*self.H):
                u_p = action[i + self.H: i + 2 * self.H]
                u_m = action[i: i+self.H]
                train_up.append(u_p)
                train_um.append(u_m)

        up = np.stack(train_up)
        um = np.stack(train_um)


        """
        json_vae = self.vae.to_json()
        json_generator = self.generator.to_json()

        with open("vae_model.json", 'w') as f:
            f.write(json_vae)

        with open("generator_model.json", 'w') as f:
            f.write(json_generator)
        """

        #save_model(self.vae, 'vae.hdf5')
        #save_model(self.generator, './dynamics/generator.hdf5')

        test_up = np.expand_dims(up[10], 0)
        test_um = np.expand_dims(um[10], 0)

        test_z = np.random.normal(loc=0.0, scale=1.0, size=(self.H, self.z_dim))

        self.action_prior.fit([up, um], up, epochs=epoch, validation_split=0.05)

        save_model(self.action_prior, './dynamics/action_prior.hdf5')

        generated_xp = self.action_prior.predict([test_um, test_z])

        #error = np.sum((test_xp.reshape(test_xp.shape[0], test_xp.shape[1]*test_xp.shape[2]) - generated_xp)**2)
        error = np.sum((test_xp - generated_xp) ** 2)
        print error
        print generated_xp
        print test_xp