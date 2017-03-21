#coding: utf-8

import numpy as np
import keras
import tensorflow as tf
from keras.models import Model, save_model
from keras.layers.merge import concatenate
from keras.layers import Input, Lambda
from keras.layers import Convolution1D as Conv1d
from keras.layers.core import Flatten, Dense, Reshape
from keras.layers import AtrousConv1D as Atrous1d
from keras import backend as K
from keras.models import load_model


import argparse
import gym
#from keras.utils.visualize_util import plot
from keras.utils.vis_utils import plot_model as plot


class Dynamics_Model(object):
    def __init__(self,action_dim, state_dim, z_dim, h_size, batch_size):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.H = h_size
        self.batch_size = batch_size
        self.vae, self.generator, self.vae_loss = self.build_network()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        self.vae.summary()
        self.generator.summary()
        plot(self.vae, to_file='vae.png')
        plot(self.generator, to_file='generator')


    def build_network(self):
        x_plus_ph = Input(shape=[self.H, self.state_dim], name="x_plus")
        x_m = Input(shape=[self.H, self.state_dim], name="x_min")
        u_plus = Input(shape=[self.H, self.action_dim], name="u_plus")
        u_m = Input(shape=[self.H, self.action_dim], name="u_min")

        #encoder

        h_1 = Conv1d(32, 2, activation='relu')(x_plus_ph)
        h_2 = Conv1d(16, 2, strides=2, activation='relu')(h_1)
        h_z = Flatten()(h_2)
        mu = Dense(self.z_dim)(h_z)
        sigma = Dense(self.z_dim, activation="softplus")(h_z)
        #sigma is sigma^2

        def sampling(t):
            z_mean, z_var = t
            z_std = K.sqrt(z_var)
            eps = K.random_normal(shape=(self.z_dim,), mean=0.0, stddev=1.0)
            return z_mean + eps * z_std

        z = Lambda(sampling, output_shape=(self.z_dim,))([mu, sigma])

        #decoder

        #layer instantiate

        x_layer_1 = Conv1d(32, 2, dilation_rate=1, name="Atrous_tanh_1")
        y_layer_1 = Conv1d(32, 2, dilation_rate=1, name="Atrous_sigmoid_1")

        z_dense_tan_1 = Dense(32, name="z_dense_tan_1")
        z_dense_sig_1 = Dense(32, name="z_dense_sig_1")

        lambda1 = Lambda(self.gated_activation, name='gate_1')

        x_layer_2 = Conv1d(32, 3, dilation_rate=2, name="Atrous_tanh_2")
        y_layer_2 = Conv1d(32, 3, dilation_rate=2, name="Atrous_sigmoid_2")

        z_dense_tan_2 = Dense(32, name="z_dense_tan_2")
        z_dense_sig_2 = Dense(32, name="z_dense_sig_2")

        lambda2 = Lambda(self.gated_activation, name='gate_2')

        x_layer_3 = Conv1d(32, 2, dilation_rate=4, name="Atrous_tanh_3")
        y_layer_3 = Conv1d(32, 2, dilation_rate=4, name="Atrous_sigmoid_3")

        z_dense_tan_3 = Dense(32, name="z_dense_tan_3")
        z_dense_sig_3 = Dense(32, name="z_dense_sig_3")

        lambda3 = Lambda(self.gated_activation, name='gate_3')

        last = Conv1d(110, 1, name='last_layer')

        in_px = concatenate([x_m, u_plus, u_m], axis=-1)

        xx1 = x_layer_1(in_px)
        yy1 = y_layer_1(in_px)

        z1 = z_dense_tan_1(z)
        z2 = z_dense_sig_1(z)

        px_h1 = lambda1([xx1, yy1, z1, z2])

        xx2 = x_layer_2(px_h1)
        yy2 = y_layer_2(px_h1)

        z1 = z_dense_tan_2(z)
        z2 = z_dense_sig_2(z)

        px_h2 = lambda2([xx2, yy2, z1, z2])

        xx3 = x_layer_3(px_h2)
        yy3 = y_layer_3(px_h2)

        z1 = z_dense_tan_3(z)
        z2 = z_dense_sig_3(z)

        atrous_out = lambda3([xx3, yy3, z1, z2])

        atrous_out = last(atrous_out)

        x_plus = Reshape((self.H, self.state_dim))(atrous_out)

        vae = Model([x_plus_ph, x_m, u_plus, u_m], x_plus)

        def vae_loss(x_original, x_generated):
            square_loss = K.mean((x_original - x_generated)**2)
            kl_loss = K.sum((-0.5*K.log(sigma)) + ((K.square(mu) + sigma) / 2.0) - 0.5)
            return square_loss + kl_loss

        def mean_squared(y_true, y_pred):
            #assert K.ndim(y_true) == 3
            #y_true = K.reshape(y_true, (K.shape(y_true)[0], K.shape(y_true)[1]*K.shape(y_true)[2]))
            #y_pred = K.reshape(y_true, (K.shape(y_pred)[0], K.shape(y_pred)[1]*K.shape(y_pred)[2]))
            return K.mean(K.square(y_pred - y_true), axis=-1)

        optimize = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

        vae.compile(optimizer="Adam", loss='mse')

        #generator
        sampled_z = Input(shape=(self.z_dim,))

        z1 = z_dense_tan_1(sampled_z)
        z2 = z_dense_sig_1(sampled_z)

        px_z1 = lambda1([xx1, yy1, z1, z2])

        xxz2 = x_layer_2(px_z1)
        yyz2 = y_layer_2(px_z1)

        z1 = z_dense_tan_2(sampled_z)
        z2 = z_dense_sig_2(sampled_z)

        px_z2 = lambda2([xxz2, yyz2, z1, z2])

        xxz3 = x_layer_3(px_z2)
        yyz3 = y_layer_3(px_z2)

        z1 = z_dense_tan_3(sampled_z)
        z2 = z_dense_sig_3(sampled_z)

        g_out = lambda3([xxz3, yyz3, z1, z2])

        g_out = last(g_out)

        g_out = Reshape((self.H, self.state_dim))(g_out)

        generator = Model([x_m, u_plus, u_m, sampled_z], g_out)

        return vae, generator, vae_loss

    def gated_activation(self, t):
        x1, y1, z, zz = t

        x = K.tanh(x1 + z[:, None, :])
        y = K.sigmoid(y1 + zz[:, None, :])

        return x*y

    def learn(self, actions, states, epoch):
        n_traj = len(actions)
        print n_traj

        action_zeros = np.zeros((self.H-1, self.action_dim))
        state_zeros = np.zeros((self.H-1, self.state_dim))
        train_xp = []
        train_xm = []
        train_up = []
        train_um = []

        for n in xrange(n_traj):
            action = np.concatenate([action_zeros, np.array(actions[n])], axis=0)
            state = np.concatenate([state_zeros, np.array(states[n])], axis=0)
            for i in xrange(len(action) - 2*self.H):
                x_p = state[i + self.H: i + 2 * self.H]
                x_m = state[i: i+self.H]
                u_p = action[i + self.H: i + 2 * self.H]
                u_m = action[i: i+self.H]
                train_xp.append(x_p)
                train_xm.append(x_m)
                train_up.append(u_p)
                train_um.append(u_m)

        xp = np.stack(train_xp)
        xm = np.stack(train_xm)
        up = np.stack(train_up)
        um = np.stack(train_um)

        print xp.shape
        print xp[3]
        print xm[4]

        #sys.exit()


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

        test_xp = np.expand_dims(xp[10], 0)
        test_xm = np.expand_dims(xm[10], 0)
        test_up = np.expand_dims(up[10], 0)
        test_um = np.expand_dims(um[10], 0)

        test_z = np.random.normal(loc=0.0, scale=1.0, size=(1, self.z_dim)).astype("float32")

        self.vae.fit([xp.astype("float32"), xm.astype("float32"), up.astype("float32"),
                      um.astype("float32")], xp.astype("float32"), epochs=epoch, validation_split=0.05)

        self.vae.compile(optimizer="Adam", loss=self.vae_loss)

        self.vae.fit([xp, xm, up, um], xp, epochs=epoch, validation_split=0.05)

        save_model(self.generator, './dynamics/generator_half.hdf5')

        generated_xp = self.generator.predict([test_xm, test_up, test_um, test_z])

        #error = np.sum((test_xp.reshape(test_xp.shape[0], test_xp.shape[1]*test_xp.shape[2]) - generated_xp)**2)
        error = np.sum((test_xp - generated_xp) ** 2)
        print error
        #print generated_xp
        #print test_xp









