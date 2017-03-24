#coding: utf-8

import numpy as np
import keras
import tensorflow as tf
from keras.models import Model, save_model
from keras.layers.merge import Concatenate, concatenate
from keras.layers import Input, Lambda
from keras.layers import Convolution1D as Conv1d
from keras.layers.core import Flatten, Dense, Reshape
from keras.losses import mean_squared_error as mse

from keras.layers import AtrousConv1D as Atrous1d
from keras import backend as K
from keras.models import load_model


import argparse
import gym
#from keras.utils.visualize_util import plot
from keras.utils.vis_utils import plot_model as plot


class Dynamics_Model(object):
    def __init__(self, action_dim, state_dim, z_dim, h_size, batch_size, epoch1, epoch2):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.H = h_size
        self.epoch1 = epoch1
        self.epoch2 = epoch2
        self.batch_size = batch_size
        self.layer_init()
        self.x_m, self.x_p, self.u_m, self.u_p, self.mse_loss, self.vae_loss, self.generator, self.vae_mse, self.vae = self.build_network()

        config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                            allow_growth=True
                        )
            )
        self.sess = tf.InteractiveSession(config=config)
        tf.global_variables_initializer().run()
        self.variables_const()
        """
        self.vae.summary()
        self.generator.summary()
        plot(self.vae, to_file='vae.png')
        plot(self.generator, to_file='generator')
        """

    def layer_const(self, layer):
        self.layers.append(layer)
        return layer

    def variables_const(self):
        self.variables = {}
        for layer in self.layers:
            name = layer.name
            weights = layer.trainable_weights
            print weights
            for n, weight in enumerate(weights):
                weight_name = "%s_%d" % (name, n)
                self.variables[weight_name] = weight
                print weight_name, type(weight)

    def layer_init(self):
        self.layers = []
        self.x_layer_1 = self.layer_const(Conv1d(32, 2, dilation_rate=1, name="Atrous_tanh_1"))
        self.y_layer_1 = self.layer_const(Conv1d(32, 2, dilation_rate=1, name="Atrous_sigmoid_1"))

        self.z_dense_tan_1 = self.layer_const(Dense(32, name="z_dense_tan_1"))
        self.z_dense_sig_1 = self.layer_const(Dense(32, name="z_dense_sig_1"))

        self.lambda1 = Lambda(self.gated_activation, name='gate_1')

        self.x_layer_2 = self.layer_const(Conv1d(32, 3, dilation_rate=2, name="Atrous_tanh_2"))
        self.y_layer_2 = self.layer_const(Conv1d(32, 3, dilation_rate=2, name="Atrous_sigmoid_2"))

        self.z_dense_tan_2 = self.layer_const(Dense(32, name="z_dense_tan_2"))
        self.z_dense_sig_2 = self.layer_const(Dense(32, name="z_dense_sig_2"))

        self.lambda2 = Lambda(self.gated_activation, name='gate_2')

        self.x_layer_3 = self.layer_const(Conv1d(32, 2, dilation_rate=4, name="Atrous_tanh_3"))
        self.y_layer_3 = self.layer_const(Conv1d(32, 2, dilation_rate=4, name="Atrous_sigmoid_3"))

        self.z_dense_tan_3 = self.layer_const(Dense(32, name="z_dense_tan_3"))
        self.z_dense_sig_3 = self.layer_const(Dense(32, name="z_dense_sig_3"))

        self.lambda3 = Lambda(self.gated_activation, name='gate_3')

        self.last = self.layer_const(Conv1d(self.state_dim, 1, name='last_layer'))

    def build_network(self):
        x_plus_ph = Input(shape=[self.H, self.state_dim], name="x_plus")
        x_m = Input(shape=[self.H, self.state_dim], name="x_min")
        u_plus = Input(shape=[self.H, self.action_dim], name="u_plus")
        u_m = Input(shape=[self.H, self.action_dim], name="u_min")

        # encoder

        h_1 = Conv1d(32, 2, activation='relu')(x_plus_ph)
        h_2 = Conv1d(16, 2, strides=2, activation='relu')(h_1)
        h_z = Flatten()(h_2)
        mu = Dense(self.z_dim)(h_z)
        var = Dense(self.z_dim, activation="softplus")(h_z)

        def sampling(t):
            z_mean, z_var = t
            z_std = K.sqrt(z_var)
            eps = K.random_normal(shape=(self.z_dim,), mean=0.0, stddev=1.0)
            return z_mean + eps * z_std

        def slicing(t, ix):
            c_u, c_x = t
            begin_index = tf.constant([0, ix, 0])
            size_index = tf.constant([-1, self.H, -1])
            u = tf.slice(c_u, begin_index, size_index)
            x = tf.slice(c_x, begin_index, size_index)
            k = tf.concat([u, x], axis=-1)
            return k

        def vae_loss_f(x_original, x_generated):
            square_loss = K.mean((x_original - x_generated)**2)
            kl_loss = K.sum((-0.5*K.log(var)) + ((K.square(mu) + var) / 2.0) - 0.5)
            return square_loss + kl_loss

        z = Lambda(sampling, output_shape=(self.z_dim,), name="sampling_lambda")([mu, var])

        # decoder

        connected_u = concatenate([u_m, u_plus], axis=1)
        connected_x = x_m

        x_plus = []
        for idx in xrange(self.H):
            arg = {"ix": idx}
            in_px = Lambda(slicing, arguments=arg, name='slicing_lambda',
                           output_shape=(self.H*2, self.state_dim+self.action_dim))([connected_u, connected_x])

            atrous_out = self.dilated_causal_conv(in_px, z)
            connected_x = concatenate([connected_x, atrous_out], axis=1)

            x_plus.append(atrous_out)

        x_plus = concatenate(x_plus, axis=1)

        mse_loss = tf.reduce_mean(mse(x_plus_ph, x_plus))
        vae_loss = tf.reduce_mean(vae_loss_f(x_plus_ph, x_plus))

        tf.summary.scalar("mse_loss", mse_loss)

        vae_mse = tf.train.AdamOptimizer().minimize(mse_loss)
        vae = tf.train.AdamOptimizer().minimize(vae_loss)

        # generator
        sampled_z = Input(shape=(self.z_dim,))

        connected_u = concatenate([u_m, u_plus], axis=1)
        connected_x = x_m

        g_out = []
        for idx in xrange(self.H):
            arg = {"ix": idx}
            in_px = Lambda(slicing, arguments=arg, name="slicing_lambda")([connected_u, connected_x])

            atrous_out = self.dilated_causal_conv(in_px, sampled_z)
            connected_x = concatenate([connected_x, atrous_out], axis=1)

            g_out.append(atrous_out)

        g_out = concatenate(g_out, axis=1)

        generator = K.function([K.learning_phase(), x_m, u_plus, u_m, sampled_z], [g_out])

        return x_m, x_plus_ph, u_m, u_plus, mse_loss, vae_loss, generator, vae_mse, vae

    def dilated_causal_conv(self, in_px, z):
        xx1 = self.x_layer_1(in_px)
        yy1 = self.y_layer_1(in_px)

        z1 = self.z_dense_tan_1(z)
        z2 = self.z_dense_sig_1(z)

        px_h1 = self.lambda1([xx1, yy1, z1, z2])

        xx2 = self.x_layer_2(px_h1)
        yy2 = self.y_layer_2(px_h1)

        z1 = self.z_dense_tan_2(z)
        z2 = self.z_dense_sig_2(z)

        px_h2 = self.lambda2([xx2, yy2, z1, z2])

        xx3 = self.x_layer_3(px_h2)
        yy3 = self.y_layer_3(px_h2)

        z1 = self.z_dense_tan_3(z)
        z2 = self.z_dense_sig_3(z)

        atrous_out = self.lambda3([xx3, yy3, z1, z2])

        atrous_out = self.last(atrous_out)

        return atrous_out

    def gated_activation(self, t):
        x1, y1, z, zz = t

        x = K.tanh(x1 + z[:, None, :])
        y = K.sigmoid(y1 + zz[:, None, :])

        return x*y

    def learn(self, actions, states):
        n_traj = len(actions)
        print n_traj
        train_xp = []
        train_xm = []
        train_up = []
        train_um = []

        for n in xrange(n_traj):
            #action = np.concatenate([action_zeros, np.array(actions[n])], axis=0)
            #state = np.concatenate([state_zeros, np.array(states[n])], axis=0)
            action = np.array(actions[n])
            state = np.array(states[n])
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

        test_xp = np.expand_dims(xp[30], 0)
        test_xm = np.expand_dims(xm[30], 0)
        test_up = np.expand_dims(up[30], 0)
        test_um = np.expand_dims(um[30], 0)

        np.save("/tmp/test_xp.npy", test_xp)
        np.save("/tmp/test_xm.npy", test_xm)
        np.save("/tmp/test_up.npy", test_up)
        np.save("/tmp/test_um.npy", test_um)

        test_z = np.random.normal(loc=0.0, scale=1.0, size=(1, self.z_dim)).astype("float32")

        data_length = len(xm)

        loss = self.mse_loss.eval(feed_dict={self.x_p: xp.astype("float32"),
                                             self.x_m: xm.astype("float32"),
                                             self.u_p: up.astype("float32"),
                                             self.u_m: um.astype("float32")})

        generated_xp = self.generator([0, test_xm, test_up, test_um, test_z])
        error = np.sum((test_xp - generated_xp) ** 2)
        print error

        for i in range(self.epoch1):
            shuffle_index = np.arange(data_length)
            np.random.shuffle(shuffle_index)
            for n_batch in range(data_length / self.batch_size - 1):
                batch_index = shuffle_index[n_batch*self.batch_size:(n_batch+1)*self.batch_size]
                self.vae_mse.run(feed_dict={self.x_m: xm[batch_index].astype("float32"),
                                            self.x_p: xp[batch_index].astype("float32"),
                                            self.u_m: um[batch_index].astype("float32"),
                                            self.u_p: up[batch_index].astype("float32")})

            print self.mse_loss.eval(feed_dict={self.x_m: test_xm.astype("float32"),
                                                self.x_p: test_xp.astype("float32"),
                                                self.u_m: test_um.astype("float32"),
                                                self.u_p: test_up.astype("float32")})

        for i in range(self.epoch2):
            shuffle_index = np.arange(data_length)
            np.random.shuffle(shuffle_index)
            for n_batch in range(data_length / self.batch_size - 1):
                batch_index = shuffle_index[n_batch*self.batch_size:(n_batch+1)*self.batch_size]
                self.vae.run(feed_dict={self.x_m: xm[batch_index].astype("float32"),
                                        self.x_p: xp[batch_index].astype("float32"),
                                        self.u_m: um[batch_index].astype("float32"),
                                        self.u_p: up[batch_index].astype("float32")})

            print self.mse_loss.eval(feed_dict={self.x_m: test_xm.astype("float32"),
                                                self.x_p: test_xp.astype("float32"),
                                                self.u_m: test_um.astype("float32"),
                                                self.u_p: test_up.astype("float32")})

        print loss

        saver = tf.train.Saver(self.variables)
        saver.save(self.sess, "/tmp/vae_dynamics_small_init.model")

        generated_xp = self.generator([0, test_xm, test_up, test_um, test_z])
        error = np.sum((test_xp - generated_xp) ** 2)
        print error
        #print generated_xp
        #print test_xp

        self.sess.close()

        with tf.Session() as sess1:
            init = tf.initialize_all_variables()
            sess1.run(init)
            generated_xp = self.generator([0, test_xm, test_up, test_um, test_z])
            error = np.sum((test_xp - generated_xp) ** 2)
            print error

        with tf.Session() as sess1:
            init = tf.initialize_all_variables()
            sess1.run(init)
            saver.restore(sess1, "/tmp/vae_dynamics_small_init.model")
            generated_xp = self.generator([0, test_xm, test_up, test_um, test_z])
            error = np.sum((test_xp - generated_xp) ** 2)
            print error
            print generated_xp
            print test_xp

        """
        merged_summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log", self.sess.graph)

        summary_str = self.sess.run(merged_summary_op, feed_dict={self.x_m: xm[batch_index].astype("float32"),
                                                                  self.x_p: xp[batch_index].astype("float32"),
                                                                  self.u_m: um[batch_index].astype("float32"),
                                                                  self.u_p: up[batch_index].astype("float32")})
        writer.add_summary(summary_str, 1)
        """








