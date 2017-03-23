#coding: utf-8

import numpy as np
import tensorflow as tf
from keras.models import Model
import gym
from keras.models import load_model
from keras.layers.merge import Concatenate, concatenate
from keras.layers import Input, Lambda
from keras.layers import Convolution1D as Conv1d
from keras.layers.core import Flatten, Dense, Reshape
from keras import backend as K
import math
import os

cur_dir = os.getcwd()

env = gym.make("ReacherBasic-v1")
try:
    ACTION_DIM = env.action_space.shape[0]
except AttributeError:
    ACTION_DIM = 1
STATE_DIM = env.observation_space.shape[0]
ACTION_BOUND = [-env.action_space.high, env.action_space.high]
H = 10
z_dim = 8


def get_action(prev_action, bounds, action_dim):
    action = prev_action + np.random.normal(size=(action_dim,))/3.0
    action = np.clip(action, bounds[0], bounds[1])
    return action


def sampling_trajectory(NUM_EPISODES):
    actions = []
    states = []
    for n_ep in xrange(NUM_EPISODES):
        print n_ep
        terminal = False
        state = env.reset()
        t = 0
        total_reward = 0
        prev_action = 0.0
        tmp_a = []
        tmp_s = []
        while not terminal:
            #env.render()
            action = get_action(prev_action, ACTION_BOUND, ACTION_DIM)
            next_state, reward, terminal, _ = env.step(action)

            tmp_s.append(state)
            tmp_a.append(action)

            total_reward += reward
            state = next_state
            t += 1
            terminal = False
            if t == 100:
                terminal=True
        actions.append(tmp_a)
        states.append(tmp_s)
    return actions, states


def gated_activation(t):
    x1, y1, z, zz = t

    x = K.tanh(x1 + z[:, None, :])
    y = K.sigmoid(y1 + zz[:, None, :])

    return x * y


def slicing(t, ix):
    c_u, c_x = t
    begin_index = tf.constant([0, ix, 0])
    size_index = tf.constant([-1, H, -1])
    u = tf.slice(c_u, begin_index, size_index)
    x = tf.slice(c_x, begin_index, size_index)
    k = tf.concat([u, x], axis=-1)
    return k

class Generator(object):
    def __init__(self,action_dim, state_dim, z_dim, h_size):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.H = h_size
        self.layer_init()
        self.generator = self.build_generator()

        config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                            allow_growth=True
                        )
            )
        self.sess = tf.InteractiveSession(config=config)
        tf.global_variables_initializer().run()

    def layer_const(self, layer):
        name = layer.name
        print layer.get_config()
        weight = layer.get_weights()
        print type(weight)
        self.variables[name] = layer
        return layer

    def layer_init(self):
        self.variables = {}
        self.x_layer_1 = self.layer_const(Conv1d(32, 2, dilation_rate=1, name="Atrous_tanh_1"))
        self.y_layer_1 = self.layer_const(Conv1d(32, 2, dilation_rate=1, name="Atrous_sigmoid_1"))

        self.z_dense_tan_1 = self.layer_const(Dense(32, name="z_dense_tan_1"))
        self.z_dense_sig_1 = self.layer_const(Dense(32, name="z_dense_sig_1"))

        self.lambda1 = self.layer_const(Lambda(self.gated_activation, name='gate_1'))

        self.x_layer_2 = self.layer_const(Conv1d(32, 3, dilation_rate=2, name="Atrous_tanh_2"))
        self.y_layer_2 = self.layer_const(Conv1d(32, 3, dilation_rate=2, name="Atrous_sigmoid_2"))

        self.z_dense_tan_2 = self.layer_const(Dense(32, name="z_dense_tan_2"))
        self.z_dense_sig_2 = self.layer_const(Dense(32, name="z_dense_sig_2"))

        self.lambda2 = self.layer_const(Lambda(self.gated_activation, name='gate_2'))

        self.x_layer_3 = self.layer_const(Conv1d(32, 2, dilation_rate=4, name="Atrous_tanh_3"))
        self.y_layer_3 = self.layer_const(Conv1d(32, 2, dilation_rate=4, name="Atrous_sigmoid_3"))

        self.z_dense_tan_3 = self.layer_const(Dense(32, name="z_dense_tan_3"))
        self.z_dense_sig_3 = self.layer_const(Dense(32, name="z_dense_sig_3"))

        self.lambda3 = self.layer_const(Lambda(self.gated_activation, name='gate_3'))

        self.last = self.layer_const(Conv1d(self.state_dim, 1, name='last_layer'))

    def build_generator(self):
        # generator
        x_m = Input(shape=[H, self.state_dim], name="x_min")
        u_plus = Input(shape=[H, self.action_dim], name="u_plus")
        u_m = Input(shape=[H, self.action_dim], name="u_min")
        sampled_z = Input(shape=(z_dim,))

        def slicing(t, ix):
            c_u, c_x = t
            begin_index = tf.constant([0, ix, 0])
            size_index = tf.constant([-1, self.H, -1])
            u = tf.slice(c_u, begin_index, size_index)
            x = tf.slice(c_x, begin_index, size_index)
            k = tf.concat([u, x], axis=-1)
            return k

        connected_u = concatenate([u_m, u_plus], axis=1)
        connected_x = x_m

        g_out = []
        for idx in xrange(H):
            arg = {"ix": idx}
            in_px = Lambda(slicing, arguments=arg, name="slicing_lambda")([connected_u, connected_x])

            atrous_out = self.dilated_causal_conv(in_px, sampled_z)
            connected_x = concatenate([connected_x, atrous_out], axis=1)

            g_out.append(atrous_out)

        g_out = concatenate(g_out, axis=1)

        generator = K.function([K.learning_phase(), x_m, u_plus, u_m, sampled_z], [g_out])

        return generator

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


def calculate_likelihood(t):
    mus, sigmas, xs = t
    #print mus
    #print xs
    log_like = 0
    
    for mu, sigma, x in zip(mus, sigmas, xs):
        c=0
        for m, s, x_elem in zip(mu, sigma, x):
            sigma = 0.01
            tmp = -0.5 * math.log(2*math.pi) - 0.5 * math.log(s) - (x_elem-m)**2 / (2.0*s)
            log_like += tmp
            #print "%d:%f" % (c,tmp)
            c += 1
    #like = (1.0 / K.sqrt(2.0 * math.pi * sigma)) * K.exp(-0.5 * (x-mu)**2 / sigma)
    return log_like


def predict_trajectory(actions, states):
    instance = Generator()
    dynamics = instance.generator
    saver = tf.train.Saver()
    saver.restore(instance.sess, "/tmp/vae_dynamics.model")

    log_like = 0.0
    count = 0

    for state, action in zip(states, actions):
        for i in range(len(action)-2*H):
            u_m = action[i: i+H]
            u_p = action[i+H: i+2*H]
            x_m = state[i: i+H]
            #print x_m[:3]

            u_m = np.expand_dims(u_m, 0)
            u_p = np.expand_dims(u_p, 0)
            x_m = np.expand_dims(x_m, 0)

            samples = []
            for _ in xrange(100):
                z = np.random.normal(loc=0.0, scale=1.0, size=(1, z_dim))
                pred_state = dynamics.predict([x_m, u_p, u_m, z])[0]
                samples.append(pred_state)
            #print "sample", samples[:3]
            mean = np.mean(samples, axis=0)
            sigma = np.var(samples, axis=0)
            true = state[i + H:i + 2 * H]

            print "sigma", sigma[3]
            print "mean", mean[3]
            print "true", true[3]
            
            tmp = calculate_likelihood([mean, sigma, true])

            print tmp

            log_like += tmp

            count += 1

    return log_like / count

if __name__ == '__main__':
    actions, states = sampling_trajectory(10)
    log_likelihood = predict_trajectory(actions, states)
    print log_likelihood



