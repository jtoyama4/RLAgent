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
from dynamics.generator import Generator
import math
import os
from dynamics.Temporal_Dynamics_2 import Dynamics_Model

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
    dynamics = Dynamics_Model(ACTION_DIM, STATE_DIM, z_dim, H, 1, 1, 1)
    saver = tf.train.Saver()
    saver.restore(dynamics.sess, "/tmp/vae_dynamics.model")

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
            for _ in xrange(2):
                z = np.random.normal(loc=0.0, scale=1.0, size=(1, z_dim))
                pred_state = dynamics.generator([0, x_m, u_p, u_m, z])[0][0]
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



