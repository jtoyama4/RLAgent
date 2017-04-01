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
import sys
import os
import argparse
sys.path.append("./dynamics")
sys.path.append("./utils")


from generator import Generator
import math
from smooth_torque import smooth_action, gaussian_action

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
    action = prev_action + np.random.normal(size=(action_dim,)) * 0.1
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
        smooth_actions = gaussian_action(100, [0.3, 0.2], 10)
        while not terminal:
            #env.render()
            action = smooth_actions[t]
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
            #s = np.sqrt(s)
            tmp = -0.5 * math.log(2*math.pi) - 0.5 * math.log(s) - (x_elem-m)**2 / (2.0*s)
            log_like += tmp
            #print "%d:%f" % (c,tmp)
            c += 1
    #like = (1.0 / K.sqrt(2.0 * math.pi * sigma)) * K.exp(-0.5 * (x-mu)**2 / sigma)
    return log_like / c



def test(instance):
    um = np.load("/tmp/test_um.npy")
    up = np.load("/tmp/test_up.npy")
    xm = np.load("/tmp/test_xm.npy")
    xp = np.load("/tmp/test_xp.npy")
    z = np.random.normal(size=(1, 8)).astype("float32")
    pred_state = instance.predict(xm, up, um, z)
    print pred_state
    print xp
    sys.exit()
    


def predict_trajectory(actions, states, steps):
    dynamics = Generator(ACTION_DIM, STATE_DIM, z_dim, H, "/tmp/vae_dynamics_3.model")
    log_like = 0.0
    count = 0

    #test(dynamics)
    #pred_all = True

    for state, action in zip(states, actions):
        print "new trajectory"
        if steps:
            for i in range(len(action)-2*H):
                i = i * H
                u_m = action[i: i+H]
                u_p = action[i+H: i+2*H]
                x_m = state[i: i+H]
                #print x_m[:3]

                u_m = np.expand_dims(u_m, 0)
                u_p = np.expand_dims(u_p, 0)
                x_m = np.expand_dims(x_m, 0)
                
                samples = []
                for _ in xrange(300):
                    z = np.random.normal(loc=0.0, scale=1.0, size=(1, z_dim))
                    pred_state = dynamics.predict(x_m, u_p, u_m, z)[0]
                    samples.append(pred_state)
                samples = np.array(samples)
                print samples.shape
                print "sample", samples[:,1,1]
                print np.var(samples[:,1,1])
                mean = np.mean(samples, axis=0)
                sigma = np.var(samples, axis=0)
                true = state[i + H:i + 2 * H]
                
                print "sigma", sigma[1,1]
                #print "mean", mean[0]
                #print "true", true[0]
            
                tmp = calculate_likelihood([mean, sigma, true])

                print tmp
 
                log_like += tmp

                count += 1
                if i + 2*H >= len(action):
                    break
        else:
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
                    pred_state = dynamics.predict(x_m, u_p, u_m, z)[0]
                    samples.append(pred_state)
                samples = np.array(samples)
                #print samples.shape
                #print "sample", samples[:,1,1]
                #print np.var(samples[:,1,1])
                mean = np.expand_dims(np.mean(samples, axis=0)[0], 0)
                sigma = np.expand_dims(np.var(samples, axis=0)[0], 0)
                true = np.expand_dims(state[i+H], 0)
                
                #print "mean", mean[0]
                #print "true", true[0]
                tmp = calculate_likelihood([mean, sigma, true])

                print tmp
                log_like += tmp
                count += 1

    return log_like / count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--steps', action='store_true', default=False)
    args = parser.parse_args()
    steps = args.steps
    actions, states = sampling_trajectory(3)
    log_likelihood = predict_trajectory(actions, states, steps)

