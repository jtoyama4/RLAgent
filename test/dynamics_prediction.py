#coding: utf-8

import numpy as np
from keras.models import Model
import gym
from keras.models import load_model
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
        prev_action = [0.0 for _ in xrange(ACTION_DIM)]
        tmp_a = []
        tmp_s = []
        while not terminal:
            #env.render()
            if t < 10:
                action = prev_action
            else:
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


def calculate_likelihood(t):
    mus, sigmas, xs = t
    #print mus
    #print xs
    log_like = 0
    
    for mu, sigma, x in zip(mus, sigmas, xs):
        c=0
        for m, s, x_elem in zip(mu, sigma, x):
            tmp = -0.5 * math.log(2*math.pi) - 0.5 * math.log(s) - (x_elem-m)**2 / (2.0*s)
            log_like += tmp
            #print "%d:%f" % (c,tmp)
            c += 1
    #like = (1.0 / K.sqrt(2.0 * math.pi * sigma)) * K.exp(-0.5 * (x-mu)**2 / sigma)
    return log_like


def predict_trajectory(actions, states):
    generative_model = load_model('%s/dynamics/generator_try_more.hdf5'% cur_dir,
                                  custom_objects={"gated_activation":gated_activation})

    #state_zeros = np.zeros((H-1, STATE_DIM))
    #action_zeros = np.zeros((H-1, ACTION_DIM))
    log_like = 0.0
    count = 0

    for state, action in zip(states, actions):
        #first_state = np.expand_dims(np.array(state)[0], axis=0)
        #first_state = np.expand_dims(np.array(state)[:10], axis=0)
        #now_state = state[:H]
        #state = np.concatenate((state_zeros, first_state), axis=0)
        #action = np.concatenate((action_zeros, np.array(action)), axis=0)
        #state = state_zeros
        #state = first_state.reshape((10,11))
        #action = np.array(action)
        one_log = 0.0
        for i in range(len(action)-2*H):
            u_m = action[i: i+H]
            u_p = action[i+H: i+2*H]
            x_m = state[i: i+H]
            #print x_m[:3]

            u_m = np.expand_dims(u_m, 0)
            u_p = np.expand_dims(u_p, 0)
            x_m = np.expand_dims(x_m, 0)

            samples = []
            for _ in xrange(30):
                z = np.random.normal(loc=0.0, scale=1.0, size=(1, z_dim))
                pred_state = generative_model.predict([x_m, u_p, u_m, z])[0]
                samples.append(pred_state)
            #print "sample", samples[:3]
            mean = np.mean(samples, axis=0)[0]
            sigma = np.var(samples, axis=0)[0]
            true = state[i+H]
            tmp = calculate_likelihood([np.expand_dims(mean,0), np.expand_dims(sigma,0), np.expand_dims(true,0)])

            print np.sum((mean-true)**2)
            
            #mean = np.mean(samples, axis=0)
            #sigma = np.var(samples, axis=0)
            #true = state[i+H:i+2*H]
            #tmp = calculate_likelihood([mean, sigma, true])
            
            #print sigma
            #print mean
            #print true
            print tmp

            one_log += tmp
            log_like += tmp

            count += 1

        print one_log
    return log_like / count

if __name__ == '__main__':
    actions, states = sampling_trajectory(10)
    log_likelihood = predict_trajectory(actions, states)
    print log_likelihood



