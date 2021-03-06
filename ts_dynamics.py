#coding: utf-8

import numpy as np
import tensorflow as tf
from keras.models import Model

from keras.layers import Input, Lambda
from keras.layers import Convolution1D as Conv1d
from keras.layers.core import Flatten, Dense, Reshape
from keras.layers import AtrousConv1D as Atrous1d
from keras import backend as K
import argparse
import gym

from utils.smooth_torque import gaussian_action, smooth_action

def get_action(prev_action, bounds, action_dim):
    action = prev_action + np.random.normal(size=(action_dim,)) * 0.1
    action = np.clip(action, bounds[0], bounds[1])
    return action

def play(gym_mode, target=None, one_step=False, epochs=(30,75)):
    BUFFER_SIZE = 100000
    GAMMA = 0.97
    TAU = 0.001
    LEARNING_RATE = 0.001
    NUM_EPISODES = 1000

    INITIAL_REPLAY_SIZE = 100

    BATCH_SIZE = 32
    Z_DIM=8
    H_SIZE=10
    NOISE_SCALE = 0.5
    ITERATION = 1
    BATCH_BOOL = True
    MOTORS = [7, 8, 9, 10]
    EPOCH1 = epochs[0]
    EPOCH2 = epochs[1]


    if one_step:
        print "one_step"
        from dynamics.Temporal_Dynamics import Dynamics_Model
    else:
        print "sequence step"
        from dynamics.Temporal_Dynamics_3 import Dynamics_Model

    np.random.seed(1234)

    if gym_mode:
        env = gym.make("ReacherBasic-v1")
        #env = gym.make("Pendulum-v0")
        try:
            ACTION_DIM = env.action_space.shape[0]
        except AttributeError:
            ACTION_DIM = 1
        STATE_DIM = env.observation_space.shape[0]
        print STATE_DIM
        ACTION_BOUND = [-env.action_space.high, env.action_space.high]
        print "action bound ", ACTION_BOUND
        print env.action_space.shape
    else:
        from agents.Agent_image_naf import Agent
        from envs.Arm_image import Arm
        ACTION_DIM = len(MOTORS)
        STATE_DIM = 84
        env = Arm(MOTORS, target)
        ACTION_BOUND = [-2.0, 2.0]

    actions = []
    states = []

    dynamics = Dynamics_Model(ACTION_DIM, STATE_DIM, Z_DIM, H_SIZE, BATCH_SIZE, EPOCH1, EPOCH2)

    for n_ep in xrange(NUM_EPISODES):
        terminal = False
        state = env.reset()
        t = 0
        total_reward = 0
        prev_action = [0.0 for _ in xrange(ACTION_DIM)]

        tmp_a = []
        tmp_s = []
        smooth_actions = gaussian_action(100, [0.3, 0.2], 10)

        while not terminal:
            env.render()
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
            if not gym_mode:
                if t == 100:
                    break
                if rospy.is_shutdown():
                    break
        actions.append(tmp_a)
        states.append(tmp_s)

    dynamics.learn(actions, states)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--gym', action='store_true', default=False)
    parser.add_argument('--target', default=None)
    parser.add_argument('--one_step', action='store_true', default=False)
    parser.add_argument('--epochs', default=None)
    args = parser.parse_args()
    gym_mode = args.gym
    log = open('log_ts.txt', 'w')
    one_step = args.one_step
    epochs = args.epochs
    print one_step
    if args.target:
        target = np.load(args.target)
    if gym_mode:
        play(gym_mode, one_step=one_step)
    else:
        import rospy

        try:
            play(gym_mode, target, one_step)
        except rospy.ROSInterruptException:
            pass
