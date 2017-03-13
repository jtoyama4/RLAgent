# coding: utf-8
import gym
import numpy as np
import sys
import argparse


def play(gym_mode, target=None):
    BUFFER_SIZE = 100000
    GAMMA = 0.95
    TAU = 0.001
    LEARNING_RATE = 0.001
    NUM_EPISODES = 10000
    INITIAL_REPLAY_SIZE = 100
    BATCH_SIZE = 100
    NOISE_SCALE=0.1
    ITERATION = 1
    BATCH_BOOL = True
    MOTORS=[7,8,9,10]

    np.random.seed(1234)

    if gym_mode:
        from agents.Agent_naf import Agent
        env = gym.make("LunarLanderContinuous-v2")
        #env = gym.make("Pendulum-v0")
        try:
            ACTION_DIM = env.action_space.shape[0]
        except AttributeError:
            ACTION_DIM = 1
        STATE_DIM = env.observation_space.shape[0]
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

    agent = Agent(BUFFER_SIZE, STATE_DIM, ACTION_DIM, BATCH_BOOL, BATCH_SIZE, TAU, GAMMA, LEARNING_RATE, NOISE_SCALE,
                  ITERATION, INITIAL_REPLAY_SIZE, ACTION_BOUND)
    
    for n_ep in xrange(NUM_EPISODES):
        terminal = False
        state = env.reset()
        t = 0
        total_reward = 0
        
        while not terminal:
            env.render()
            action = agent.get_action(agent.state_shaping(state))
            next_state, reward, terminal, _ = env.step(action)
            agent.run(state, action, reward, terminal, next_state)
            total_reward += reward
            state = next_state
            t += 1
            if not gym_mode:
                if t == 100:
                    break
                if rospy.is_shutdown():
                    break

        print "Episode:%d Reward:%d" % (n_ep, total_reward)
        print >> log, "Episode:%d Reward:%d" % (n_ep, total_reward)
        if not gym_mode:
            if rospy.is_shutdown():
                break
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--gym', action='store_true', default=False)
    parser.add_argument('--target', default=None)
    args = parser.parse_args()
    gym_mode = args.gym
    log = open('log.txt', 'a')
    if args.target:
        target = np.load(args.target)
    if gym_mode:
        play(gym_mode)
    else:
        import rospy
        try:
            play(gym_mode, target)
        except rospy.ROSInterruptException:
            pass
