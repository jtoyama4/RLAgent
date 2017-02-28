#coding: utf-8
import gym
import numpy as np
import sys
import argparse


from agents.Agent_ddpg import Agent

def play(gym_mode):
    BUFFER_SIZE = 500000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001
    LRA = 0.0001
    LRC = 0.001
    ACTION_DIM=1
    STATE_DIM=3
    NUM_EPISODES = 3000
    INITIAL_REPLAY_SIZE = 20000
    BATCH_SIZE = 32

    np.random.seed(1234)

    agent = Agent(ACTION_DIM, STATE_DIM, TAU, GAMMA, LRA, LRC, INITIAL_REPLAY_SIZE, BATCH_SIZE)
    
    if gym_mode:
        env = gym.make("Pendulum-v0")
    else:
        from Arm import Arm
        import rospy
        env = Arm()

    for _ in xrange(NUM_EPISODES):
        terminal = False
        state = env.reset()
        t = 0
        print _
        while not terminal:
            env.render()
            action = agent.get_action(state.reshape(1, state.shape[0]))
            next_state, reward, terminal, _ = env.step(action)
            agent.run(state, action, reward, terminal, next_state)
            t += 1
            
            if t == 300:
                break

        if not gym_mode:
            if rospy.is_shutdown():
                break

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--gym', action='store_true', default=False)
    args = parser.parse_args()
    gym_mode = args.gym
    try:
        play(gym_mode)
    except rospy.ROSInterruptException:
        pass
