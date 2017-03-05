import gym
import numpy as np
import sys
import argparse
from agents.Agent_naf import Agent

def play(gym_mode):
    BUFFER_SIZE = 100000
    GAMMA = 0.95
    TAU = 0.001
    LEARNING_RATE = 0.001
    NUM_EPISODES = 400
    INITIAL_REPLAY_SIZE = 1000
    BATCH_SIZE = 100
    NOISE_SCALE=0.3
    ITERATION = 1
    BATCH_BOOL = True
    MOTORS=[6,7,8,9]

    np.random.seed(1234)

    if gym_mode:
        env = gym.make("MountainCarContinuous-v0")
        try:
            ACTION_DIM = env.action_space.shape[0]
        except AttributeError:
            ACTION_DIM = 1
        STATE_DIM = env.observation_space.shape[0]
        ACTION_BOUND = [-env.action_space.high, env.action_space.high]
    else:
        from envs.Arm_image import Arm
        ACTION_DIM = len(MOTORS)
        STATE_DIM = 84
        env = Arm(MOTORS)
        ACTION_BOUND = [-1.5, 1.5]

    agent = Agent(BUFFER_SIZE, STATE_DIM, ACTION_DIM, BATCH_BOOL, BATCH_SIZE, TAU, GAMMA, LEARNING_RATE, NOISE_SCALE,
                  ITERATION, INITIAL_REPLAY_SIZE, ACTION_BOUND)
    
    for _ in xrange(NUM_EPISODES):
        terminal = False
        state = env.reset()
        t = 0
        total_reward = 0
        print _
        while not terminal:
            env.render()
            action = agent.get_action(state.reshape(1, state.shape[0]))
            next_state, reward, terminal, _ = env.step(action)
            agent.run(state, action, reward, terminal, next_state)
            total_reward += reward
            state = next_state
            t += 1
            if not gym_mode:
                if rospy.is_shutdown():
                    break

        print total_reward
        if not gym_mode:
            if rospy.is_shutdown():
                break
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--gym', action='store_true', default=False)
    parser.add_argument('--target', default=None)
    args = parser.parse_args()
    gym_mode = args.gym
    target = np.load(args.target)
    if gym_mode:
        play(gym_mode)
    else:
        import rospy
        try:
            play(gym_mode)
        except rospy.ROSInterruptException:
            pass
