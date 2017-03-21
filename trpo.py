import gym
import numpy as np
import sys
import argparse
from agents.Agent_trpo import Agent


def play(gym_mode):
    BUFFER_SIZE = 100000
    GAMMA = 0.99
    TAU = 0.001
    LEARNING_RATE = 0.001
    NUM_EPISODES = 200
    INITIAL_REPLAY_SIZE = 1000
    BATCH_SIZE = 1000
    NOISE_SCALE = 0.3
    ITERATION = 1
    BATCH_BOOL = True
    MOTORS = [6, 7, 8, 9]
    NUM_PATH=50
    EXTRA = 50

    def estimate_q(reward):
        def discount_sum(items):
            gamma = GAMMA
            sum = 0.0
            for n, item in enumerate(items):
                sum += (gamma ** n) * item
            return sum

        # q = [discount_sum(reward[i:i + EXTRA]) for i in range(len(reward) - EXTRA)]
        q = [discount_sum(reward[i:]) for i in range(len(reward))]
        return q

    np.random.seed(1234)

    if gym_mode:
        env = gym.make("InvertedDoublePendulum-v1")
        try:
            ACTION_DIM = env.action_space.shape[0]
        except AttributeError:
            ACTION_DIM = 1
        STATE_DIM = env.observation_space.shape[0]
        ACTION_BOUND = [-env.action_space.high, env.action_space.high]
        print env.action_space.shape
        print "action_bound: ",ACTION_BOUND
    else:
        from envs.Arm_image import Arm
        ACTION_DIM = len(MOTORS)
        STATE_DIM = 84
        env = Arm(MOTORS)
        ACTION_BOUND = [-1.5, 1.5]

    agent = Agent(STATE_DIM, ACTION_DIM, BATCH_SIZE, EXTRA, ACTION_BOUND=ACTION_BOUND)

    for _ in xrange(NUM_EPISODES):
        t = 0
        total_reward = 0
        print _
        s_list = []
        a_list = []
        q_list = []
        for __ in xrange(NUM_PATH):
            s_tmp=[]
            a_tmp=[]
            r_tmp=[]
            terminal = False
            state = env.reset()
            #state = state.reshape(state.shape[0], 1)
            while not terminal:
                env.render()
                action = agent.get_action(state.reshape(1, state.shape[0]))
                next_state, reward, terminal, _ = env.step(action)
                s_tmp.append(state)
                a_tmp.append(action)
                r_tmp.append(reward)
                total_reward += reward
                state = next_state
                t += 1
                if t == 1000:
                    break
                if not gym_mode:
                    if rospy.is_shutdown():
                        break
            #s_list += s_tmp[:-EXTRA]
            #a_list += a_tmp[:-EXTRA]
            #q_list += estimate_q(r_tmp)

            s_list += s_tmp
            a_list += a_tmp
            q_list += estimate_q(r_tmp)
        agent.run(s_list, a_list, q_list)

        print total_reward/NUM_PATH
        if not gym_mode:
            if rospy.is_shutdown():
                break




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--gym', action='store_true', default=False)
    parser.add_argument('--target', default=None)
    args = parser.parse_args()
    gym_mode = args.gym
    if args.target:
        target = np.load(args.target)
    if gym_mode:
        play(gym_mode)
    else:
        import rospy

        try:
            play(gym_mode)
        except rospy.ROSInterruptException:
            pass
