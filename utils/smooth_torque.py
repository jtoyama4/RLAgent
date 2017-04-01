#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d as gf


def smooth_action(seq_len, action_bound, first_zero):
    action_bound = np.array(action_bound)
    f0 = np.random.uniform(0.1, 0.3, size=2)
    f1 = np.random.uniform(0.1, 0.3, size=2)
    f2 = np.random.uniform(0.1, 0.3, size=2)
    f3 = np.random.uniform(0.1, 0.3, size=2)

    C = np.array([1.0, 1.0])

    c0 = C / f0
    c1 = C / f1
    c2 = C / f2
    c3 = C / f3

    a0 = np.random.uniform(-c0, c0)
    a1 = np.random.uniform(-c1, c1)
    a2 = np.random.uniform(-c2, c2)
    a3 = np.random.uniform(-c3, c3)

    max_constraint = abs(a0) + abs(a1) + abs(a2) + abs(a3)

    bounds = C / action_bound

    a0 /= max_constraint * bounds
    a1 /= max_constraint * bounds
    a2 /= max_constraint * bounds
    a3 /= max_constraint * bounds

    #o0 = np.random.uniform(0.0, 6.14)
    #o1 = np.random.uniform(0.0, 6.14)
    #o2 = np.random.uniform(0.0, 6.14)
    #o3 = np.random.uniform(0.0, 6.14)

    def function(x):
        return a0*np.sin(f0*x) + a1*np.sin(f1*x) + a2*np.sin(f2*x) + a3*np.sin(f3*x)

    seq = np.arange(seq_len-first_zero)
    zeros = np.zeros((first_zero, 2))
    action = np.concatenate([zeros, map(function, seq)])
    seq = np.arange(seq_len)

    #plt.plot(seq, action[:, 0])
    #plt.show()

    return action

def gaussian_action(seq_len, action_bounds, first_zero):
    zeros = np.zeros((first_zero, 2))
    actions = np.random.normal(0.0, 1.0, size=(seq_len-first_zero, 2))
    b = gf(np.array(actions), 10.0, axis=0)
    bias = -b[0]
    b += bias
    ret = np.concatenate((zeros, b), axis=0)
    seq = np.arange(100)
    #plt.plot(seq, ret)
    #plt.show()
    return ret

if __name__ == "__main__":
    a = gaussian_action(100, [0.5, 0.1], 10)

