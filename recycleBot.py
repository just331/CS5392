import numpy as np


state_size = 5

# Future reward discount rate
discount = 0.9

# rewards for states
reward_wait = 0
reward_search = 1
reward_rescue = -3
reward_recharge = 0

# define the states of being high and low as 1 and 0 respectively
states = np.array([0, 1])
# action sets for states high and low: rescue, wait, recharge ,search
actions = np.array([-1, 0, 1, 2])
action_prop = .25


def next_state(state, action):
    global n_state, reward
    if state == states[1]:  # When state is high
        if actions[1]:  # State = high and action = wait
            n_state = np.array(state) - np.array(action)
            reward = reward_wait
        elif actions[3]:  # state = high and action = search
            n_state = np.array(state) - np.array(action)
            reward = reward_search
    elif state == states[0]:  # When state is low
        if actions[0]:  # State = low and action = rescue
            n_state = np.array(state) - np.array(action)
            reward = reward_rescue
        elif actions[1]:  # State = low and action = wait
            n_state = np.array(state) - np.array(action)
            reward = reward_wait
        elif actions[2]:  # State = low and action = recharge
            n_state = np.array(state) - np.array(action)
            reward = reward_recharge
        elif actions[3]:  # State = low and action = search
            n_state = np.array(state) - np.array(action)
            reward = reward_search

    return n_state, reward


def main():
    v = np.zeros(state_size)  # create the recycling robot v* array
    num_iter = 1000  # number of times we wish to run our experiment
    for i in range(num_iter):  # run our experiment
        new_v = np.zeros_like(v)  # create new array for calculating v
        for j in range(state_size):
            for a in actions:
                x, reward = next_state(j, a)
                new_v[j] += action_prop * (reward + discount * v[x])  # compute v
        v = new_v  # set new v to our old v

    for i in range(state_size):  # Print our new v neatly
        print("v: ", v[i])
