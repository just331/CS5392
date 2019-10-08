import numpy as np

# Grid World n x n matrix
world_size = 5

# future reward discount rate
discount = .9

# rewards for grid states
reward_A = 10
reward_B = 5
reward_norm = 0
reward_offgrid = -1

# actions - left, up, right, down
actions = [np.array([0, -1]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0])]
action_prob = .25

# special states
a_pos = [0, 1]
a_prime_pos = [4, 1]
b_pos = [0, 3]
b_prime_pos = [2, 3]


# Given a state and action, determine the next state and reward
def next_state(state, action):
    '''Python doctests to test next_state
    >>> next_state([0,1],np.array([0,-1]))
    ([4, 1], 10)
    >>> next_state([0,3],np.array([-1,0]))
    ([2, 3], 5)
    >>> next_state([0,0],np.array([-1,0]))
    ([0, 0], -1)
    >>> next_state([3,3],np.array([0,1]))
    ([3, 4], 0)
    '''
    if state == a_pos:  # check for state A
        n_state = a_prime_pos
        reward = reward_A
    elif state == b_pos:  # check for state B
        n_state = b_prime_pos
        reward = reward_B
    else:  # check for next state on or off the grid
        n_state = (np.array(state) + action).tolist()
        if n_state[0] < 0 or n_state[0] >= world_size or \
                n_state[1] < 0 or n_state[1] >= world_size:
            n_state = state
            reward = reward_offgrid
        else:
            reward = reward_norm
    return n_state, reward


def main():
    v = np.zeros((world_size, world_size))  # create the grid world v* matrix
    num_iter = 1000  # number of experimental iterations
    for i in range(num_iter):  # run experiments
        new_v = np.zeros_like(v)  # create a new matrix for calculating v
        for j in range(world_size):  # for each state in the grid world
            for k in range(world_size):
                for a in actions:  # for each action for a state
                    (x, y), reward = next_state([j, k], a)
                    new_v[j, k] += action_prob * (reward + discount * v[x, y])  # compute v
        v = new_v  # set the new matrix to v
    for i in range(world_size):  # print v* neatly
        for j in range(world_size):
            print("{:12.4f}".format(v[i, j]), end=' ')
        print('\n')


# use Python doctest to test the nextstate function
if __name__ == '__main__':
    from doctest import testmod

    testmod(name='nextstate', verbose=False)
