import numpy as py

# Future reward discount rate
discount = 0.9

# rewards for states
reward_wait = 0.5
reward_search = 1
reward_rescue = -3
reward_recharge = 0

# action sets for states high and low
actionH = [search, wait]
actionH_prop = .5

actionL = [search, wait, recharge]
actionL_prob = .333

def next_state(state, action):


    return n_state, reward
