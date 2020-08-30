# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range

# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
from grid_world import windy_grid, ACTION_SPACE

SMALL_ENOUGH = 1e-3  # threshold for convergence


def print_values(V, g):
    for i in range(g.rows):
        print("-" * 32)
        for j in range(g.cols):
            v = V.get((i, j), 0)
            if v >= 0:
                print(" %.4f|" % v, end="")
            else:
                print("%.4f|" % v, end="")  # -ve sign takes up an extra space
        print("")


def print_policy(P, g):
    for i in range(g.rows):
        print("-" * 88)
        for j in range(g.cols):
            a = P.get((i, j), " ")
            # print("  %s  |" % a, end="")
            print(f" {str(a).rjust(20)}|", end="")
        print("")


def is_probabilistic_state(grid, state):

    for k, v in grid.probs.items():
        if state in k and len(v) > 1:
            return True

    return False


def print_world(g):
    """
  Given a grid, visualize it using ascii

  'X' marks unreachable state e.g. wall

  '?' marks a state with a probabilistic state transition

  Integer values denote terminal state(s) and the associated reward
  """
    for i in range(g.rows):
        print("-" * 24)
        for j in range(g.cols):
            state = (i, j)
            if state in g.all_states():
                if state in g.rewards:
                    print(f"   {str(g.rewards[state]).rjust(2)}|", end="")
                else:
                    if is_probabilistic_state(g, state):
                        print("  ?  |", end="")
                    else:
                        print("     |", end="")
            else:
                print("  X  |", end="")
        print("")


def build_transition_probabilities_and_rewards(grid):
    ### define transition probabilities and grid ###
    # the key is (s, a, s'), the value is the probability
    # that is, transition_probs[(s, a, s')] = p(s' | s, a)
    # any key NOT present will considered to be impossible (i.e. probability 0)
    # we can take this from the grid object and convert it to the format we want
    transition_probs = {}

    # to reduce the dimensionality of the dictionary, we'll use deterministic
    # rewards, r(s, a, s')
    # note: you could make it simpler by using r(s') since the reward doesn't
    # actually depend on (s, a)
    rewards = {}

    for (s, a), v in grid.probs.items():
        for s2, p in v.items():
            transition_probs[(s, a, s2)] = p
            rewards[(s, a, s2)] = grid.rewards.get(s2, 0)

    return transition_probs, rewards


def solve_gridworld(grid, policy, gamma=0.9):
    grid = windy_grid()
    transition_probs, rewards = build_transition_probabilities_and_rewards(grid)
    
    # initialize V(s) = 0
    V = {}
    for s in grid.all_states():
        V[s] = 0

    print("start:")
    print_values(V, grid)
    print("\n")
    # repeat until convergence
    it = 1
    while True:
        biggest_change = 0
        for s in grid.all_states():
            if not grid.is_terminal(s):
                old_v = V[s]
                new_v = 0  # we will accumulate the answer
                for a in ACTION_SPACE:
                    for s2 in grid.all_states():

                        # action probability is deterministic
                        action_prob = policy[s].get(a, 0)

                        # reward is a function of (s, a, s'), 0 if not specified
                        r = rewards.get((s, a, s2), 0)
                        new_v += (
                            action_prob
                            * transition_probs.get((s, a, s2), 0)
                            * (r + gamma * V[s2])
                        )

                # after done getting the new value, update the value table
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        print(f"iter: {it} biggest_change: {biggest_change:.4f}")
        print_values(V, grid)
        print("")
        it += 1

        if biggest_change < SMALL_ENOUGH:
            break
    # print("V:", V)
    print("\n\n")
    # sanity check
    # at state (1, 2), value is 0.5 * 0.9 * 1 + 0.5 * (-1) = -0.05


if __name__ == "__main__":
    pass

