# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range

# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
from grid_world import standard_grid, ACTION_SPACE
import pandas as pd

SMALL_ENOUGH = 1e-3  # threshold for convergence


def print_values(V, g):
    for i in range(g.rows):
        print("-" * 24)
        for j in range(g.cols):
            v = V.get((i, j), 0)
            if v >= 0:
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="")  # -ve sign takes up an extra space
        print("")


def print_policy(P, g):
    for i in range(g.rows):
        print("-" * 24)
        for j in range(g.cols):
            a = P.get((i, j), " ")
            print("  %s  |" % a, end="")
        print("")


def print_world(g):
    """
  Given a grid, visualize it using ascii

  'X' marks unreachable state e.g. wall

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
                    print("     |", end="")
            else:
                print("  X  |", end="")
        print("")


def build_transition_probabilities_and_rewards(grid):
    ### define transition probabilities and grid ###
    # the key is (s, a, s'), the value is the probability
    # that is, transition_probs[(s, a, s')] = p(s' | s, a)
    # any key NOT present will considered to be impossible (i.e. probability 0)
    transition_probs = {}

    # to reduce the dimensionality of the dictionary, we'll use deterministic
    # rewards, r(s, a, s')
    # note: you could make it simpler by using r(s') since the reward doesn't
    # actually depend on (s, a)
    rewards = {}

    for i in range(grid.rows):
        for j in range(grid.cols):
            s = (i, j)
            if not grid.is_terminal(s):
                for a in ACTION_SPACE:
                    s2 = grid.get_next_state(s, a)
                    transition_probs[(s, a, s2)] = 1
                    if s2 in grid.rewards:
                        rewards[(s, a, s2)] = grid.rewards[s2]

    return transition_probs, rewards


def print_state_transitions(grid):
    transition_probabilities, rewards = build_transition_probabilities_and_rewards(grid)
    df = pd.DataFrame(columns=["s", "a", "s'", "r", "p(s', r | s, a)"])

    for (current_state, action, future_state), prob in transition_probabilities.items():
        reward_key = (current_state, action, future_state)
        reward = rewards.get(reward_key, 0)

        df = df.append(
            {
                "s": current_state,
                "a": action,
                "s'": future_state,
                "r": reward,
                "p(s', r | s, a)": prob,
            },
            ignore_index=True,
        )

    return df


def solve_gridworld(grid, policy, gamma=0.9):

    grid = standard_grid()
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
                        action_prob = 1 if policy.get(s) == a else 0

                        # reward is a function of (s, a, s'), 0 if not specified
                        r = rewards.get((s, a, s2), 0)
                        # update new_v using the bellman equation for state-value function
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
        it += 1

        if biggest_change < SMALL_ENOUGH:
            break
        print("\n")
    print("\n\n")


if __name__ == "__main__":
    pass

