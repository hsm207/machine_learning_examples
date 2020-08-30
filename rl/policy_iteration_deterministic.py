# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
from grid_world import standard_grid, ACTION_SPACE
from iterative_policy_evaluation_deterministic import print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9


# copied from iterative_policy_evaluation
def get_transition_probs_and_rewards(grid):
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


def evaluate_deterministic_policy(grid, policy):
  transition_probs, rewards = get_transition_probs_and_rewards(grid)
  # initialize V(s) = 0
  V = {}
  for s in grid.all_states():
    V[s] = 0

  # repeat until convergence
  it = 0
  while True:
    biggest_change = 0
    for s in grid.all_states():
      if not grid.is_terminal(s):
        old_v = V[s]
        new_v = 0 # we will accumulate the answer
        for a in ACTION_SPACE:
          for s2 in grid.all_states():

            # action probability is deterministic
            action_prob = 1 if policy.get(s) == a else 0
            
            # reward is a function of (s, a, s'), 0 if not specified
            r = rewards.get((s, a, s2), 0)
            new_v += action_prob * transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])

        # after done getting the new value, update the value table
        V[s] = new_v
        biggest_change = max(biggest_change, np.abs(old_v - V[s]))
    it += 1

    if biggest_change < SMALL_ENOUGH:
      break
  return V

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

def solve_grid(grid, policy, gamma=0.9):
  transition_probs, rewards = get_transition_probs_and_rewards(grid)

  # initial policy
  print("Initial policy:")
  print_policy(policy, grid)

  # repeat until convergence - will break out when policy does not change
  i = 1
  while True:
    print(f"\nOn iteration {i} ...")
    # policy evaluation step - we already know how to do this!
    print("Calculating V(s) under current policy ...")
    V = evaluate_deterministic_policy(grid, policy)

    # policy improvement step
    print("Calculating Q(s, a) to find a better policy ...")
    is_policy_converged = True
    # we only consider the states we need to take action
    # i.e. ignore unreachable and terminal states
    for s, _ in grid.actions.items():
      old_a = policy[s] # the action to take under the current policy
      new_a = None
      best_value = float('-inf')

      
      # loop through all possible actions to find the best current action
      for a in ACTION_SPACE:
        v = 0
        for s2 in grid.all_states():
          # reward is a function of (s, a, s'), 0 if not specified
          r = rewards.get((s, a, s2), 0)
          v += transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])

        if v > best_value:
          # track the best action we have found at state s so far
          best_value = v
          new_a = a
          

      # new_a now represents the best action in this state
      policy[s] = new_a
      if new_a != old_a:
        print(f"Found an improvement!\nInstead of {old_a} at state {s}, do {new_a}")
        print("Current policy is:")
        print_policy(policy, grid)
        is_policy_converged = False

    if is_policy_converged:
      print("Policy converged!\n")
      break
    i += 1

  # once we're done, print the final policy and values
  print("V(s) under optimal policy:")
  print_values(V, grid)
  print("Optimal policy:")
  print_policy(policy, grid)


if __name__ == '__main__':
  pass