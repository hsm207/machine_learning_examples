# From the course: Bayesin Machine Learning in Python: A/B Testing
# https://deeplearningcourses.com/c/bayesian-machine-learning-in-python-ab-testing
# https://www.udemy.com/bayesian-machine-learning-in-python-ab-testing
from __future__ import print_function, division
from builtins import range

# Note: you may need to update your version of future
# sudo pip install -U future


import matplotlib.pyplot as plt
import numpy as np
from typing import List


class Bandit:
    def __init__(self, p):
        # p: the win rate
        self.p = p
        self.p_estimate = 5.0
        self.N = 1.0  # num samples collected so far

    def pull(self):
        # draw a 1 with probability p
        return np.random.random() < self.p

    def update(self, x):
        self.N += 1.0
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N


def experiment(
    num_trials: int = 10000,
    eps: float = 0.1,
    bandit_probabilities: List[float] = [0.2, 0.5, 0.75],
):
    bandits = [Bandit(p) for p in bandit_probabilities]
    optimal_bandit = np.argmax([b.p for b in bandits])

    print(f"Given bandits with rewards:")
    for i, b in enumerate(bandits, 1):
        print(f"  bandit {i}: {b.p:.2f}")
    print(
        f"Optimal bandit is {optimal_bandit + 1} with reward {bandits[optimal_bandit].p}"
    )

    rewards = np.zeros(num_trials)
    preferred_bandit = 0
    print(f"Starting with bandit {preferred_bandit + 1} ...")

    for i in range(num_trials):
        # use optimistic initial values to select the next bandit
        j = np.argmax([b.p_estimate for b in bandits])
        if j != preferred_bandit:
            preferred_bandit = j
            print(
                f"Switching to bandit {preferred_bandit + 1} at iteration {i + 1}. Estimated rewards: {[b.p_estimate for b in bandits]}"
            )

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        # update rewards log
        rewards[i] = x

        # update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

    # print mean estimates for each bandit
    for i, b in enumerate(bandits, 1):
        print(f"mean estimate of bandit {i}: {b.p_estimate:.4f}")

    # print total reward
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / num_trials)
    print("num times selected each bandit:", [b.N for b in bandits])

    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(num_trials) + 1)
    plt.ylim([0, 1])
    plt.plot(win_rates)
    plt.plot(np.ones(num_trials) * np.max(bandit_probabilities))
    plt.show()


if __name__ == "__main__":
    experiment()
