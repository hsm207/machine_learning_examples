# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
from collections import Counter

# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt


def _compute_new_mean(
    previous_mean: float, newest_sample: float, total_samples: int
) -> float:
    """
  A method to compute mean in constant space and time

  total_samples includes number of samples in previous_mean and newest_sample
  """
    previous_samples = total_samples - 1
    return (previous_samples * previous_mean + newest_sample) / total_samples


class Bandit:
    def __init__(self, m):
        self.true_mean = m
        self.estimate_mean = 0
        self.N = 0

    def pull(self):
        return np.random.randn() + self.true_mean

    def update(self, x):
        self.N += 1
        self.estimate_mean = _compute_new_mean(self.estimate_mean, x, self.N)


def run_experiment(m1: float, m2: float, m3: float, eps: float, N: int):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]
    optimal_bandit = np.argmax([b.true_mean for b in bandits])

    rewards = np.empty(N)
    pulls = np.empty(N)

    for i in range(N):
        # epsilon greedy
        p = np.random.random()
        if p < eps:
            j = np.random.choice(3)
        else:

            j = np.argmax([b.estimate_mean for b in bandits])

        pulls[i] = j
        x = bandits[j].pull()
        bandits[j].update(x)

        # for the plot
        rewards[i] = x
    cumulative_average = np.cumsum(rewards) / (np.arange(N) + 1)

    # plot moving average ctr
    plt.plot(cumulative_average)
    plt.plot(np.ones(N) * m1)
    plt.plot(np.ones(N) * m2)
    plt.plot(np.ones(N) * m3)
    plt.xscale("log")
    plt.show()

    print(
        f"Optimal bandit is {optimal_bandit + 1} with mean reward {bandits[optimal_bandit].true_mean}"
    )
    for i, b in enumerate(bandits, 1):
        print(f"Estimated reward for bandit {i}: {b.estimate_mean:.4f}")

    print(f"Fraction not optimal pull: {1 - np.mean(pulls == optimal_bandit):.2f}")

    print(f"Distribution of bandit pulls")
    for bandit, pulls in sorted(Counter(pulls).items()):
        print(f"  {int(bandit + 1)}: {pulls:,d}")

    return cumulative_average


if __name__ == "__main__":
    c_1 = run_experiment(1.0, 2.0, 3.0, 0.1, 100000)
    c_05 = run_experiment(1.0, 2.0, 3.0, 0.05, 100000)
    c_01 = run_experiment(1.0, 2.0, 3.0, 0.01, 100000)

    # log scale plot
    plt.plot(c_1, label="eps = 0.1")
    plt.plot(c_05, label="eps = 0.05")
    plt.plot(c_01, label="eps = 0.01")
    plt.legend()
    plt.xscale("log")
    plt.show()
    # linear plot
    plt.plot(c_1, label="eps = 0.1")
    plt.plot(c_05, label="eps = 0.05")
    plt.plot(c_01, label="eps = 0.01")
    plt.legend()
    plt.show()

