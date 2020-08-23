# From the course: Bayesin Machine Learning in Python: A/B Testing
# https://deeplearningcourses.com/c/bayesian-machine-learning-in-python-ab-testing
# https://www.udemy.com/bayesian-machine-learning-in-python-ab-testing
from __future__ import print_function, division
from builtins import range

# Note: you may need to update your version of future
# sudo pip install -U future

from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import pandas as pd
from altair.vegalite.v4.api import Chart

alt.data_transformers.disable_max_rows()


NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


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
    """
    A bandit is an instance of a slot machine
    In this example, a multiarm bandit is a list of Bandits
  """

    def __init__(self, p):
        # p: the win rate
        self.p = p
        self.p_estimate = 0
        self.N = 0

    def pull(self):
        # draw a 1 with probability p
        return np.random.random() < self.p

    def update(self, x):
        self.N += 1
        self.p_estimate = _compute_new_mean(self.p_estimate, x, self.N)


def experiment() -> Tuple[List[float], float]:
    """
  Finds the optimal bandit (the one that has the highest reward) given a list of bandits
  """
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    print("true win rate for each bandit")
    for i, b in enumerate(bandits, 1):
        print(f"  bandit {i}: {b.p:.4f}")

    rewards = np.zeros(NUM_TRIALS)
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0
    optimal_j = np.argmax([b.p for b in bandits])
    print("optimal bandit:", optimal_j + 1)

    for i in range(NUM_TRIALS):

        # use epsilon-greedy to select the next bandit
        # pick a random bandit with probability EPS, otherwise
        # pick the bandit with the highest estimated win rate
        if np.random.random() < EPS:
            num_times_explored += 1
            j = np.random.randint(len(bandits))
        else:
            num_times_exploited += 1
            j = np.argmax([b.p_estimate for b in bandits])

        if j == optimal_j:
            num_optimal += 1

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        # update rewards log
        rewards[i] = x

        # update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

    # print mean estimates for each bandit
    for i, b in enumerate(bandits, 1):
        print(f"mean estimate for bandit {i}: {b.p_estimate:.4f}")

    # print total reward
    print(f"total reward earned: {rewards.sum():.0f}")
    print(f"overall win rate: {rewards.sum() / NUM_TRIALS:.4f}")
    print(f"num_times_explored: {num_times_explored:,d}")
    print(f"num_times_exploited: {num_times_exploited:,d}")
    print(f"num times selected optimal bandit: {num_optimal:,d}")

    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)

    return (win_rates.tolist(), np.max(BANDIT_PROBABILITIES))


def plot_experiment_results(
    cumulative_win_rates: List[float], max_theoratical_win_rate: float
) -> Chart:
    n = len(cumulative_win_rates)
    t = list(range(1, n + 1))
    x_cumul_win_rate = ["cumulative win rate"] * n
    x_theo_max_win_rate = ["theoratical max win rate"] * n
    y_theo_max_win_rate = [max_theoratical_win_rate for _ in range(n)]

    df = pd.DataFrame(
        {
            "t": t * 2,
            "win rate": cumulative_win_rates + y_theo_max_win_rate,
            "legend": x_cumul_win_rate + x_theo_max_win_rate,
        }
    )

    return alt.Chart(df).mark_line().encode(x="t", y="win rate", color="legend")


if __name__ == "__main__":
    cumulative_winrates, theoratical_max_winrate = experiment()
    plot_experiment_results(cumulative_winrates, theoratical_max_winrate)
