# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
# https://books.google.ca/books?id=_ATpBwAAQBAJ&lpg=PA201&ots=rinZM8jQ6s&dq=hoeffding%20bound%20gives%20probability%20%22greater%20than%201%22&pg=PA201#v=onepage&q&f=false
from __future__ import print_function, division
from builtins import range

# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from typing import List

import altair as alt
from altair.vegalite.v4.api import Chart

alt.data_transformers.disable_max_rows()


class Bandit:
    def __init__(self, p):
        # p: the win rate
        self.p = p
        self.p_estimate = 0.0
        self.N = 0.0  # num samples collected so far

    def pull(self):
        # draw a 1 with probability p
        return np.random.random() < self.p

    def update(self, x):
        self.N += 1.0
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N


def ucb(mean, n, nj, h=2):
    return mean + np.sqrt(h * np.log(n) / nj)


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

    base = alt.Chart(df).mark_line()

    return alt.hconcat(
        base.encode(
            alt.X(
                field="t",
                type="quantitative",
                scale=alt.Scale(type="log"),
                axis=alt.Axis(tickCount=np.log10(n)),
            ),
            y="win rate",
            color="legend",
        ),
        base.encode(x="t", y="win rate", color="legend"),
    )


def run_experiment(
    num_trials=100000, eps=0.1, bandit_probabilities=[0.2, 0.5, 0.75], h=2
):
    bandits = [Bandit(p) for p in bandit_probabilities]
    rewards = np.empty(num_trials)
    total_plays = 0
    optimal_bandit = np.argmax([b.p for b in bandits])

    print(f"Given bandits with rewards:")
    for i, b in enumerate(bandits, 1):
        print(f"  bandit {i}: {b.p:.4f}")
    print(
        f"Optimal bandit is {optimal_bandit + 1} with reward {bandits[optimal_bandit].p:.4f}"
    )

    # initialization: play each bandit once
    for j in range(len(bandits)):
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)

    for i in range(num_trials):
        j = np.argmax([ucb(b.p_estimate, total_plays, b.N, h) for b in bandits])
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)

        # for the plot
        rewards[i] = x
    cumulative_average = np.cumsum(rewards) / (np.arange(num_trials) + 1)

    for i, b in enumerate(bandits, 1):
        print(f"estimated reward for bandit {i}: {b.p_estimate:.4f}")

    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / num_trials)
    print("num times selected each bandit:", [b.N for b in bandits])

    return plot_experiment_results(
        cumulative_average.tolist(), np.max(bandit_probabilities)
    )


if __name__ == "__main__":
    run_experiment()

