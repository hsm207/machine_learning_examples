# From the course: Bayesin Machine Learning in Python: A/B Testing
# https://deeplearningcourses.com/c/bayesian-machine-learning-in-python-ab-testing
# https://www.udemy.com/bayesian-machine-learning-in-python-ab-testing
from __future__ import division, print_function

from builtins import range
from typing import List

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from altair.vegalite.v4.api import Chart
from scipy.stats import beta

# Note: you may need to update your version of future
# sudo pip install -U future


alt.data_transformers.disable_max_rows()


# np.random.seed(2)


class Bandit:
    def __init__(self, p, i: int):
        self.true_p = p
        self.a = 1
        self.b = 1
        self.N = 0  # for information only
        self.name = i

    def pull(self):
        return np.random.random() < self.true_p

    def sample(self):
        return np.random.beta(self.a, self.b)

    def update(self, x):
        # since we let the likelihood be a bernouli and the prior
        # be a beta distribution, the  posterior is going to be a
        # beta distribution and this is how we can update the
        # posterior's
        # more details here: https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions
        self.a += x
        self.b += 1 - x
        self.N += 1

    def confidence_interval(self, a: float = 0.05):
        lower_bound = a / 2
        upper_bound = 1 - a / 2
        n = int(1e6)

        samples = np.random.beta(self.a, self.b, size=n)
        l, u = np.quantile(samples, [lower_bound, upper_bound])
        return (l, u)


def experiment(
    bandit_probabilities: List[float] = [0.2, 0.5, 0.75], num_trials: int = 2000
):
    bandits = [Bandit(p, i) for i, p in enumerate(bandit_probabilities, 1)]

    optimal_bandit = np.argmax([b.true_p for b in bandits])

    print(f"Given bandits with rewards:")
    for i, b in enumerate(bandits, 1):
        print(f"  bandit {i}: {b.true_p:.4f}")
    print(
        f"Optimal bandit is {optimal_bandit + 1} with reward {bandits[optimal_bandit].true_p:.4f}"
    )

    rewards = np.zeros(num_trials)
    for i in range(num_trials):
        # Thompson sampling
        j = np.argmax([b.sample() for b in bandits])

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        # update rewards
        rewards[i] = x

        # update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

    # print total reward
    for i, b in enumerate(bandits, 1):
        l, u = b.confidence_interval(0.05)
        print(f"95% confidence inteval for bandit {i}: ({l:.2f}, {u:.2f})")

    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / num_trials)
    print("num times selected each bandit:", [b.N for b in bandits])

    bandit_states = [plot_bandit_state(b) for b in bandits]
    ((bandit_states[0] | bandit_states[1]) & bandit_states[2]).display()


def plot_bandit_state(bandit: Bandit) -> Chart:
    a = bandit.a
    b = bandit.b
    n = bandit.N
    true_p = bandit.true_p
    i = bandit.name

    x = np.linspace(0, 1, 200)
    y = beta.pdf(x, a, b)

    df = pd.DataFrame({"p": x, "density": y, "true_p": true_p})

    base = alt.Chart(df)

    # plot the distribution
    p1 = (
        base.mark_line()
        .encode(x="p", y="density",)
        .properties(title=f"Bandit {i} ({a-1}/{n})")
    )

    # mark the true win rate
    p2 = base.mark_rule(color="red").encode(x=alt.X("true_p", title="p"))

    return (p1 + p2).properties(height=240, width=330)


if __name__ == "__main__":
    experiment()
