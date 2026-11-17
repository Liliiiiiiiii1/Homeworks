"""
A/B testing with multi-armed bandits: Epsilon-Greedy vs Thompson Sampling.

Four advertisement options (arms) with true mean rewards [1, 2, 3, 4] are
simulated for 20000 trials with each algorithm. Rewards are Gaussian with a
known precision (tau = 1 / variance), which is what Thompson Sampling with
known precision relies on.

Run this file directly to execute the experiments, produce the plots,
write the CSV file and print the reports.
"""
############################### LOGGER
from abc import ABC, abstractmethod

import matplotlib
matplotlib.use("Agg")  # no GUI needed, plots are saved to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

BANDIT_REWARDS = [1, 2, 3, 4]
NUM_TRIALS = 20000
TRUE_PRECISION = 1.0  # known precision of the reward noise (variance = 1)
SEED = 42


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():
    """Plotting helpers for the bandit experiments."""

    def plot1(self, algorithms, filename="plot1_learning_process.png"):
        """Visualize the learning process of each algorithm (linear and log scale).

        For every algorithm, the running average reward is plotted against
        the trial number, on a linear x-axis (left) and on a log x-axis (right).
        The dashed line is the optimal mean reward.

        Parameters
        ----------
        algorithms : list of Bandit
            Algorithms that already ran ``experiment()``.
        filename : str
            Where the figure is saved.
        """
        fig, axes = plt.subplots(len(algorithms), 2, figsize=(13, 4.5 * len(algorithms)))
        axes = np.atleast_2d(axes)
        for row, algo in enumerate(algorithms):
            trials = np.arange(1, len(algo.rewards) + 1)
            avg = np.cumsum(algo.rewards) / trials
            for col, scale in enumerate(["linear", "log"]):
                ax = axes[row, col]
                ax.plot(trials, avg, label="Average reward")
                ax.axhline(max(algo.p), color="r", linestyle="--", label="Optimal mean")
                ax.set_xscale(scale)
                ax.set_title(f"{algo.name}: learning process ({scale} scale)")
                ax.set_xlabel("Trial")
                ax.set_ylabel("Average reward")
                ax.legend()
                ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(filename, dpi=120)
        plt.close(fig)
        logger.info(f"Saved {filename}")

    def plot2(self, eg, ts, filename="plot2_comparison.png"):
        """Compare cumulative rewards and cumulative regrets of both algorithms.

        Parameters
        ----------
        eg : EpsilonGreedy
            Epsilon-Greedy algorithm after ``experiment()``.
        ts : ThompsonSampling
            Thompson Sampling algorithm after ``experiment()``.
        filename : str
            Where the figure is saved.
        """
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for algo in (eg, ts):
            axes[0].plot(np.cumsum(algo.rewards), label=algo.name)
            axes[1].plot(np.cumsum(algo.regrets), label=algo.name)
        axes[0].set_title("Cumulative reward")
        axes[1].set_title("Cumulative regret")
        for ax in axes:
            ax.set_xlabel("Trial")
            ax.legend()
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Cumulative reward")
        axes[1].set_ylabel("Cumulative regret")
        fig.tight_layout()
        fig.savefig(filename, dpi=120)
        plt.close(fig)
        logger.info(f"Saved {filename}")

#--------------------------------------#


class EpsilonGreedy(Bandit):
    """Epsilon-Greedy bandit with epsilon decaying as 1/t.

    Parameters
    ----------
    p : list of float
        True mean reward of each arm.
    precision : float
        Known precision (1/variance) of the Gaussian reward noise.
    eps0 : float
        Initial epsilon; at trial t, epsilon = eps0 / t.
    """

    name = "EpsilonGreedy"

    def __init__(self, p, precision=TRUE_PRECISION, eps0=1.0, seed=SEED):
        self.p = np.asarray(p, dtype=float)
        self.k = len(p)
        self.precision = precision
        self.eps0 = eps0
        self.rng = np.random.default_rng(seed)
        self.estimates = np.zeros(self.k)  # estimated mean of each arm
        self.counts = np.zeros(self.k, dtype=int)
        self.t = 0
        self.arms = []
        self.rewards = []
        self.regrets = []

    def __repr__(self):
        return f"EpsilonGreedy(p={self.p.tolist()}, eps0={self.eps0})"

    def pull(self):
        """Choose an arm (explore with probability eps0/t) and sample its reward.

        Returns
        -------
        tuple of (int, float)
            The chosen arm and the observed reward.
        """
        self.t += 1
        epsilon = self.eps0 / self.t
        if self.rng.random() < epsilon:
            arm = int(self.rng.integers(self.k))
        else:
            arm = int(np.argmax(self.estimates))
        reward = self.rng.normal(self.p[arm], 1 / np.sqrt(self.precision))
        return arm, reward

    def update(self, arm, reward):
        """Update the running mean estimate of the chosen arm.

        Parameters
        ----------
        arm : int
            Index of the pulled arm.
        reward : float
            Observed reward.
        """
        self.counts[arm] += 1
        self.estimates[arm] += (reward - self.estimates[arm]) / self.counts[arm]
        self.arms.append(arm)
        self.rewards.append(reward)
        self.regrets.append(self.p.max() - self.p[arm])

    def experiment(self, num_trials=NUM_TRIALS):
        """Run the experiment for ``num_trials`` trials."""
        logger.info(f"Running {self.name} for {num_trials} trials")
        for _ in range(num_trials):
            arm, reward = self.pull()
            self.update(arm, reward)

    def report(self):
        """Print the average reward, average regret and cumulative totals.

        Returns
        -------
        pandas.DataFrame
            Columns ``Bandit``, ``Reward``, ``Algorithm`` for every trial.
        """
        return _report(self)

#--------------------------------------#


class ThompsonSampling(Bandit):
    """Thompson Sampling for Gaussian rewards with known precision.

    Each arm has a Gaussian posterior over its mean. With known reward
    precision ``tau`` and a prior N(mu0, 1/lambda0), after n observations
    with sum S the posterior is N((lambda0*mu0 + tau*S)/lambda, 1/lambda),
    where lambda = lambda0 + n*tau.

    Parameters
    ----------
    p : list of float
        True mean reward of each arm.
    precision : float
        Known precision (1/variance) of the reward noise.
    prior_mean, prior_precision : float
        Parameters of the Gaussian prior on each arm's mean.
    """

    name = "ThompsonSampling"

    def __init__(self, p, precision=TRUE_PRECISION, prior_mean=0.0,
                 prior_precision=0.01, seed=SEED + 1):
        self.p = np.asarray(p, dtype=float)
        self.k = len(p)
        self.precision = precision
        self.post_mean = np.full(self.k, prior_mean, dtype=float)
        self.post_precision = np.full(self.k, prior_precision, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.arms = []
        self.rewards = []
        self.regrets = []

    def __repr__(self):
        return f"ThompsonSampling(p={self.p.tolist()}, precision={self.precision})"

    def pull(self):
        """Sample a mean from each posterior, play the best, observe its reward.

        Returns
        -------
        tuple of (int, float)
            The chosen arm and the observed reward.
        """
        samples = self.rng.normal(self.post_mean, 1 / np.sqrt(self.post_precision))
        arm = int(np.argmax(samples))
        reward = self.rng.normal(self.p[arm], 1 / np.sqrt(self.precision))
        return arm, reward

    def update(self, arm, reward):
        """Bayesian update of the chosen arm's Gaussian posterior.

        Parameters
        ----------
        arm : int
            Index of the pulled arm.
        reward : float
            Observed reward.
        """
        new_precision = self.post_precision[arm] + self.precision
        self.post_mean[arm] = (
            self.post_precision[arm] * self.post_mean[arm] + self.precision * reward
        ) / new_precision
        self.post_precision[arm] = new_precision
        self.arms.append(arm)
        self.rewards.append(reward)
        self.regrets.append(self.p.max() - self.p[arm])

    def experiment(self, num_trials=NUM_TRIALS):
        """Run the experiment for ``num_trials`` trials."""
        logger.info(f"Running {self.name} for {num_trials} trials")
        for _ in range(num_trials):
            arm, reward = self.pull()
            self.update(arm, reward)

    def report(self):
        """Print the average reward, average regret and cumulative totals.

        Returns
        -------
        pandas.DataFrame
            Columns ``Bandit``, ``Reward``, ``Algorithm`` for every trial.
        """
        return _report(self)


def _report(algo):
    """Shared reporting logic: log averages/cumulatives, return the data frame."""
    n = len(algo.rewards)
    logger.info(f"[{algo.name}] Average reward: {np.mean(algo.rewards):.4f}")
    logger.info(f"[{algo.name}] Average regret: {np.mean(algo.regrets):.4f}")
    logger.info(f"[{algo.name}] Cumulative reward: {np.sum(algo.rewards):.2f}")
    logger.info(f"[{algo.name}] Cumulative regret: {np.sum(algo.regrets):.2f}")
    counts = np.bincount(algo.arms, minlength=algo.k)
    logger.info(f"[{algo.name}] Pulls per arm over {n} trials: {counts.tolist()}")
    return pd.DataFrame({
        "Bandit": algo.arms,
        "Reward": algo.rewards,
        "Algorithm": algo.name,
    })


def comparison(eg, ts, filename="comparison_per_arm.png"):
    """Compare both algorithms visually: how often each arm was chosen.

    Complements ``Visualization.plot2`` (cumulative reward / regret) by showing
    that Thompson Sampling concentrates its pulls on the best arm faster.

    Parameters
    ----------
    eg : EpsilonGreedy
    ts : ThompsonSampling
    filename : str
        Where the figure is saved.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.38
    x = np.arange(eg.k)
    ax.bar(x - width / 2, np.bincount(eg.arms, minlength=eg.k), width, label=eg.name)
    ax.bar(x + width / 2, np.bincount(ts.arms, minlength=ts.k), width, label=ts.name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Bandit {int(r)}" for r in eg.p])
    ax.set_ylabel("Number of pulls")
    ax.set_title("Pulls per bandit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    logger.info(f"Saved {filename}")


if __name__ == '__main__':
    eg = EpsilonGreedy(BANDIT_REWARDS)
    ts = ThompsonSampling(BANDIT_REWARDS)
    logger.info(f"{eg}")
    logger.info(f"{ts}")

    eg.experiment()
    ts.experiment()

    df = pd.concat([eg.report(), ts.report()], ignore_index=True)
    df.to_csv("rewards.csv", index=False)
    logger.info(f"Saved rewards.csv with {len(df)} rows")

    viz = Visualization()
    viz.plot1([eg, ts])
    viz.plot2(eg, ts)
    comparison(eg, ts)
