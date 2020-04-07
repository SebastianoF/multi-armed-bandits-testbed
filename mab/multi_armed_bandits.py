import numpy as np
from matplotlib import pyplot as plt

from mab.plot_utils import violin_plot


class MultiArmedBandit:
    def __init__(self, K=10, means=None, stds=None, seed=None):
        self.K = K
        self.means = means
        self.stds = stds
        self.seed = seed

        if self.means is None and self.stds is None:
            self._init_random_distributions()

        if self.seed is not None:
            np.random.seed(self.seed)

    def _init_random_distributions(self):
        """Set up seeds and initial mus and stds"""

        self.means = list(np.random.uniform(-3, 3, size=self.K))
        self.stds = list(np.abs(np.random.normal(2, 4, size=self.K)))

    def sample_all_arms(self, num_samples=100):
        """Return num_samples for each arm in a list of lists."""
        return [sorted(np.random.normal(m, s, num_samples)) for m, s in zip(self.means, self.stds)]

    def get_plot_distributions(self, num_samples_per_violin=100, y_axis_limit=None, timepoint=None):
        """Returns the fig (matplotlib instance) of a representation of the arms"""
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
        violin_plot(ax, self.sample_all_arms(num_samples=num_samples_per_violin), y_axis_limit, timepoint)
        fig.subplots_adjust(bottom=0.15, wspace=0.05)
        return fig

    def draw_from_arm(self, k):
        """A random sample from the given arm k. t specified for non-stationary cases."""
        if k not in range(1, self.K + 1):
            raise ValueError(f"The arm k={k} must be between 1 and {self.K + 1}.")
        return np.random.normal(self.means[k], self.stds[k])

    def compute_optimal_k(self):
        """Get the arm with the best reward - based on empirical sampling"""
        arms_samples = self.sample_all_arms()
        arms_samples = np.where(arms_samples < 0, 0, arms_samples)
        return np.argmax(np.sum(arms_samples))


class NonStationaryMultiArmedBandit(MultiArmedBandit):

    def __init__(self, K=10, means=None, stds=None, seed=None, updates_intervals=()):
        super().__init__(K=K, means=means, stds=stds, seed=seed)
        self.timepoint = 0
        self.updates_interval = updates_intervals
        self.update_num = 0

        if not updates_intervals:
            self._init_update_intervals()

        if self.seed is not None:
            np.random.seed(self.seed)

        # to keep track of the history of changes:
        self.historical_means = np.zeros([len(self.updates_interval), self.K], dtype=np.float)
        self.historical_stds = np.zeros([len(self.updates_interval), self.K], dtype=np.float)
        self.historical_means[self.update_num, :] = self.means
        self.historical_stds[self.update_num, :] = self.stds

    def _init_update_intervals(self):
        """Set up default interval updates: one update every 100 draws for 1000 draws expected"""
        self.updates_interval = [100 * (t + 1) for t in range(9)]

    def update_parameters_random(self, epsilon_mean=0.1, epsilon_std=0.1):
        """Update step for non stationary mab"""
        for k in range(self.K):
            sign = np.random.choice([1, -1], 2)
            self.means[k] = sign[0] * epsilon_mean + self.means[k]
            self.stds[k] = sign[1] * epsilon_std + self.stds[k]

        self.update_num += 1
        self.historical_means[self.update_num, :] = self.means
        self.historical_stds[self.update_num, :] = self.stds

    def draw_from_arm(self, k):
        """A random sample from the given arm k. t specified for non-stationary cases."""
        if k not in range(1, self.K + 1):
            raise ValueError(f"The arm k={k} must be between 1 and {self.K + 1}.")
        if self.timepoint in self.updates_interval:
            self.update_parameters_random()
        self.timepoint += 1
        return np.random.normal(self.means[k], self.stds[k])

