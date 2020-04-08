import numpy as np
from matplotlib import pyplot as plt

from mab.plot_utils import violin_plot


class MultiArmedBandit:
    def __init__(self, K=10, means=None, stds=None):
        self.K = K
        self.means = means
        self.stds = stds

        if self.means is None and self.stds is None:
            self._init_random_distributions()

    def _init_random_distributions(self):
        """Set up initial mus and stds"""
        self.means = list(np.random.uniform(-3, 3, size=self.K))
        self.stds = list(np.abs(np.random.normal(3, 1, size=self.K)))

    def sample_all_arms(self, num_samples=100):
        """Return num_samples for each arm in a list of lists."""
        return [sorted(np.random.normal(m, s, num_samples)) for m, s in zip(self.means, self.stds)]

    def get_plot_distributions(
            self, num_samples_per_violin=1000, y_axis_limit=None, timepoint=None, arms_annotations=None
    ):
        """Returns the fig (matplotlib instance) of a representation of the arms"""
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
        violin_plot(
            ax, self.sample_all_arms(num_samples=num_samples_per_violin), y_axis_limit, timepoint, arms_annotations
        )
        fig.subplots_adjust(bottom=0.15, wspace=0.05)
        return fig

    def draw_from_arm(self, k):
        """A random sample from the given arm k. t specified for non-stationary cases."""
        if k not in range(self.K):
            raise ValueError(f"The arm k={k} must be between 1 and {self.K + 1}.")
        return np.random.normal(self.means[k], self.stds[k])

    def compute_optimal_k(self):
        """Get the arm with the best reward - based on empirical sampling"""
        arms_samples = self.sample_all_arms()
        arms_samples = [np.where(np.array(ars) < 0, 0, ars) for ars in arms_samples]
        return np.argmax(np.sum(arms_samples))


class NonStationaryMultiArmedBandit(MultiArmedBandit):

    def __init__(self, K=10, means=None, stds=None, updates_intervals=()):
        super().__init__(K=K, means=means, stds=stds)
        self.timepoint = 0
        self.updates_interval = updates_intervals
        self.update_num = 0

        if not updates_intervals:
            self._init_update_intervals()

        # to keep track of the history of changes:
        self.historical_means = np.zeros([len(self.updates_interval), self.K], dtype=np.float)
        self.historical_stds = np.zeros([len(self.updates_interval), self.K], dtype=np.float)
        self.historical_means[self.update_num, :] = self.means
        self.historical_stds[self.update_num, :] = self.stds

    def _init_update_intervals(self):
        """Set up default interval updates: one update every 100 draws for 1000 draws expected"""
        self.updates_interval = [100 * (t + 1) for t in range(9)]

    def update_parameters_random(self, epsilon_mean=0.1, epsilon_std=0.1, sign_means=None, sign_stds=None):
        """Update step for non stationary mab, variations trend can be given or random."""
        if sign_means is None:
            sign_means = np.random.choice([1, -1], self.K)
        if sign_stds is None:
            sign_stds = np.random.choice([1, -1], self.K)

        self.means = sign_means * epsilon_mean + self.means
        self.stds = sign_stds * epsilon_std + self.stds

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
