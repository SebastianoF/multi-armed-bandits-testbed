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

    def _init_random_distributions(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.means = list(np.random.uniform(-3, 3, size=self.K))
        self.stds = list(np.abs(np.random.normal(2, 4, size=self.K)))

    def sample_all_arms(self, num_samples=100):
        return [sorted(np.random.normal(m, s, num_samples)) for m, s in zip(self.means, self.stds)]

    def get_plt_distribution(self, num_samples_per_violin=100, y_axis_limit=None, timepoint=None):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
        violin_plot(ax, self.sample_all_arms(num_samples=num_samples_per_violin), y_axis_limit, timepoint)
        fig.subplots_adjust(bottom=0.15, wspace=0.05)
        return fig

    def draw_from_arm(self, k):
        if k not in range(1, self.K + 1):
            raise ValueError(f"The arm k={k} must be between 1 and {self.K + 1}.")
        return np.random.normal(self.means[k], self.stds[k])

    def update_parameters_random(self, epsilon_mean=0.1, epsilon_std=0.1):
        for k in range(self.K):
            sign = np.random.choice([1, -1], 2)
            self.means[k] = sign[0] * epsilon_mean + self.means[k]
            self.stds[k] = sign[1] * epsilon_std + self.stds[k]
