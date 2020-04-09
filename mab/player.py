import numpy as np
import matplotlib.pyplot as plt

from mab.multi_armed_bandit import MultiArmedBandit
from mab.utils import evolutionary_grid


class Player:
    def __init__(self, T: int , mab: MultiArmedBandit):
        self.T = T
        self.mab = mab
        self.q = np.ones((self.mab.K, self.T)) * np.nan
        self.total_reward = 0

    def reset_parameters(self):
        self.q = np.ones((self.mab.K, self.T)) * np.nan
        self.total_reward = 0

    def get_evolving_grid_plot(
            self,
            ax,
            show_data_at_tp=50,
            offset_before=12,
            offset_after=5,
            show_plus_one=False
    ):
        ax, im = evolutionary_grid(
            ax,
            self.q,
            show_data_at_tp=show_data_at_tp,
            offset_before=offset_before,
            offset_after=offset_after,
            show_plus_one=show_plus_one
        )
        return ax, im


