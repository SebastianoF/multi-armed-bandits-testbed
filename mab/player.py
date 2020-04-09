import numpy as np

from mab.multi_armed_bandit import MultiArmedBandit
from mab.utils import evolutionary_grid


class Player:
    def __init__(self, T: int , mab: MultiArmedBandit):
        self.T = T
        self.mab = mab
        self.q = np.ones((self.mab.K, self.T)) * np.nan
        self.total_reward = 0
        self.tp = 0

    def reset_parameters(self):
        self.q = np.ones((self.mab.K, self.T)) * np.nan
        self.total_reward = 0
        self.tp = 0

    def select_arm(self, k):
        q = self.mab.draw_from_arm(k)
        self.q[k, self.tp] = q
        self.tp += 1
        return q

    def get_evolving_grid_plot(
            self,
            ax,
            show_data_at_tp=50,
            offset_before=12,
            offset_after=5,
            last_tp_off_grid=False
    ):
        ax, im = evolutionary_grid(
            ax,
            self.q,
            show_data_at_tp=show_data_at_tp,
            offset_before=offset_before,
            offset_after=offset_after,
            last_tp_off_grid=last_tp_off_grid
        )
        return ax, im


