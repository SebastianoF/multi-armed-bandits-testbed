import numpy as np

from mab.multi_armed_bandits import MultiArmedBandit


class Player:
    def __init__(self, T: int , mab: MultiArmedBandit, seed: int):
        self.T = T
        self.mab = mab
        self.q = np.ones((self.mab.K, self.T)) * np.nan
        self.total_reward = 0
        self.seed = seed

    def reset_parameters(self):
        self.q = np.ones((self.mab.K, self.T)) * np.nan
        self.total_reward = 0

    def get_plot_q(self, t, t_on_x_axis=20, left_margin=3):
        """Returns the fig (matplotlib instance) of a representation of the q matrix"""
        raise NotImplementedError()


