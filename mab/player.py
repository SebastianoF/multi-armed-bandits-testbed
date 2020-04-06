import numpy as np

from mab.multi_armed_bandits import MultiArmedBandit


class Player:
    def __init__(self, T: int , mab: MultiArmedBandit, seed: int):
        self.T = T
        self.mab = mab
        self.q = np.zeros((self.mab.K, self.T))
        self.total_reward = 0
        self.seed = seed

    def play(self, srategy):
        # TODO
        raise NotImplementedError
