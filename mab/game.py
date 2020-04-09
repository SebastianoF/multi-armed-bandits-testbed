import numpy as np

from mab.algorithms import epsilon_greedy


class Game:

    def __init__(self, num_draws, means=None, stds=None, params_generators=(10, 500, 50)):
        if means is None or stds is None:
            means, stds = self._generate_parameters(params_generators)

        self.means = means
        self.stds = stds

        if not means.shape == stds.shape:
            raise ValueError("means and stds must have the same shape")

        if len(means.shape) <= 1:
            self.means = means.reshape([1, means.shape[0]])
            self.stds = stds.reshape([1, stds.shape[0]])

        self.K = self.means.shape[1]
        self.T = num_draws
        self.tp = 0
        self.q = np.zeros([num_draws, self.K], dtype=np.float)

    @staticmethod
    def _generate_parameters(params_generators):
        number_of_arms = params_generators[0]
        total_tp = params_generators[1]
        change_intervals = params_generators[2]

        means = np.zeros([total_tp, number_of_arms], dtype=np.float)
        stds = np.zeros([total_tp, number_of_arms], dtype=np.float)

        means[0, :] = np.random.uniform(-3, 3, size=number_of_arms)
        stds[0, :] = np.random.uniform(1, 3, size=number_of_arms)

        for row in range(1, total_tp):
            if row % change_intervals == 0:
                means[row, :] = means[row - 1, :] + 0.1 * np.random.choice([-1, 1], size=[1, number_of_arms])
                stds[row, :] = stds[row - 1, :] + 0.1 * np.random.choice([-1, 1], size=[1, number_of_arms], p=(.4, .6))
            else:
                means[row, :] = means[row - 1, :]
                stds[row, :] = stds[row - 1, :]

        return means, stds

    def reset(self):
        self.tp = 0
        self.q = np.zeros([self.T, self.K], dtype=np.float)

    def select_arm(self, k):
        """A random sample from the given arm k. t specified for non-stationary cases."""
        if k not in range(self.K):
            raise ValueError(f"The arm k={k} must be between 0 and {self.K}.")

        row = self.tp % self.means.shape[0]
        q = np.random.normal(self.means[row, k], self.stds[row, k])
        self.q[self.tp, k] = q
        self.tp += 1
        return q

    def play(self, initial_t_explorations=100, exploration_strategy="random"):
        return epsilon_greedy(self, initial_t_explorations, exploration_strategy=exploration_strategy)

    def sample_all_arms(self, num_samples=1000, time_point=None):
        """Return num_samples for each arm in a list of lists."""
        if time_point is None:
            time_point = self.tp
        return [sorted(np.random.normal(m, s, num_samples))
                for m, s in zip(self.means[time_point, :], self.stds[time_point, :])]

    def compute_optimal_k(self, time_point=None):
        if time_point is None:
            time_point = self.tp
        sampled_reward_per_arm = [np.sum(np.where(np.array(arm_sampling) < 0, 0, arm_sampling))
                                  for arm_sampling in self.sample_all_arms(time_point=time_point)]
        return np.argmax(sampled_reward_per_arm)
