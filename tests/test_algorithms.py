import numpy as np

from mab.multi_armed_bandit import MultiArmedBandit
from mab.player import Player
from mab.strategies import epsilon_greedy


def get_mab_one_winning_arm(winning_arm, K=10):
    means, stds = -2 * np.ones(K), np.ones(K)
    means[winning_arm] = 2
    mab = MultiArmedBandit(means, stds)
    return mab


def test_compute_optimal_k():

    k_best = 7
    mab = get_mab_one_winning_arm(k_best)

    # store initial values:
    K, means, stds = mab.K, mab.means, mab.stds

    # Check the internal method works:
    k_hat = mab.compute_optimal_k()
    np.testing.assert_equal(k_hat, k_best)

    # check no input have changed during manipulations:
    np.testing.assert_equal(K, mab.K)
    np.testing.assert_array_equal(means, mab.means)
    np.testing.assert_array_equal(stds, mab.stds)


def test_one_winning_arm_random_strategy():
    k_best = 7
    mab = get_mab_one_winning_arm(k_best)
    player = Player(T=1000, mab=mab)

    # store initial values:
    K, means, stds = mab.K, mab.means, mab.stds

    # Check random strategy works:
    means_hat, _, rewards_per_arm, pulls_per_arm = epsilon_greedy(
        player, initial_t_explorations=100, exploration_strategy="random"
    )

    np.testing.assert_equal(np.argmax(np.clip(means_hat, 0, np.inf)), k_best)
    np.testing.assert_equal(np.argmax(rewards_per_arm), k_best)
    np.testing.assert_equal(np.argmax(pulls_per_arm), k_best)

    # check no input have changed during manipulations:
    np.testing.assert_equal(K, mab.K)
    np.testing.assert_array_equal(means, mab.means)
    np.testing.assert_array_equal(stds, mab.stds)


def test_one_winning_arm_best_reward_strategy():
    k_best = 7
    mab = get_mab_one_winning_arm(k_best)
    player = Player(T=1000, mab=mab)

    # store initial values:
    K, means, stds = mab.K, mab.means, mab.stds

    np.random.seed(25)

    # Check random strategy works:
    means_hat, _, rewards_per_arm, pulls_per_arm = epsilon_greedy(
        player, initial_t_explorations=100, exploration_strategy="best reward"
    )

    np.testing.assert_equal(np.argmax(means_hat), k_best)
    np.testing.assert_equal(np.argmax(rewards_per_arm), k_best)
    np.testing.assert_equal(np.argmax(pulls_per_arm), k_best)

    # check no input have changed during manipulations:
    np.testing.assert_equal(K, mab.K)
    np.testing.assert_array_equal(means, mab.means)
    np.testing.assert_array_equal(stds, mab.stds)


def test_one_winning_arm_least_explored_strategy():
    k_best = 7
    mab = get_mab_one_winning_arm(k_best)
    player = Player(T=1000, mab=mab)

    # store initial values:
    K, means, stds = mab.K, mab.means, mab.stds

    # Check random strategy works:
    means_hat, _, rewards_per_arm, pulls_per_arm = epsilon_greedy(
        player, initial_t_explorations=100, exploration_strategy="least explored"
    )

    np.testing.assert_equal(np.argmax(means_hat), k_best)
    np.testing.assert_equal(np.argmax(rewards_per_arm), k_best)
    np.testing.assert_equal(np.argmax(pulls_per_arm), k_best)

    # check no input have changed during manipulations:
    np.testing.assert_equal(K, mab.K)
    np.testing.assert_array_equal(means, mab.means)
    np.testing.assert_array_equal(stds, mab.stds)


def test_one_winning_arm_upper_confidence_strategy():
    k_best = 7
    mab = get_mab_one_winning_arm(k_best)
    player = Player(T=1000, mab=mab)

    # store initial values:
    K, means, stds = mab.K, mab.means, mab.stds

    # Check random strategy works:
    means_hat, _, rewards_per_arm, pulls_per_arm = epsilon_greedy(
        player, initial_t_explorations=100, exploration_strategy="upper confidence"
    )

    np.testing.assert_equal(np.argmax(means_hat), k_best)
    np.testing.assert_equal(np.argmax(rewards_per_arm), k_best)
    np.testing.assert_equal(np.argmax(pulls_per_arm), k_best)

    # check no input have changed during manipulations:
    np.testing.assert_equal(K, mab.K)
    np.testing.assert_array_equal(means, mab.means)
    np.testing.assert_array_equal(stds, mab.stds)


def test_one_winning_arm_gradient_strategy():
    k_best = 7
    mab = get_mab_one_winning_arm(k_best)
    player = Player(T=1000, mab=mab)

    # store initial values:
    K, means, stds = mab.K, mab.means, mab.stds

    # Check random strategy works:
    means_hat, _, rewards_per_arm, pulls_per_arm = epsilon_greedy(
        player, initial_t_explorations=100, exploration_strategy="gradient"
    )

    np.testing.assert_equal(np.argmax(means_hat), k_best)
    np.testing.assert_equal(np.argmax(rewards_per_arm), k_best)
    np.testing.assert_equal(np.argmax(pulls_per_arm), k_best)

    # check no input have changed during manipulations:
    np.testing.assert_equal(K, mab.K)
    np.testing.assert_array_equal(means, mab.means)
    np.testing.assert_array_equal(stds, mab.stds)
