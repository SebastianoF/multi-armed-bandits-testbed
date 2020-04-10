import numpy as np

from mab.game import Game

K = 10
K_BEST = 7
K_BEST_ALTERNATIVE = 4
T = 500
T_INTERVAL = 50


def get_game_one_winning_arm():
    means, stds = -2 * np.ones([T, K]), np.ones([T, K])
    for row in range(T):
        if 0 <= row % (T_INTERVAL) < int(T_INTERVAL):
            means[row, K_BEST] = 2
        else:
            means[row, K_BEST_ALTERNATIVE] = 1
    mab = Game(T, means, stds)
    return mab


def test_compute_optimal_k_stationary():
    mab = get_game_one_winning_arm()

    # store initial values:
    K, means, stds = mab.K, mab.means, mab.stds

    # Check the internal method works:
    k_hat = mab.compute_optimal_k()
    np.testing.assert_equal(k_hat, K_BEST)

    # check no input have changed during manipulations:
    np.testing.assert_equal(K, mab.K)
    np.testing.assert_array_equal(means, mab.means)
    np.testing.assert_array_equal(stds, mab.stds)


def test_one_winning_arm_random_strategy():
    game = get_game_one_winning_arm()

    # store initial values:
    K, means, stds = game.K, game.means, game.stds

    # Check random strategy works:
    means_hat, _, rewards_per_arm, pulls_per_arm = game.play(
        initial_t_explorations=100, exploration_strategy="naive"
    )

    np.testing.assert_equal(np.argmax(np.clip(means_hat, 0, np.inf)), K_BEST)
    np.testing.assert_equal(np.argmax(rewards_per_arm), K_BEST)
    np.testing.assert_equal(np.argmax(pulls_per_arm), K_BEST)

    # check no input have changed during manipulations:
    np.testing.assert_equal(K, game.K)
    np.testing.assert_array_equal(means, game.means)
    np.testing.assert_array_equal(stds, game.stds)


def test_one_winning_arm_best_reward_strategy():
    game = get_game_one_winning_arm()

    # store initial values:
    K, means, stds = game.K, game.means, game.stds

    # Check random strategy works:
    means_hat, _, rewards_per_arm, pulls_per_arm = game.play(
        initial_t_explorations=100, exploration_strategy="best reward"
    )

    np.testing.assert_equal(np.argmax(means_hat), K_BEST)
    np.testing.assert_equal(np.argmax(rewards_per_arm), K_BEST)
    np.testing.assert_equal(np.argmax(pulls_per_arm), K_BEST)

    # check no input have changed during manipulations:
    np.testing.assert_equal(K, game.K)
    np.testing.assert_array_equal(means, game.means)
    np.testing.assert_array_equal(stds, game.stds)


def test_one_winning_arm_least_explored_strategy():
    game = get_game_one_winning_arm()

    # store initial values:
    K, means, stds = game.K, game.means, game.stds

    # Check random strategy works:
    means_hat, _, rewards_per_arm, pulls_per_arm = game.play(
        initial_t_explorations=100, exploration_strategy="least explored"
    )

    np.testing.assert_equal(np.argmax(means_hat), K_BEST)
    np.testing.assert_equal(np.argmax(rewards_per_arm), K_BEST)
    np.testing.assert_equal(np.argmax(pulls_per_arm), K_BEST)

    # check no input have changed during manipulations:
    np.testing.assert_equal(K, game.K)
    np.testing.assert_array_equal(means, game.means)
    np.testing.assert_array_equal(stds, game.stds)


def test_one_winning_arm_upper_confidence_strategy():
    game = get_game_one_winning_arm()

    # store initial values:
    K, means, stds = game.K, game.means, game.stds

    # Check random strategy works:
    means_hat, _, rewards_per_arm, pulls_per_arm = game.play(
        initial_t_explorations=100, exploration_strategy="upper confidence"
    )

    np.testing.assert_equal(np.argmax(means_hat), K_BEST)
    np.testing.assert_equal(np.argmax(rewards_per_arm), K_BEST)
    np.testing.assert_equal(np.argmax(pulls_per_arm), K_BEST)

    # check no input have changed during manipulations:
    np.testing.assert_equal(K, game.K)
    np.testing.assert_array_equal(means, game.means)
    np.testing.assert_array_equal(stds, game.stds)


def test_one_winning_arm_gradient_strategy():
    game = get_game_one_winning_arm()

    # store initial values:
    K, means, stds = game.K, game.means, game.stds

    # Check random strategy works:
    means_hat, _, rewards_per_arm, pulls_per_arm = game.play(
        initial_t_explorations=100, exploration_strategy="gradient"
    )

    np.testing.assert_equal(np.argmax(means_hat), K_BEST)
    np.testing.assert_equal(np.argmax(rewards_per_arm), K_BEST)
    np.testing.assert_equal(np.argmax(pulls_per_arm), K_BEST)

    # check no input have changed during manipulations:
    np.testing.assert_equal(K, game.K)
    np.testing.assert_array_equal(means, game.means)
    np.testing.assert_array_equal(stds, game.stds)
