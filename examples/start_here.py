import numpy as np

from mab import visualize
from mab.game import Game


def stationary_mab_distribution(save_plot=True):
    # Sampling mean and standard deviation for a stationary MAB with 10 arms
    np.random.seed(42)
    means = np.random.uniform(-3, 3, size=10)
    stds = np.random.uniform(1, 2, size=10)
    # instantiate the MAB
    game = Game(10, means, stds)
    # Plot the distribution
    game.play(5)

    if save_plot:
        visualize.violin_plot(
            game, save_path="../docs/figures/initial_distributions.pdf"
        )
    else:
        visualize.violin_plot(game, show=True, vertical=True, figsize=(8, 4))


def non_stationary_benchmark_slideshow():
    K = 8
    timepoints = 15
    output_folder = "tmp_data_1"

    np.random.seed(42)

    means = np.zeros([timepoints, K], dtype=np.float)
    stds = np.zeros([timepoints, K], dtype=np.float)

    means[0, :] = np.random.uniform(-3, 3, size=K)
    stds[0, :] = np.random.uniform(1, 3, size=K)

    for row in range(1, timepoints):
        means[row, :] = means[row - 1, :] + 0.1 * np.random.choice([-1, 1], size=[1, K])
        stds[row, :] = stds[row - 1, :] + 0.1 * np.random.choice(
            [-1, 1], size=[1, K], p=(0.2, 0.8)
        )

    game = Game(timepoints, means, stds)

    visualize.slideshow_violin_distributions(
        game,
        output_folder=output_folder,
        violin_axis_limit=(-15, 15),
        time_point_annotation=True,
    )


def play_a_thousand_dollars_stationary_game():
    # Sampling mean and standard deviation
    np.random.seed(43)
    means = np.random.uniform(-3, 3, size=10)
    stds = np.abs(np.random.normal(3, 1, size=10))
    # initialising mab
    game = Game(1000, means, stds)

    means_hat, stds_hat, reward_per_arm, pulls_per_arm = game.play(
        initial_t_explorations=100, exploration_strategy="naive"
    )
    visualize.violin_plot(
        game, arms_annotations=pulls_per_arm, violin_axis_limit=(-20, 20), show=True
    )


def play_a_thousand_dollars_non_stationary_game():
    game = Game(1000)

    means_hat, stds_hat, reward_per_arm, pulls_per_arm = game.play(
        initial_t_explorations=100, exploration_strategy="naive"
    )
    visualize.violin_plot(
        game, arms_annotations=pulls_per_arm, violin_axis_limit=(-20, 20), show=True
    )


def visualize_q_matrix():
    K = 10

    np.random.seed(42)

    means = np.random.uniform(-3, 3, size=K)
    stds = np.random.uniform(1, 3, size=K)

    game = Game(200, means, stds)

    for t in range(game.T):
        k = np.random.choice(range(K))
        game.select_arm(k)

    visualize.get_evolving_grid(
        game,
        show_data_at_tp=54,
        offset_before=12,
        offset_after=5,
        last_tp_off_grid=False,
        show=True,
    )


def visualize_q_matrix_slideshow():
    timepoints = 40
    K = 10
    initial_t_explorations = 15
    output_folder = "tmp_data_2"

    np.random.seed(42)

    means = np.zeros([timepoints, K], dtype=np.float)
    stds = np.zeros([timepoints, K], dtype=np.float)

    means[0, :] = np.random.uniform(-3, 3, size=K)
    stds[0, :] = np.random.uniform(1, 3, size=K)

    for row in range(1, timepoints):
        means[row, :] = means[row - 1, :]
        stds[row, :] = stds[row - 1, :]

    game = Game(timepoints, means, stds)

    game.play(initial_t_explorations=initial_t_explorations)

    visualize.get_grid_and_violins_dynamic(game, output_folder=output_folder)


def visualize_q_matrix_slideshow_second_case():
    timepoints = 120
    K = 10
    initial_t_explorations = 20
    output_folder = "tmp_data_2"

    np.random.seed(40)

    means = np.zeros([timepoints, K], dtype=np.float)
    stds = np.zeros([timepoints, K], dtype=np.float)

    means[0, :] = np.array([3 * np.sin((np.pi / 4) * x) for x in range(K)])
    stds[0, :] = np.random.uniform(1, 3, size=K)

    for row in range(1, timepoints):
        means[row, :] = np.array(
            [3 * np.sin((np.pi / 4) * x + 0.05 * row + 3) for x in range(K)]
        )
        stds[row, :] = stds[row - 1, :]

    game = Game(timepoints, means, stds)

    game.play(
        initial_t_explorations=initial_t_explorations,
        epsilon=0.2,
        exploration_strategy="upper confidence",
        adjust_alpha=True,
    )

    visualize.get_grid_and_violins_dynamic(game, output_folder=output_folder)


if __name__ == "__main__":
    # -- Uncomment the one you want to run --

    stationary_mab_distribution()
    # non_stationary_benchmark_slideshow()
    # play_a_thousand_dollars_stationary_game()
    # play_a_thousand_dollars_non_stationary_game()
    # visualize_q_matrix()
    # visualize_q_matrix_slideshow()
    # visualize_q_matrix_slideshow_second_case()
