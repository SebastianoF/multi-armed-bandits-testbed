import os
import shutil

import numpy as np
from matplotlib import pyplot as plt

from mab.algorithms import epsilon_greedy
from mab.game import Game
from mab import visualize


def stationary_mab_distribution(save_plot=False):
    # Sampling mean and standard deviation for a stationary MAB with 10 arms
    np.random.seed(42)
    means = np.random.uniform(-3, 3, size=10)
    stds = np.random.uniform(1, 2, size=10)
    # instantiate the MAB
    game = Game(10, means, stds)
    # Plot the distribution
    game.play(5)

    if save_plot:
        visualize.violin_plot(game, save_path="initial_distributions.pdf")
    else:
        visualize.violin_plot(game, show=True)


def non_stationary_benchmark_slideshow(K=8, timepoints=15, output_folder="tmp_data_1"):

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder, ignore_errors=True)
    os.mkdir(output_folder)

    np.random.seed(42)

    means = np.zeros([timepoints, K], dtype=np.float)
    stds = np.zeros([timepoints, K], dtype=np.float)

    means[0, :] = np.random.uniform(-3, 3, size=K)
    stds[0, :] = np.random.uniform(1, 3, size=K)

    for row in range(1, timepoints):
        means[row, :] = means[row - 1, :] + 0.1 * np.random.choice([-1, 1], size=[1, K])
        stds[row, :] = stds[row - 1, :] + 0.1 * np.random.choice([-1, 1], size=[1, K], p=(.2, .8))

    game = Game(timepoints, means, stds)

    visualize.slideshow_violin_distributions(game, output_folder="tmp_data_1", y_axis_limit=(-15, 15), time_point_annotation=True)


def play_a_thousand_dollars_stationary_game():
    # Sampling mean and standard deviation
    np.random.seed(43)
    means = np.random.uniform(-3, 3, size=10)
    stds = np.abs(np.random.normal(3, 1, size=10))
    # initialising mab
    game = Game(1000, means, stds)

    means_hat, stds_hat, reward_per_arm, pulls_per_arm = game.play(
        initial_t_explorations=100, exploration_strategy="random"
    )
    visualize.violin_plot(game, arms_annotations=pulls_per_arm,  y_axis_limit=(-20, 20), show=True)


def play_a_thousand_dollars_non_stationary_game():
    game = Game(1000)

    means_hat, stds_hat, reward_per_arm, pulls_per_arm = game.play(
        initial_t_explorations=100, exploration_strategy="random"
    )
    visualize.violin_plot(game, arms_annotations=pulls_per_arm, y_axis_limit=(-20, 20), show=True)


def visualize_q_matrix(K=10):
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
        show=True
    )


# TODO
# def visualize_q_matrix_slideshow(K=10, timepoints=100, output_folder="tmp_data_2"):
#     if os.path.exists(output_folder):
#         shutil.rmtree(output_folder, ignore_errors=True)
#     os.mkdir(output_folder)
#
#     # to generate a non-stationary bandit, add a dimension to the means and standard deviation
#     np.random.seed(42)
#
#     means = np.zeros([timepoints, K], dtype=np.float)
#     stds = np.zeros([timepoints, K], dtype=np.float)
#
#     means[0, :] = np.random.uniform(-3, 3, size=K)
#     stds[0, :] = np.random.uniform(1, 3, size=K)
#
#     for row in range(1, timepoints):
#         means[row, :] = means[row - 1, :] + 0.1 * np.random.choice([-1, 1], size=[1, K])
#         stds[row, :] = stds[row - 1, :] + 0.1 * np.random.choice([-1, 1], size=[1, K], p=(.2, .8))
#
#     mab = MultiArmedBandit(means, stds)
#
#     player = Player(T=200, mab=mab)
#
#     frames_list = []
#     offset_before = 12
#     offset_after = 5
#     total_offset = offset_before + offset_after
#
#     for t in range(30):  # timepoints
#
#         if t != timepoints - 1:
#             # each draw will update the mean and std to the next row of the input matrix
#             k = np.random.choice(range(K))
#             player.select_arm(k)
#
#         fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
#
#         ax[0], im = player.get_evolving_grid_plot(
#             ax[0],
#             show_data_at_tp=player.mab.tp,
#             offset_before=np.min([offset_before, player.mab.tp]),
#             offset_after=np.max([offset_after, total_offset - player.mab.tp]),
#             last_tp_off_grid=True
#         )
#
#         ax[1] = player.mab.get_violin_plot_distributions(ax[1], y_axis_limit=(-20, 20), vertical=True)
#         plt.ioff()
#
#         fig.subplots_adjust(bottom=0.15, wspace=0.05)
#         plt.show()
#
#         pfi_frame = os.path.abspath(os.path.join(output_folder, 'step_{}.jpg'.format(t)))
#         plt.savefig(pfi_frame)
#         frames_list.append("file '" + pfi_frame + "'")
#
#
#     pfi_frames_list = os.path.abspath(os.path.join(output_folder, 'frames_list.txt'))
#
#     with open(pfi_frames_list, "w+") as outfile:
#         outfile.write("\n".join(frames_list))
#
#     pfi_output_gif = os.path.abspath(os.path.join(output_folder, 'sequence.gif'))
#     os.system(f"ffmpeg -r 3 -f concat -safe 0 -i {pfi_frames_list} -y {pfi_output_gif}")
#     print(f"gif created and stored in {pfi_output_gif}")


if __name__ == "__main__":
    # stationary_mab_distribution()
    # non_stationary_benchmark_slideshow()
    # play_a_thousand_dollars_stationary_game()
    # play_a_thousand_dollars_non_stationary_game()
    visualize_q_matrix()
    # visualize_q_matrix_slideshow()
