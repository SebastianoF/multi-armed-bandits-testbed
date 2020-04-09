import os
import shutil

import numpy as np
from matplotlib import pyplot as plt

from mab.multi_armed_bandit import MultiArmedBandit
from mab.player import Player
from mab.strategies import epsilon_greedy


def stationary_mab_distribution(save_result=False):
    # Sampling mean and standard deviation for a stationary MAB with 10 arms
    np.random.seed(42)
    means = np.random.uniform(-3, 3, size=10)
    stds = np.abs(np.random.normal(3, 1, size=10))
    # instantiate the MAB
    mab = MultiArmedBandit(means, stds)
    # Plot the distribution
    mab.get_plot_distributions()
    if save_result:
        plt.savefig("initial_distributions.pdf")
    else:
        plt.show()


def non_stationary_benchmark_slideshow(K=8, timepoints=15, output_folder="tmp_data"):

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder, ignore_errors=True)
    os.mkdir(output_folder)

    # to generate a non-stationary bandit, add a dimension to the means and standard deviation
    np.random.seed(42)

    means = np.zeros([timepoints, K], dtype=np.float)
    stds = np.zeros([timepoints, K], dtype=np.float)

    means[0, :] = np.random.uniform(-3, 3, size=K)
    stds[0, :] = np.random.uniform(1, 3, size=K)

    for row in range(1, timepoints):
        means[row, :] = means[row - 1, :] + 0.1 * np.random.choice([-1, 1], size=[1, K])
        stds[row, :] = stds[row - 1, :] + 0.1 * np.random.choice([-1, 1], size=[1, K], p=(.2, .8))

    mab = MultiArmedBandit(means, stds)

    frames_list = []

    for t in range(timepoints):
        mab.get_plot_distributions(y_axis_limit=(-20, 20))
        plt.ioff()
        pfi_frame = os.path.abspath(os.path.join(output_folder, 'step_{}.jpg'.format(t)))
        plt.savefig(pfi_frame)
        frames_list.append("file '" + pfi_frame + "'")
        if t != timepoints - 1:
            # each draw will update the mean and std to the next row of the input matrix
            mab.draw_from_arm(np.random.choice(range(K)))

    pfi_frames_list = os.path.abspath(os.path.join(output_folder, 'frames_list.txt'))

    with open(pfi_frames_list, "w+") as outfile:
        outfile.write("\n".join(frames_list))

    pfi_output_gif = os.path.abspath(os.path.join(output_folder, 'sequence.gif'))
    os.system(f"ffmpeg -r 3 -f concat -safe 0 -i {pfi_frames_list} -y {pfi_output_gif}")
    print(f"gif created and stored in {pfi_output_gif}")


def play_a_thousand_dollars_stationary_game():
    # Sampling mean and standard deviation
    np.random.seed(43)
    means = np.random.uniform(-3, 3, size=10)
    stds = np.abs(np.random.normal(3, 1, size=10))
    # initialising mab
    mab = MultiArmedBandit(means, stds)
    player = Player(T=1000, mab=mab)
    means_hat, stds_hat, reward_per_arm, pulls_per_arm = epsilon_greedy(
        player, initial_t_explorations=100, exploration_strategy="random"
    )
    mab.get_plot_distributions(y_axis_limit=(-20, 20), arms_annotations=pulls_per_arm)
    plt.show()


def play_a_thousand_dollars_non_stationary_game():
    mab = MultiArmedBandit()
    player = Player(T=1000, mab=mab)
    means_hat, stds_hat, reward_per_arm, pulls_per_arm = epsilon_greedy(
        player, initial_t_explorations=100, exploration_strategy="random"
    )
    mab.get_plot_distributions(y_axis_limit=(-20, 20), arms_annotations=pulls_per_arm)
    plt.show()


def visualize_q_graph():
    np.random.seed(42)
    means = np.random.uniform(-3, 3, size=10)
    stds = np.abs(np.random.normal(3, 1, size=10))

    mab = MultiArmedBandit(means, stds)
    player = Player(T=1000, mab=mab)
    means_hat, stds_hat, reward_per_arm, pulls_per_arm = epsilon_greedy(
        player, initial_t_explorations=100, exploration_strategy="random"
    )
    player.q


if __name__ == "__main__":
    # stationary_mab_distribution()
    # non_stationary_benchmark_slideshow()
    # play_a_thousand_dollars_stationary_game()
    # play_a_thousand_dollars_non_stationary_game()

    import matplotlib

    np.random.seed(42)
    means = np.random.uniform(-3, 3, size=10)
    stds = np.abs(np.random.normal(3, 1, size=10))

    mab = MultiArmedBandit(means, stds)
    player = Player(T=400, mab=mab)
    means_hat, stds_hat, reward_per_arm, pulls_per_arm = epsilon_greedy(
        player, initial_t_explorations=100, exploration_strategy="random"
    )

    data = player.q
    show_data_at_tp = 50
    offset_before = 12
    offset_after = 5
    show_plus_one = True

    if show_plus_one:
        delta = 1
    else:
        delta = 0

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    cmap = matplotlib.cm.inferno
    cmap.set_bad(color='#DDDDDD')

    num_arms = data.shape[0]

    window_data = np.nan * np.ones([num_arms, offset_before + offset_after])
    window_data[:, :offset_before+delta] = data[:, show_data_at_tp-offset_before:show_data_at_tp+delta]

    im = ax.imshow(
        window_data, interpolation='nearest', cmap=cmap, aspect='equal', vmin=-4, vmax=4, origin='upper'
    )

    ax.set_xticks(np.arange(0, offset_before, 1))
    ax.set_xticklabels(np.arange(show_data_at_tp - offset_before + 1, show_data_at_tp + 1, 1))

    ax.set_yticks(np.arange(0, num_arms, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, offset_before, 1), minor=True)
    ax.set_yticks(np.arange(-.5, num_arms, 1), minor=True)

    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    plt.show()
