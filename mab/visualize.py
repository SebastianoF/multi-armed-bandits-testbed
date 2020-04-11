import os
import shutil

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def violin_plot_ax(
    ax,
    data,
    time_point=0,
    violin_axis_limit=None,
    time_point_annotation=False,
    arms_annotations=None,
    vertical=True,
    annotate_total_reward=False,
):
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

    def set_axis_style(vertical):

        if vertical:
            labels = [r"$K_{%d}$" % j for j in range(len(data))]
            ax.get_xaxis().set_tick_params(direction="out")
            ax.xaxis.set_ticks_position("bottom")
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            ax.set_xlim(0.25, len(labels) + 0.75)
            if violin_axis_limit:
                ax.set_ylim(*violin_axis_limit)
            # ax.set_xlabel('Sample name')
            xlim = ax.get_xlim()
            ax.plot((xlim[0], xlim[1]), (0, 0), color="r", lw=0.5, ls="--")
        else:
            labels = [r"$K_{%d}$" % j for j in range(len(data))]
            ax.get_yaxis().set_tick_params(direction="out")
            ax.yaxis.set_ticks_position("left")
            ax.set_yticks(np.arange(1, len(labels) + 1))
            ax.set_yticklabels(labels)
            ax.set_ylim(0.25, len(labels) + 0.75)
            if violin_axis_limit:
                ax.set_xlim(*violin_axis_limit)
            # ax.set_ylabel('Sample name')
            ylim = ax.get_ylim()
            ax.plot((0, 0), (ylim[0], ylim[1]), color="r", lw=0.5, ls="--")

        ax.set_axisbelow(True)
        ax.grid(True, lw=0.5)

    if time_point_annotation is True:
        ax.set_title(f"Probability distribution per arm, time-point {time_point}")
    else:
        ax.set_title("Probability distribution per arm")

    parts = ax.violinplot(
        data, showmeans=False, showmedians=False, showextrema=False, vert=vertical
    )

    for pc in parts["bodies"]:
        pc.set_facecolor("#add8e6")
        pc.set_edgecolor("black")
        pc.set_alpha(1)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
    whiskers = np.array(
        [
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)
        ]
    )
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    if vertical:
        ax.scatter(inds, medians, marker="o", color="white", s=30, zorder=3)
        ax.vlines(inds, quartile1, quartile3, color="k", linestyle="-", lw=5)
        ax.vlines(inds, whiskers_min, whiskers_max, color="k", linestyle="-", lw=1)
    else:
        ax.scatter(medians, inds, marker="o", color="white", s=30, zorder=3)
        ax.hlines(inds, quartile1, quartile3, color="k", linestyle="-", lw=5)
        ax.hlines(inds, whiskers_min, whiskers_max, color="k", linestyle="-", lw=1)
    set_axis_style(vertical)

    if arms_annotations is not None:
        if vertical:
            y_lims = ax.get_ylim()
            y_level_annotations = y_lims[1] - 0.1 * (
                y_lims[1] - y_lims[0]
            )  # 10% below line
            for aa_index, aa in enumerate(arms_annotations):
                ax.annotate(
                    "{}".format(aa),
                    xy=(aa_index + 1, y_level_annotations),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                )
        else:
            pass

    return ax


def evolutionary_grid_ax(
    ax,
    data,
    show_data_at_tp=50,
    offset_before=12,
    offset_after=5,
    last_tp_off_grid=False,
    aspect="equal",
):

    if last_tp_off_grid:
        delta = -1
    else:
        delta = 0

    cmap = matplotlib.cm.inferno
    cmap.set_bad(color="#DDDDDD")

    data = np.clip(data, 0, np.inf).T  # we want to visualise it horizontally

    num_arms = data.shape[0]

    window_data = np.nan * np.ones([num_arms, offset_before + offset_after])
    window_data[:, :offset_before] = data[
        :, show_data_at_tp - offset_before : show_data_at_tp
    ]

    im = ax.imshow(
        window_data,
        interpolation="nearest",
        cmap=cmap,
        aspect=aspect,
        vmin=0,
        vmax=np.max(np.nan_to_num(data)),
        origin="lower",
    )

    ax.set_xticks(np.arange(0, offset_before, 1))
    ax.set_xticklabels(
        np.arange(show_data_at_tp - offset_before + 1, show_data_at_tp + 1, 1)
    )

    ax.set_yticks(np.arange(0, num_arms, 1))
    ax.set_yticklabels([r"$K_{%d}$" % j for j in range(num_arms)])

    ax.set_xticks(np.arange(-0.5, offset_before + delta, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_arms, 1), minor=True)

    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)

    ax.set_title("Rewards matrix")

    return ax, im


def violin_plot(
    game,
    show=False,
    save_path=None,
    num_samples_per_violin=1000,
    violin_axis_limit=None,
    time_point_annotation=False,
    arms_annotations=None,
    vertical=True,
    figsize=(9, 4),
):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    violin_plot_ax(
        ax,
        game.sample_all_arms(num_samples=num_samples_per_violin, time_point=game.tp),
        game.tp,
        violin_axis_limit,
        time_point_annotation,
        arms_annotations,
        vertical,
    )
    fig.subplots_adjust(bottom=0.15, wspace=0.05)

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()


def slideshow_violin_distributions(
    game,
    output_folder,
    num_samples_per_violin=1000,
    violin_axis_limit=None,
    time_point_annotation=False,
    arms_annotations=None,
    vertical=True,
):

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder, ignore_errors=True)
    os.mkdir(output_folder)

    frames_list = []

    for t in tqdm(range(game.T)):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
        violin_plot_ax(
            ax,
            game.sample_all_arms(num_samples=num_samples_per_violin, time_point=t),
            t,
            violin_axis_limit,
            time_point_annotation,
            arms_annotations,
            vertical,
        )
        plt.ioff()
        pfi_frame = os.path.abspath(
            os.path.join(output_folder, "step_{}.jpg".format(t))
        )
        fig.subplots_adjust(bottom=0.15, wspace=0.05)

        plt.savefig(pfi_frame)
        frames_list.append("file '" + pfi_frame + "'")

        plt.close()

    pfi_frames_list = os.path.abspath(os.path.join(output_folder, "frames_list.txt"))

    with open(pfi_frames_list, "w+") as outfile:
        outfile.write("\n".join(frames_list))

    pfi_output_gif = os.path.abspath(os.path.join(output_folder, "sequence.gif"))
    os.system(f"ffmpeg -r 3 -f concat -safe 0 -i {pfi_frames_list} -y {pfi_output_gif}")
    print(f"gif created and stored in {pfi_output_gif}")


def get_evolving_grid(
    game,
    show_data_at_tp=54,
    offset_before=12,
    offset_after=5,
    last_tp_off_grid=True,
    save_path=None,
    show=False,
):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

    evolutionary_grid_ax(
        ax,
        game.q,
        show_data_at_tp=show_data_at_tp,
        offset_before=offset_before,
        offset_after=offset_after,
        last_tp_off_grid=last_tp_off_grid,
    )
    fig.subplots_adjust(bottom=0.15, wspace=0.05)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()


def get_grid_and_violins_dynamic(game, output_folder, violin_axis_limit=(-20, 20)):

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder, ignore_errors=True)
    os.mkdir(output_folder)

    frames_list = []
    offset_before = 12
    offset_after = 5
    total_offset = offset_before + offset_after

    for t in tqdm(range(game.T)):
        fig = plt.figure(figsize=(12, 5.7))

        epsilon_x = 0
        epsilon_y = 0.02

        ax0 = fig.add_axes([0.05 + epsilon_x, 0.08 + epsilon_y, 0.6, 0.8])
        ax1 = fig.add_axes([0.7 + epsilon_x, 0.05 + epsilon_y, 0.25, 0.85])
        ax2 = fig.add_axes([0.61 + epsilon_x, 0.652 + epsilon_y, 0.01, 0.2])

        ax0, im = evolutionary_grid_ax(
            ax0,
            game.q,
            show_data_at_tp=t,
            offset_before=np.min([offset_before, t]),
            offset_after=np.max([offset_after, total_offset - t]),
            last_tp_off_grid=True,
            aspect="auto",
        )

        ax1 = violin_plot_ax(
            ax1,
            game.sample_all_arms(time_point=t),
            time_point=t,
            violin_axis_limit=violin_axis_limit,
            vertical=False,
        )

        fig.colorbar(im, cax=ax2)

        plt.ioff()

        pfi_frame = os.path.abspath(
            os.path.join(output_folder, "step_{}.jpg".format(t))
        )
        plt.savefig(pfi_frame)
        frames_list.append("file '" + pfi_frame + "'")

        plt.close()

    pfi_frames_list = os.path.abspath(os.path.join(output_folder, "frames_list.txt"))

    with open(pfi_frames_list, "w+") as outfile:
        outfile.write("\n".join(frames_list))

    pfi_output_gif = os.path.abspath(os.path.join(output_folder, "sequence.gif"))
    os.system(f"ffmpeg -r 3 -f concat -safe 0 -i {pfi_frames_list} -y {pfi_output_gif}")
    print(f"gif created and stored in {pfi_output_gif}")
