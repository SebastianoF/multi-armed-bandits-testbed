import numpy as np
import matplotlib


def violin_plot(ax, data, time_point=0, y_axis_limit=None, time_point_annotation=False, arms_annotations=None, vertical=True):

    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

    def set_axis_style():
        labels = [r'$K_{%d}$' % j for j in range(len(data))]
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        if y_axis_limit:
            ax.set_ylim(*y_axis_limit)
        ax.set_xlabel('Sample name')
        ax.set_axisbelow(True)
        ax.grid(True, lw=0.5)

    if time_point_annotation is True:
        ax.set_title(f'Probability distribution per arm, time-point {time_point}')
    else:
        ax.set_title('Probability distribution per arm')

    parts = ax.violinplot(
        data,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        vert=vertical
    )

    for pc in parts['bodies']:
        pc.set_facecolor('#add8e6')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data, quartile1, quartile3)
    ])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
    set_axis_style()

    if arms_annotations is not None:
        y_lims = ax.get_ylim()
        y_level_annotations = y_lims[1] - 0.1 * (y_lims[1] - y_lims[0])  # 10% below line
        for aa_index, aa in enumerate(arms_annotations):
            ax.annotate("{}".format(aa),
                        xy=(aa_index + 1, y_level_annotations),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='center')

    return ax


def evolutionary_grid(
        ax,
        data,
        show_data_at_tp=50,
        offset_before=12,
        offset_after=5,
        last_tp_off_grid=False
    ):

    if last_tp_off_grid:
        delta = -1
    else:
        delta = 0

    cmap = matplotlib.cm.inferno
    cmap.set_bad(color='#DDDDDD')

    num_arms = data.shape[0]

    window_data = np.nan * np.ones([num_arms, offset_before + offset_after])
    window_data[:, :offset_before] = data[:, show_data_at_tp-offset_before:show_data_at_tp]

    im = ax.imshow(
        window_data,
        interpolation='nearest',
        cmap=cmap,
        aspect='equal',
        vmin=-4,
        vmax=4,
        origin='upper'
    )

    ax.set_xticks(np.arange(0, offset_before, 1))
    ax.set_xticklabels(np.arange(show_data_at_tp - offset_before + 1, show_data_at_tp + 1, 1))

    ax.set_yticks(np.arange(0, num_arms, 1))

    ax.set_xticks(np.arange(-.5, offset_before + delta, 1), minor=True)
    ax.set_yticks(np.arange(-.5, num_arms, 1), minor=True)

    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    return ax, im
