import numpy as np
from matplotlib import pyplot as plt


def violin_plot(data):
    # https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.violinplot.html
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

    def set_axis_style():
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Sample name')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    ax.set_title('Probability distribution per arm')
    parts = ax.violinplot(
        data,
        showmeans=False,
        showmedians=False,
        showextrema=False
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

    labels = [r'$K_{%d}$' % j for j in range(1, len(data) + 1)]

    set_axis_style()

    fig.subplots_adjust(bottom=0.15, wspace=0.05)
    ax.grid(True)
    return plt
