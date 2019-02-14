import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


root_path = "/home/pranathi/pipeline_skeleton_results/"
filename = "sfn_results_nov8.json"

sns.__doc__  # importing sns adds functionality to matplotlib.

palette = ["#016263",  # dark teal
           "#70CBCE",  # light teal
           "#EFC199",  # light orange
           "#DB7527",  # dark orange
           ]
sns.set_palette(palette)
sns.set_style(
    "whitegrid",
    {
        'ytick.color': '0.15',
        'ytick.direction': u'out',
        'ytick.major.size': 10,
        'ytick.minor.size': 10,
        'text.color': '.15',
        'axes.linewidth': 2.0,
        'axes.edgecolor': '.15',
        'grid.color': '1',
    }
)
sns.set_context("poster", font_scale=1.2)

plt.ion()


def fraction_to_percent(list_var):
    return [100 * var for var in list_var]


def show_bar_value(ax, rects, offset_factor=.3):
    """
    Attach a text label above each bar displaying its height
    rects = list of matplotlib containers, each with a list of rect handles
    offset_factor = factor of bar width to vertically offset the text, 1=barwidth
    """
    ylim = ax.get_ylim()
    for rect in rects:
        height = rect.get_height()
        # this offset isnt scaling with ylim, and it should.
        voffset = rect.get_width() * offset_factor
        v_position = rect.get_height() + voffset
        if height < ylim[0] or height > ylim[1]:
            continue
        ax.text(
            rect.get_x() + rect.get_width() / 2.,
            v_position,
            '{:.2f}'.format(height),
            ha='center', va='bottom',
            fontsize=10)


def base_bar_chart(ax, xindex, values, barwidth, palette=palette):
    """
    Create basic bar chart with option for bar labels
    """
    rects = []  # holds list of rect collections
    # create new axis if we aren't passed one
    for i, value in enumerate(values):
        rects.append(ax.bar(xindex + i * barwidth, value, barwidth, color=palette[i]))
    return rects


def grouped_bar_chart(
        comparison_lists,
        y_label,
        category_labels,
        legend,
        title,
        ax=None,
        factor_xticks=1,
        barwidth=0.1,
        show_values=False,
        palette=palette,
        rotate=0):
    """
    Plot bar values on a matplotlib axis
    `comparison_lists` is a list of lists, with the outer list matching the number of categories,
    and the inner lists across the groups (given in legend)
    NOTE: this is a lot easier to set up the base graph with pandas
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    xindex = np.arange(len(comparison_lists[0]))  # the x locations for the groups
    rects = base_bar_chart(
        ax, xindex, comparison_lists,
        barwidth=barwidth, palette=palette)
    if show_values:
        for data_rects in rects:
            show_bar_value(ax, data_rects)
    # add text for labels, title and axes ticks
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(xindex + factor_xticks * barwidth)
    ax.set_xticklabels(category_labels, rotation=rotate, ha='center', fontsize='x-small')
    ax.legend(rects, legend)
    fig.tight_layout()

    sns.despine()


def grouped_bar_break_y(
        comparison_lists,
        y_label,
        category_labels,
        legend,
        title,
        break_ylim,
        height_ratios=(1, 3),
        factor_xticks=1,
        barwidth=0.1,
        show_values=False,
        palette=palette,
        rotate=0):
    """
    Create a bar chart with outlier data separated with a broken y axis
    Passes forward any kwargs from `grouped_bar_chart`
    break_ylim is a 2-tuple containing the values to break the axes between
    height_ratios are the relative height ratios of the two axes
    """
    # create two new axes, link their x axis
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={'height_ratios': height_ratios})
    # plot the data on both axes

    xindex = np.arange(len(comparison_lists[0]))  # the x locations for the groups
    rects = base_bar_chart(
        ax_top, xindex, comparison_lists,
        barwidth=barwidth, palette=palette)
    rects = base_bar_chart(
        ax_bot, xindex, comparison_lists,
        barwidth=barwidth, palette=palette)
    fig.tight_layout()

    # add text for labels, title and axes ticks
    ax_top.set_title(title)
    ax_bot.set_ylabel(y_label)
    ax_bot.set_xticks(xindex + factor_xticks * barwidth)
    ax_bot.set_xticklabels(category_labels, rotation=rotate, ha='center', fontsize='xx-small')
    ax_top.legend(rects, legend)

    sns.despine()

    # zoom-in / limit the view to different portions of the data
    ylim_top = ax_top.get_ylim()
    ylim_bot = ax_bot.get_ylim()
    ax_top.set_ylim(break_ylim[1], ylim_top[1])  # outliers only
    ax_bot.set_ylim(ylim_bot[0], break_ylim[0])  # most of the data

    # set the graph formatting
    # hide the spines between ax_top and ax_bot
    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)  # redundant with despine, leave in for clarity
    # ax_top.xaxis.tick_top()
    # ax_top.tick_params(labeltop='off')  # don't put tick labels at the top
    # ax_bot.xaxis.tick_bottom()

    # NOTE: we only plot the break on the left, since we are despining
    # add the diagonal lines for the cut
    d = .015  # how big to make the diagonal lines in axes coordinates
    # get the unit axis to keep points consistent, regardless of xlim/ylim
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot(
        (-d, +d),
        (-d / height_ratios[0], +d / height_ratios[0]),
        **kwargs)  # top-left diagonal
    # ax_top.plot(
    #     (1 - d, 1 + d), (-d / height_ratios[0], +d / height_ratios[0]),
    #     **kwargs)  # top-right diagonal

    kwargs.update(transform=ax_bot.transAxes)  # switch to the bottom axes
    ax_bot.plot(
        (-d, +d),
        (1 - d / height_ratios[1], 1 + d / height_ratios[1]),
        **kwargs)  # bottom-left diagonal
    # ax_bot.plot(
    #     (1 - d, 1 + d), (1 - d / height_ratios[1], 1 + d / height_ratios[1]),
    #     **kwargs)  # bottom-right diagonal

    if show_values:
        for data_rects in rects:
            show_bar_value(ax_top, data_rects)
        for data_rects in rects:
            show_bar_value(ax_bot, data_rects)


def inserts(l, vals, idxs):
    for idx, val in zip(idxs, vals):
        l.insert(idx, val)
    return l


# # ######################
fpr_palagyi = [0.0001, 0.0001, 0.0001, 0.049, 0.0466, 0.05849, 0.0556, 0.1207]
fnr_palagyi = [0, 0, 0, 0.049, 0.0532, 0.05737, 0.0619, 0.0656]
fpr_lee = [1, 1, 1, 0, 0.0418, 0.04668, 0.0393, 0.0735]
fnr_lee = [1, 1, 1, 0.012, 0.0548, 0.0816, 0.0514, 0.05906]

idxs = [5, 7, 9, 11]
new_vals_fpr_pr = [0.0618333, 0.0762518, 0.0697375, 0.0924684]
new_vals_fnr_pr = [0.0845346, 0.0947862, 0.0919783, 0.102528]
new_vals_fpr_py = [0.0478111, 0.052452, 0.0487203, 0.0441275]
new_vals_fnr_py = [0.0472635, 0.0535765, 0.0428091, 0.0501252]

percent_fpr_pr = inserts(fpr_palagyi, new_vals_fpr_pr, idxs)
percent_fnr_pr = inserts(fnr_palagyi, new_vals_fnr_pr, idxs)
percent_fpr_py = inserts(fpr_lee, new_vals_fpr_py, idxs)
percent_fnr_py = inserts(fnr_lee, new_vals_fnr_py, idxs)

# set palette for NetMets
palette = [palette[0], palette[3]]

# Vectorization FPR & FNR comparison
labels = [
    'Cyl X',
    'Cyl Y',
    'Cyl Z',
    'Cyl XYZ',
    'Phantom 1',
    'Phantom 1\n+ decimation',
    'Phantom 1\n+ noise',
    'Phantom 1\n+ noise +\ndecimation',
    'Phantom 2',
    'Phantom 2\n+ decimation',
    'Phantom 2\n+ noise',
    'Phantom 2\n+ noise +\ndecimation'
]
legend = ["Palagyi", "Lee"]
# grouped_bar_chart(
#     [fpr_palagyi, fpr_lee], "False Positive Rate", labels, legend,
#     'Netmets: False Postive Rate', barwidth=0.30,
#     palette=palette, rotate=0, show_values=True)
# grouped_bar_chart(
#     [fnr_palagyi, fnr_lee], "False Negative Rate", labels, legend,
#     'Netmets: False Negative Rate', barwidth=0.30,
#     palette=palette, rotate=0, show_values=True)
grouped_bar_break_y(
    [percent_fpr_pr, percent_fpr_py], "False Positive Rate", labels, legend,
    'Netmets: False Postive Rate',
    break_ylim=(2, 90),
    barwidth=0.3,
    palette=palette, rotate=0, show_values=True)

grouped_bar_break_y(
    [percent_fnr_pr, percent_fnr_py], "False Negative Rate", labels, legend,
    'Netmets: False Negative Rate', barwidth=0.3,
    break_ylim=(2, 90),
    palette=palette, rotate=0, show_values=True)

# #####
# volume segmentation comparison
labels = (
    'Phantom 1 + noise',
    'Phantom 2 + noise',
    'India ink brain'
)

# F1 scores (should always be a fraction)
otsu = [0.1, 0.0855, 0.1443]
yen = [0.866, 0.875, 0.917]
watershed = [0.9566, 0.967, 0.9016]
adaptive_threshold = [0.9778, 0.9846, 0.8613]
legend = ('Watershed', 'Adaptive', 'Otsu', 'Yen')
title = 'Segmentation validation: F1 scores'

grouped_bar_chart(
    [watershed, adaptive_threshold, otsu, yen],
    "F1 score", labels, legend, title, factor_xticks=2,
    show_values=True, palette=palette)

# mean surface distance (in voxels?)

watershed = [0.6013, 0.5122, 0.1269]
adaptive_threshold = [0.3532, 0.309, 0.0507]
otsu = [43.77, 58.35, 19.298]
yen = [11.46, 19.3, 0.18]
ind = np.arange(len(otsu))  # the x locations for the groups
title = 'Segmentation validation: Mean surface distance'

grouped_bar_chart(
    [watershed, adaptive_threshold, otsu, yen],
    "Mean Surface Distance (voxels)", labels, legend, title, factor_xticks=2,
    show_values=True, palette=palette)

# show all the new figures
plt.show()

# ############
# save figures
save_path = os.path.join(root_path, 'graphs', "validationmetrics")
os.makedirs(save_path, exist_ok=True)

save_uri = "file://" + save_path

for i in plt.get_fignums():
    plt.figure(i)
    plt.savefig(os.path.join(save_path, 'figure%d.png' % i))
