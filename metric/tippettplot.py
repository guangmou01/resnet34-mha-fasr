# PATH: metric/tippettplot.py

import numpy as np
import matplotlib.pyplot as plt


def tippett_plot(ss_llr, ds_llr,
                 x_range=(-5, 5),
                 y_range=(0, 1),
                 line_type="-",
                 fontsize=18):

    ss_llr = np.asarray(ss_llr, dtype=float)
    ds_llr = np.asarray(ds_llr, dtype=float)

    # ln(LR) -> log10(LR)
    ss_log10lr = ss_llr / np.log(10)
    ds_log10lr = ds_llr / np.log(10)

    ss_sorted = np.sort(ss_log10lr)
    ss_cumulative = np.arange(1, len(ss_sorted) + 1) / len(ss_sorted)

    ds_sorted = np.sort(ds_log10lr)[::-1]
    ds_cumulative = np.arange(1, len(ds_sorted) + 1) / len(ds_sorted)

    fig_tippett, ax = plt.subplots(figsize=(8, 6))

    ax.plot(ds_sorted, ds_cumulative,
            color="blue",
            linestyle=line_type)

    ax.plot(ss_sorted, ss_cumulative,
            color="red",
            linestyle=line_type)

    # Zero line: log10(LR) = 0
    ax.axvline(0, color="black", linestyle="--")

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    ax.set_xlabel(r"$\log_{10}(\Lambda)$", fontsize=fontsize)
    ax.set_ylabel("cumulative proportion", fontsize=fontsize)

    # y-axis ticks every 0.1
    ax.set_yticks(np.arange(y_range[0], y_range[1] + 0.001, 0.1))

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=fontsize - 2
    )

    ax.grid(True, alpha=0.4)

    return fig_tippett