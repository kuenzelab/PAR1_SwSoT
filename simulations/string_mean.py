import numpy as np
import glob as glob
from math import ceil
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import pickle
import logging
import sys
import os

def colorbar(mappable, cmap, norm, label0, size=10):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    cbar.set_label(label0, size=size)
    return cbar

plt.rcParams["axes.facecolor"] = "#f9f9fb"
plt.rcParams["grid.color"] = "grey"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 1
plt.rcParams["axes.grid"] = False
plt.rcParams["lines.solid_capstyle"] = "round"

def natural_sort(l):
    """
    Takes as input a list l of strings and sorts it with natural order.
      Parameters
      ----------
      l: list of strings.
      Returns
      -------
      l sorted
    """
    from re import split

    assert isinstance(l, list), "l is not a list!"
    for i in l:
        assert isinstance(i, str), "List contains non-string elements."
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)

def resample_2d_axis0(arr, new_length):
    """
    Resample a 2D array along axis 0 to new_length rows.
    
    Parameters
    ----------
    arr : array-like, shape (n, m)
    new_length : int
    
    Returns
    -------
    np.ndarray of shape (new_length, m)
    """
    arr = np.asarray(arr)
    n, m = arr.shape

    if n == new_length:
        return arr.copy()

    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, new_length)

    resampled = np.empty((new_length, m))
    for j in range(m):
        resampled[:, j] = np.interp(x_new, x_old, arr[:, j])

    return resampled

#files = natural_sort(glob.glob("./strings_Gq/string*txt"))
#for index, file in enumerate(files):
#    data = np.loadtxt(file)
#    data_2 = resample_2d_axis0(data, 22)
#    np.savetxt("./strings_Gq/string_new_{}.txt".format(index), data_2)

files_new = natural_sort(glob.glob("./strings_Gq/string_new_*.txt"))
strings = np.array([np.loadtxt(file).T for file in files_new])


start_iteration = 1
n_average = 60

print(np.shape(strings))
n_plots = strings.shape[1]
n_strings = strings.shape[0]
#fig, ax = plt.subplots(ceil(n_plots / 2), 2, figsize=(20, 8 * ceil(n_plots / 2)))
fig, ax = plt.subplots(ceil(n_plots), 1, figsize=(10, 8 * ceil(n_plots)))
ax = ax.flatten()
cmap = plt.cm.viridis_r
n_colors = (n_strings - start_iteration) // n_average + 1
colors = cmap(np.linspace(0, 1, n_colors))  # yellow to blue
norm = mpl.colors.Normalize(vmin=start_iteration, vmax=n_strings - 1)

for i, a in enumerate(ax[:n_plots]):
    a.plot(strings[0, i, :], ls=":", marker=".", label="string0", color="r")
    for jj, j in enumerate(range(start_iteration, n_strings, n_average)):
        string = np.mean(strings[j : j + n_average, i, :], axis=0)
        a.plot(string, ls="-", marker="o", color=colors[jj])
    av = np.mean(strings[241:, i, :], axis=0)
    std = np.std(strings[241:, i, :], axis=0)
    a.fill_between(
        np.arange(len(av)),
        av + std,
        av - std,
        alpha=0.4,
        label=f"std(string{start_iteration}-{n_strings-1})",
    )
    #a.plot(
    #    av,
    #    ls="-",
    #    marker=".",
    #    color="k",
    #    label=f"mean(string{start_iteration}-{n_strings-1})",
    #)
    #a.set_ylabel(
    #    f"{list(ndx_groups.keys())[2*i]} - {list(ndx_groups.keys())[2*i+1]} (nm)",
    #    size=18,
    #    labelpad=16,
    #)
    #a.set_xlabel("bead number", size=15, labelpad=13)
    a.set_xlim(left=0, right=strings.shape[2] - 1)
    a.xaxis.set_minor_locator(MultipleLocator(5))
    a.xaxis.set_major_locator(MultipleLocator(10))
    a.yaxis.set_minor_locator(MultipleLocator(0.1))
    a.yaxis.set_major_locator(MultipleLocator(0.1))
    a.grid(which="minor")
    a.tick_params(axis="y", labelsize=20)
    a.tick_params(axis="x", labelsize=20)
    #a.set_title(f"cv{i}")
    a.spines['top'].set_linewidth(3)
    a.spines['bottom'].set_linewidth(3)
    a.spines['left'].set_linewidth(3)
    a.spines['right'].set_linewidth(3)
    a.set_xticks([0, 11, 22], [0.0, 0.25, 0.5], fontsize = 20)
    a.tick_params(direction='out', length=7, width=3, labelsize = 20)
    if i % 1 != 0:
        a.legend()
        cbar = colorbar(a, cmap, norm, "iteration number", 20)
#if n_plots % 2:
#    fig.delaxes(ax[-1])

fig.savefig(f"strings_mean_Gq_new.svg", transparent=True)