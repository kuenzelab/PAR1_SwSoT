import numpy as np
import glob as glob
from math import ceil
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from MDAnalysis.analysis.align import AlignTraj
import MDAnalysis as mda
#import nglview as nv
import pickle
import logging
import sys
import os

logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("pyemma").setLevel(logging.NOTSET)

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

simulation_directory = "."
os.chdir(simulation_directory)
os.getcwd()

sys.path.append("../../")

from stringmethod.config import *
from stringmethod.postprocessing import *

def load_swarm_data(extract, first_iteration=201, last_iteration=None):
    #from stringmethod postprocessing cv_value_extraction
    if last_iteration == None:
        last_iteration = sys.maxsize
    if extract:
        config = load_config("config.json")

        ce = CvValueExtractor.from_config(
            config=config,
            first_iteration=201,  # Exclude the first iterations to let the system equilibrate.
            last_iteration=300,  # Usefull to make blocks of the simulation
        )
        ce.run()
        ce.persist()
    return np.load("postprocessing/cv_coordinates.npy")

cv_coordinates = load_swarm_data(extract=True, first_iteration=201, last_iteration=None)

my_cvs = [0]
cv_coordinates_clean = cv_coordinates[:, :, my_cvs]

def calculate_transition_matrix(
    cv_coordinates, n_grid_points=15, T=300, kB=0.0019872041
):
    config = load_config("config_22.json")
    tc = TransitionCountCalculator.from_config(
        config=config,
        # You probably want to play around with n_grid_points.
        # It sets the resolution. Its optimal value depends on your swarm trajectory length and sample size
        n_grid_points=n_grid_points,
        cv_coordinates=cv_coordinates,
    )
    tc.run()
    tc.persist()
    fc = FreeEnergyCalculator.from_config(
        config=config, grid=tc.grid, transition_count=tc.transition_count, T=T, kB=kB
    )
    fc.run()
    fc.persist()
    return tc.grid, fc.free_energy

grid, free_energy = calculate_transition_matrix(
    cv_coordinates_clean, n_grid_points=40, T=300, kB=0.0019872041
)

def show_fes(grid, free_energy, fe_cut_off=None, cv_labels=["cv0 (nm)", "cv1 (nm)"], f_min=None, f_max=None):
    if fe_cut_off == None:
        fe_cut_off = sys.maxsize
    free_energy[free_energy > fe_cut_off] = np.nan
    cv_0 = grid[:, 0]
    fig, ax = plt.subplots(1, 1)
    if free_energy.shape[1] == 1:
        ax.plot(cv_0, free_energy, "--o", linewidth=4, markersize=7)
        #ax.set_ylabel("Free Energy (kT)")
    else:
        cv_1 = grid[:, 1]
        im = plt.contourf(
            cv_0,
            cv_1,
            free_energy.T,
            levels=[0, 0.1, 0.2, 0.6, 1.0, 1.5, 2.2, 3.0, 4.0, 5.0],
            #levels=10,
            norm=mpl.colors.PowerNorm(gamma=1 / 3),
            # Used to be rainbow cmap=plt.cm.rainbow
            cmap=plt.cm.hot,
            #vmin=0,
            vmin=f_min,
            vmax=f_max,
            #vmax=5,
        )
        cbar = plt.colorbar(im)
        cbar.set_label("$\Delta G$ [kcal/mol]")
        ax.set_ylabel(cv_labels[1])
        #ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        #ax.yaxis.set_major_locator(MultipleLocator(0.1))
    #ax.set_xlabel(cv_labels[0])
    ax.set_ylim(-0.1, 4.1)
    ax.set_xlim(-75, 125)
    #ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    #ax.yaxis.set_major_locator(MultipleLocator(50), fontsize = 14)
    ax.tick_params(direction='out', length=7, width=3)
    ax.set_xticks([-50, 0, 50, 100], [-50, 0, 50, 100], fontsize = 25)
    ax.set_yticks([0, 2.0, 4.0], ["","",""])
    ax.spines['top'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    #ax.annotate('active', xy=(0.8, 3.0), xytext=(0.8, 3.0))
    #ax.annotate('inactive', xy=(1.3, 2.0), xytext=(1.3, 2.0))
    #av0 = np.load('mean_cv0.npy')
    #av1 = np.load('mean_cv1.npy')
    #ax.plot(av0, av1, color='grey', label='String mean')
    #ax.plot(1.1, 2.05, 'v', markersize=7, markeredgecolor='navy', label='Connector twist', color = 'blue')
    #ax.plot(1.15, 2.01, 'o', markersize=7, markeredgecolor='navy', label='first DPxxY twist', color = 'blue')
    #ax.plot(0.98, 2.5, 'd', markersize=7, markeredgecolor='navy', label='second DPxxY twist', color = 'blue')
    #ax.plot(1.0, 2.8, 'v', markersize=7, markeredgecolor='navy', label='TM6 displaced', color = 'blue')
    #ax.plot(1.1, 2.655, 's', markersize=7, markeredgecolor='navy', label='Y-Y interaction', color = 'blue')
    #ax.plot(1.2, 2.0, '>', markersize=7, markeredgecolor='navy', label='TM5 bulge', color = 'blue')
    #ax.legend()
    fig.tight_layout()
    return fig, ax

fig, ax = show_fes(
    grid, free_energy, fe_cut_off=None, cv_labels=["a100", "$\Delta G$ [kcal/mol]"]
)

fig.savefig("free_energy_apo_201_300_a100_fig2_neu2.svg", transparent=True)
