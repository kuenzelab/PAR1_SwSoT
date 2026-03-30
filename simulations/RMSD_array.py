import numpy as np
import logging
from subprocess import PIPE, run
from typing import List
import os
import shutil
import sys
from glob import glob
import argparse

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--siteration", help="start iteration the features are calculated for", type=int)
parser.add_argument("--literation", help="last iteration the features are calculated for", type=int)
parser.add_argument("--system", help="name of the simulation Folder", type=str)
parser.add_argument("--swarmnumber", help="number of swarms in iteration", type=int)
parser.add_argument("--swarmsize", help="size of swarms in iteration", type=int)
args = parser.parse_args()

def comp_rmsd_v_plumed(first_iteration, last_iteration, swarm_size, number_moving_beads):
    rmsd = None
    rmsd1 = None
    #logging.info('Current working directory: {}'.format(os.getcwd))
    for it in range(first_iteration, last_iteration + 1):
        for sw in range(1, number_moving_beads + 1):
            logging.info('Processing swarm {} from iteration {}'.format(sw, it))
            for si in range(0, swarm_size):
                wdir = '/simulations/{}/md/{}/{}/s{}/'.format(args.system, it, sw, si)
                os.chdir(wdir)
                #let plumed calculate rmsds: npxxy first, then connector to inactive, connector to active
                command = f"plumed driver --plumed /simulations/{}/rmsd/plumed_rmsd_holo.dat --ixtc traj_comp.xtc --pdb /work/ks40pymo-PAR1_apo/ks40pymo-PAR1_apo-1758331202/simulations/PAR1_apo/rmsd/start.pdb"
                logging.info(f"Running command {command}")
                result = run(
                    command,
                    stdout=PIPE,
                    stderr=PIPE,
                    shell=True,
                )
                #error = result.stdout
                #logging.info(f"info plumed: {error}")
                if rmsd is None:
                    rmsd = np.empty([swarm_size, 2, 3])
                xvg_file = load_xvg(file_name = 'colvar_rmsd')
                rmsd[si, :, :] = xvg_file[:, 1:]
            rmsd_con = rmsd[:, :, 1] - rmsd[:, :, 2]
            if rmsd1 is None:
                rmsd1 = rmsd
                rmsd_con1 = rmsd_con
            else:
                rmsd1 = np.append(rmsd1, rmsd, axis=0)
                rmsd_con1 = np.append(rmsd_con1, rmsd_con, axis=0)
            #print(distances)
            rmsd = None
            rmsd_con = None
        np.save('/simulations/{}/postprocessing/rmsds{}.npy'.format(args.system, it), rmsd1)
        np.save('/simulations/{}/postprocessing/rmsd_con{}.npy'.format(args.system, it), rmsd_con1)
        rmsd1 = None
        rmsd_con1 = None

def load_xvg(file_name: str, usemask: bool = False) -> np.array:
    """
    Originally from https://github.com/vivecalindahl/awh-use-case/blob/master/scripts/analysis/read_write.py
    if file does not exist, exit
    if exists, check number of commentlines to skip
    extract data and return
    :param file_name:
    :param usemask:
    :return:
    """

    if not os.path.exists(file_name):
        raise FileNotFoundError("WARNING: file " + file_name + " not found.")

    # Since xvg/colvar files can have both @ and # as a head, we only read lines that don't start with these chars
    data_lines = [line for line in open(file_name) if not line.startswith(("#", "@"))]
    # then convert them to lists of floats
    data_lines_num = [[float(x) for x in line.split()] for line in data_lines]
    # and remove lines that have inconsistent number of fields (e.g. due to write errors during restarting).
    data = [line for line in data_lines_num if len(line) == len(data_lines_num[0])]
    if len(data) == 0:
        raise IOError("No data found in file " + file_name)
    return np.array(data)

array = int(os.environ["SLURM_ARRAY_TASK_ID"])
comp_rmsd_v_plumed(args.siteration, args.siteration, args.swarmsize, args.swarmnumber)
