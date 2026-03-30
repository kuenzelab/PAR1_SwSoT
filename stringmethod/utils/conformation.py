import numpy as np
import os
import re
import sys
import glob
from dataclasses import dataclass
from typing import Optional
import logging

from logging import Logger

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(name)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger: Logger = logging.getLogger("stringmethod")

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
    
def _natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)
    
def compute_coordinates(iteration):
    first_iteration = iteration - 1
    last_iteration = sys.maxsize
    inter_out_dir = "confouts"
    cv_coordinates = None
    use_plumed = True
    for it in range (first_iteration, last_iteration + 1):
        logger.info("Extrakting CV coordinates of output conformation of iteration {}".format(it))
        iter_data = "{}/cv_confout{}.npy".format(inter_out_dir, it)
        if os.path.isfile(iter_data):
            coordinates = np.load(iter_data)
        else:
       	    if not use_plumed:
                iteration_md_dir = "md/{}/*/*/*xvg".format(it)
                #iteration_md_dir_swarms = "md/{}/*/s*/*xvg".format(it)
                #iteration_md_dir_restrained = "md/{}/*/restrained/*xvg".format(it)
            else:
                iteration_md_dir = "md/{}/*/*/colvar".format(it)
                #iteration_md_dir_swarms = "md/{}/*/s*/colvar".format(it)
                #iteration_md_dir_restrained = "md/{}/*/restrained/colvar".format(it)
            xvg_files = _natural_sort(glob.glob(iteration_md_dir))
            #xvg_files_swarms = self._natural_sort(glob.glob(iteration_md_dir_swarms))
            #xvg_files_restrained = self._natural_sort(glob.glob(iteration_md_dir_restrained))
            if len(xvg_files) == 0:
                logger.info(
                    "No output files found for iteration %s. Not looking further",
                    it,
                )
                return cv_coordinates
            coordinates = None
            for file_idx, xf in enumerate(xvg_files):
                data = load_xvg(file_name=xf)
                # only print last frame (output CV coordinates)
                data = data[[-1], 1:]
                # to do: add iteration, ad file index (restrained, which swarm)
                # in class die ausrechnet welches array am nächsten ist --> bekommt iteration( in dem ordner), kann dann selbst iteration_dir natural sort performen und anhand der liste confout finden und einsetzen)
                # für erste testläufe: print list of used confouts, überrüfen der coordinaten und string
                if coordinates is None:
                    n_cvs = data.shape[1]
                    coordinates = np.empty((len(xvg_files), 1, n_cvs + 2))
                # values[file_idx, :, :] = data
                # add iteration and file index as first two positions of the array
                # problem: file_idx und it immer neu für alle Zeilen insertet -> besser Array nach für nach aufzubauen, oder in file_idx position einbauen?
                value = np.insert(data, 0, file_idx, axis = 1) 
                values = np.insert(value, 0, it, axis = 1)
                coordinates[file_idx, :, :] = values
            if it < last_iteration:
                np.save(iter_data, coordinates)
        if cv_coordinates is None:
            cv_coordinates = coordinates
        else:
            cv_coordinates = np.append(cv_coordinates, coordinates, axis=0)
    return cv_coordinates
    
def closest_conformation(string_point, iteration):
    # string_point: cv_coordinates of the string point for that the closest conformation shall be found
    # second step: compare array with string_point, calculate distances, identify lowest distance, read out iteration, swarm or restrained, point idx, return these values
    dist_min = None
    confout_points = None
    cv_coordinates = compute_coordinates(iteration)
    use_plumed = True
    for compare_point_idx, compare_point in enumerate(cv_coordinates):
        compare = compare_point[[0],2:]
        #dist = string_point - compare
        #dist = abs(np.sum(dist))
        dist = np.linalg.norm(compare - string_point)
        #print(dist)
        # to do: distance normalisieren
        if dist_min is None:
            dist_min = dist
            compare_min = compare_point
            it_min = compare_point[[0],0]
            file_idx_min = compare_point[[0],1]
        if dist < dist_min:
            dist_min = dist
            #point_min = point
            compare_min = compare_point
            it_min = compare_point[[0],0]
            file_idx_min = compare_point[[0],1]
    it_min_int = int(it_min)
    #print(point_min, compare_min, dist_min)
    #print(confout_points)
    #return list of iteration, point_idx, swarm_idx
    if not use_plumed:
        iteration_md_dir = "md/{}/*/*/*xvg".format(it_min_int)
    else:
        iteration_md_dir = "md/{}/*/*/colvar".format(it_min_int)
    xvg_files = _natural_sort(glob.glob(iteration_md_dir))
    file_idx_min_int = int(file_idx_min)
    xvg_files_last = xvg_files[-1]
    logger.info("file_idx = {}, file_idx_int = {}, xvg_files last item: {}, it_min {}, compare_min: {}, iteration_dir {}".format(file_idx_min, file_idx_min_int, xvg_files_last, it_min, compare_min, iteration_md_dir))
    path = xvg_files[file_idx_min_int]
    if not use_plumed:
        path_confout = path.replace("pullx.xvg", "confout.gro")
    else:
        path_confout = path.replace("colvar", "confout.gro")
    logger.info("Confout found for point {} is {} with {} and path {}".format(string_point, path_confout, compare_min, path_confout))
    return path_confout
