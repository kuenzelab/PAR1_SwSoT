import glob
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import stringmethod.simulations.mdtools as mdtools
from stringmethod import logger
from stringmethod.config import Config

from .base import AbstractPostprocessor


@dataclass
class ConfoutValueExtractor(AbstractPostprocessor):
    md_dir: Optional[str] = "md"
    """
    Loads all swarms' start and end CV coordinates and puts them in an array for further postprocessing
    """
    first_iteration: Optional[int] = 1
    last_iteration: Optional[int] = sys.maxsize
    """"
    First index corresponds to a trajectory.
    Second index sets the frame in that trajectory.
    Third index sets the CV values in that frame.
    """
    cv_coordinates: Optional[np.array] = None
    use_plumed: Optional[bool] = False
    inter_out_dir = 'cv_confout'

    def __post_init__(self):
        pass

    def _natural_sort(self, l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
        return sorted(l, key=alphanum_key)

    def _do_run(self) -> bool:
        self.cv_coordinates = self.compute_cv_coordinates()
        return True

    def compute_cv_coordinates(self) -> np.array:
        if not os.path.isdir(inter_out_dir):
            os.mkdir(inter_out_dir)
        cv_coordinates = None
        for it in range(self.first_iteration, self.last_iteration + 1):
            logger.info("Extrakting CV coordinates of output conformation of iteration {}".format(it))
            iter_data = "{}/cv_confout{}.npy".format(inter_out_dir, it)
            if os.path.isfile(iter_data):
                values = np.load(iter_data)
            else:
                if not self.use_plumed:
                    iteration_md_dir = "{}/{}/*/*/*xvg".format(self.md_dir, it)
                else:
                    iteration_md_dir = "{}/{}/*/*/colvar".format(self.md_dir, it)
                xvg_files = self._natural_sort(glob.glob(iteration_md_dir))
                if len(xvg_files) == 0:
                    logger.info(
                        "No output files found for iteration %s. Not looking further",
                        it,
                    )
                    return cv_coordinates
                values = None
                for file_idx, xf in enumerate(xvg_files):
                    data = mdtools.load_xvg(file_name=xf)
                    # Skip first column which contains the time and include only last frame
                    data = data[[-1], 1:]
                    # to do: add iteration, ad file index (restrained, which swarm)
                    # in class die ausrechnet welches array am nächsten ist --> bekommt iteration( in dem ordner), kann dann selbst iteration_dir natural sort performen und anhand der liste confout finden und einsetzen)
                    # für erste testläufe: print list of used confouts, überrüfen der coordinaten und string
                    if values is None:
                        n_cvs = data.shape[1]
                        values = np.empty((len(xvg_files), 1, n_cvs))
                    values[file_idx, :, :] = data
                    # add iteration and file index as first two positions of the array
                    values = np.insert(values, 0, file_idx, axis = 2) 
                    values = np.insert(values, 0, it, axis = 2)
                if it < self.last_iteration:
                    np.save(iter_data, values)
            if cv_coordinates is None:
                cv_coordinates = values
            else:
                cv_coordinates = np.append(cv_coordinates, values, axis=0)
        return cv_coordinates

    def _do_persist(self):
        np.save(
            "{}/cv_coordinates".format(inter_out_dir),
            self.cv_coordinates,
        )
