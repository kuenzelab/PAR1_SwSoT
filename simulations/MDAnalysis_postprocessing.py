import numpy as np
import MDAnalysis as mda
import argparse
import logging
import time
import os
import mdtraj as md
from MDAnalysis.analysis import distances
from MDAnalysis.analysis import dihedrals

seconds = time.time()
result = time.localtime(seconds)


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--iteration", help="iteration the features are calculated for", type=int)
parser.add_argument("--system", help="name of the simulation Folder", type=str)
parser.add_argument("--swarmnumber", help="number of swarms in iteration", type=int)
parser.add_argument("--swarmsize", help="size of swarms in iteration", type=int)
args = parser.parse_args()

def distance_calculation_atoms_mda(u, atom1, atom2):
    sel1 = u.select_atoms(atom1)
    sel2 = u.select_atoms(atom2)
    dist_arrays = None
    for ts in u.trajectory:
        dist_arr = distances.distance_array(sel1.positions, sel2.positions, box=u.dimensions)
        if dist_arrays is None:
            dist_arrays= dist_arr
        else:
            dist_arrays= np.append(dist_arrays,dist_arr, axis=0)
    return dist_arrays

def distance_calculation_atoms_mdtraj(traj, top, atom1, atom2):
    at_pairs = [top.select(atom1 + " or " + atom2)]
    dist_arrays = None
    dist = md.compute_distances(traj, at_pairs)
    return dist

def dihedral_calculation_dihedral_mdtraj(traj, top, residue):
    atom_indices = [top.select("resSeq {} and name N CA CB CG".format(residue))]
    angles = md.compute_dihedrals(traj, atom_indices)
    angles_deg = np.degrees(angles)
    return angles_deg

def dihedral_calculation_dihedral_mda(u, residue):
    sel = u.select_atoms('resid {}'.format(residue))
    chis = [res.chi1_selection() for res in sel.residues]
    dihs = dihedrals.Dihedral(chis).run()
    chi1_angles = dihs.results.angles
    return chi1_angles

distances = None
distances1 = None
for sw in range(1, args.swarmnumber + 1):
    for si in range(0, args.swarmsize):
        logging.info('Processing swarm {} from bead {}'.format(si, sw))
        #print(sw, si)
        top = md.load('{}/topology/start.pdb'.format(args.system)).topology
        t = md.load('{}/md/{}/{}/s{}/traj_comp.xtc'.format(args.system, args.iteration, sw, si), top='{}/topology/start.pdb'.format(args.system))
    	#functions to calculate
        if distances is None:
            distances = np.empty([args.swarmsize, 2, 50])
        #TM6 down
        distances[0] = distance_calculation_atoms_mdtraj(t, top, "resSeq 305 and name CA" , "resSeq 186 and name CA")
        #Asp256 contact to TYR353, TY# 350
        distances[1] = distance_calculation_atoms_mdtraj(t, top, "resSeq 256 and name OD1" , "resSeq 353 and name OH")
        distances[2] = distance_calculation_atoms_mdtraj(t, top, "resSeq 256 and name OD1" , "resSeq 350 and name OH")
        distances[3] = distance_calculation_atoms_mdtraj(t, top, "resSeq 256 and name OD1" , "resSeq 162 and name OH")
        #VAL257 some distance speciffiyng movement ot ECL2 --> LEU258 may be even better?
        distances[4] = distance_calculation_atoms_mdtraj(t, top, "resSeq 257 and name CB" , "resSeq 175 and name CA")
        #THR261 is locked with ASN259 in the inactive state
        distances[5] = distance_calculation_atoms_mdtraj(t, top, "resSeq 261 and name N" , "resSeq 259 and name OD1")
        #TYR270 --> movement TM5 towards TM4 (inward movement)
        distances[6] = distance_calculation_atoms_mdtraj(t, top, "protein and resSeq 270 and name OH" , "protein and resSeq 233 and name O")
        distances[7] = distance_calculation_atoms_mdtraj(t, top, "resSeq 270 and name OH" , "resSeq 266 and name OH")
        #PHE274 chi angle
        distances[8] = dihedral_calculation_dihedral_mdtraj(t, top, 274)
        #TYR290 along TM6 inside or outside --> may be visible in dihedral?
        distances[9] = dihedral_calculation_dihedral_mdtraj(t, top, 290)
        #CYS296 hobnd to SER299
        distances[10] = distance_calculation_atoms_mdtraj(t, top, "protein and resSeq 296 and name O" , "resSeq 299 and name OG")
        distances[11] = distance_calculation_atoms_mdtraj(t, top, "resSeq 296 and name SG" , "protein and resSeq 204 and name O")
        #SER315 hbond to Arg200 in the inactive state
        distances[12] = distance_calculation_atoms_mdtraj(t, top, "resSeq 315 and name OG" , "resSeq 200 and name NH2")
        #THR329 distance to SER360 (inactive)
        distances[13] = distance_calculation_atoms_mdtraj(t, top, "resSeq 329 and name OG1" , "resSeq 360 and name OG")
        #ASN330 active hbond to TYR187
        distances[14] = distance_calculation_atoms_mdtraj(t, top, "resSeq 330 and name OD1" , "resSeq 187 and name OH")
        #HIS336 different conformations, hard to interpret
        distances[15] = dihedral_calculation_dihedral_mdtraj(t, top, 336)
        #TYR337 interaction with ligand, VPX, not easy to find measure
        distances[16] = distance_calculation_atoms_mdtraj(t, top, "resSeq 337 and name OH" , "resSeq 158 and name CA")
        #SER338 small differences
        distances[17] = distance_calculation_atoms_mdtraj(t, top, "resSeq 338 and name OG" , "resSeq 272 and name OG")
        #GLU347
        distances[18] = distance_calculation_atoms_mdtraj(t, top, "resSeq 347 and name OE1" , "resSeq 95 and name OH")
        #TYR350 1.active (but also inactive from other site) 2.inactive
        distances[19] = distance_calculation_atoms_mdtraj(t, top, "resSeq 350 and name OH" , "resSeq 256 and name OD1")
        distances[20] = distance_calculation_atoms_mdtraj(t, top, "resSeq 350 and name OH" , "protein and resSeq 96 and name O")
        #PHE351 different confromations
        distances[21] = dihedral_calculation_dihedral_mdtraj(t, top, 351)
        #TYR351 three, 1. active
        distances[22] = distance_calculation_atoms_mdtraj(t, top, "resSeq 353 and name OH" , "resSeq 183 and name OH")
        distances[23] = distance_calculation_atoms_mdtraj(t, top, "resSeq 353 and name OH" , "resSeq 255 and name NE2")
        distances[24] = distance_calculation_atoms_mdtraj(t, top, "resSeq 353 and name OH" , "resSeq 256 and name OD1")
        #LEU355
        distances[25] = dihedral_calculation_dihedral_mdtraj(t, top, 355)
        #ASP367 two conf, one more to middle, one less
        distances[26] = distance_calculation_atoms_mdtraj(t, top, "resSeq 367 and name CG" , "resSeq 317 and name CA")
        #DPxxY
        distances[27] = dihedral_calculation_dihedral_mdtraj(t, top, 372)
        distances[28] = dihedral_calculation_dihedral_mdtraj(t, top, 373)
        #SER306
        distances[29] = distance_calculation_atoms_mdtraj(t, top, "resSeq 305 and name CA" , "resSeq 186 and name CA")
        distances[30] = distance_calculation_atoms_mdtraj(t, top, "resSeq 306 and name OG" , "resSeq 308 and name NZ")
        distances[31] = distance_calculation_atoms_mdtraj(t, top, "resSeq 306 and name OG" , "protein and resSeq 302 and name O")
        #TM6
        distances[32] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 312 and name CA',  'resSeq 200 and name CA')
        #YY
        distances[33] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 290 and name CZ', 'resSeq 371 and name CZ')
        #TM5 bulge
        distances[34] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 275 and name CA', 'resSeq 357 and name CA')
        #cvs
        distances[35] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 145 and name CA', 'resSeq 372 and name CA')
        distances[36] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 216 and name CA', 'resSeq 308 and name CA')
        distances[37] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 372 and name CA', 'resSeq 216 and name CA')
        distances[38] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 184 and name CA', 'resSeq 315 and name CA')
        distances[39] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 307 and name CA', 'resSeq 294 and name CA')
        #a100 dist
        distances[40] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 123 and name CA', 'resSeq 373 and name CA')
        distances[41] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 148 and name CA', 'resSeq 187 and name CA')
        distances[42] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 192 and name CA', 'resSeq 219 and name CA')
        distances[43] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 298 and name CA', 'resSeq 312 and name CA')
        distances[44] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 336 and name CA', 'resSeq 353 and name CA')
        distances[45] = - 14.43 * distances[46]*10 - 7.62 * distances[47]*10 + 9.11 * distances[48]*10 - 6.32 * distances[49]*10 - 5.22 * distances[50]*10 + 278.88
        #TM5 inward movement
        distances[46] = distance_calculation_atoms_mdtraj(t, top, "resSeq 275 and name CA" , "resSeq 357 and name CA")
        #KRK motif
        distances[47] = distance_calculation_atoms_mdtraj(t, top, "resSeq 200 and name CZ" , "resSeq 135 and name NZ")
        distances[48] = distance_calculation_atoms_mdtraj(t, top, "resSeq 200 and name CZ" , "resSeq 307 and name NZ")
        distances[49] = distance_calculation_atoms_mdtraj(t, top, "resSeq 307 and name NZ" , "resSeq 135 and name NZ")
    print(distances)
    if distances1 is None:
    	distances1 = distances
    else:
    	distances1 = np.append(distances1, distances, axis = 0)
    distances = None
    dist = None
    np.save('{}/postprocessing/PAR1_spec_it{}.npy'.format(args.system, args.iteration), distances1)