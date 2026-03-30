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
parser.add_argument("--pdb", help="path to the pdb file", type=str)
parser.add_argument('--name', help='name of system', type=str)
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
    #print(atom1 + " or " + atom2)
    at_pairs = [top.select(atom1 + " or " + atom2)]
    dist_arrays = None
    dist = md.compute_distances(traj, at_pairs)
    return dist

def dihedral_calculation_dihedral_mdtraj(traj, top, residue):
    #print("resSeq {} and name N CA CB CG".format(residue))
    atom_indices = [top.select("resSeq {} and name N CA CB CG".format(residue))]
    #print(atom_indices)
    angles = md.compute_dihedrals(traj, atom_indices)
    angles_deg = np.degrees(angles)
    return angles_deg

def dihedral_calculation_dihedral_mda(u, residue):
    sel = u.select_atoms('resid {}'.format(residue))
    chis = [res.chi1_selection() for res in sel.residues]
    dihs = dihedrals.Dihedral(chis).run()
    #chi1_atoms = u.select_atoms('resid {} and (name N or name CA or name CB or name CG)'.format(residue))
    #atom_indices = [atom.index for atom in chi1_atoms]
    #print(atom_indices)
    #dih = dihedrals.Dihedral(chi1_atoms).run()
    chi1_angles = dihs.results.angles
    return chi1_angles

logging.info('Processing pdb {}'.format(args.pdb))
top = md.load('{}'.format(args.pdb)).topology
t = md.load('{}'.format(args.pdb), top=top)
distances = np.empty([52])
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
#LYS307: how to show rotation of TM5 to prevent clash with gprotein
#SER306: how to show interaction with gprotein?
#SER315 hbond to Arg200 in the inactive state
distances[12] = distance_calculation_atoms_mdtraj(t, top, "resSeq 315 and name OG" , "resSeq 200 and name NH2")
#PHE322 connector: how to describe difference?
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
#GLU347 1.active, 2. inactive
distances[18] = 0
distances[19] = distance_calculation_atoms_mdtraj(t, top, "resSeq 347 and name OE1" , "resSeq 95 and name OH")
#TYR350 1.active (but also inactive from other site) 2.inactive
distances[20] = distance_calculation_atoms_mdtraj(t, top, "resSeq 350 and name OH" , "resSeq 256 and name OD1")
distances[21] = distance_calculation_atoms_mdtraj(t, top, "resSeq 350 and name OH" , "protein and resSeq 96 and name O")
#PHE351 different confromations
distances[22] = dihedral_calculation_dihedral_mdtraj(t, top, 351)
#TYR351 three, 1. active
distances[23] = distance_calculation_atoms_mdtraj(t, top, "resSeq 353 and name OH" , "resSeq 183 and name OH")
distances[24] = distance_calculation_atoms_mdtraj(t, top, "resSeq 353 and name OH" , "resSeq 255 and name NE2")
distances[25] = distance_calculation_atoms_mdtraj(t, top, "resSeq 353 and name OH" , "resSeq 256 and name OD1")
#LEU355
distances[26] = dihedral_calculation_dihedral_mdtraj(t, top, 355)
#ASP367 two conf, one more to middle, one less
distances[27] = distance_calculation_atoms_mdtraj(t, top, "resSeq 367 and name CG" , "resSeq 317 and name CA")
#DPxxY
distances[28] = dihedral_calculation_dihedral_mdtraj(t, top, 372)
distances[29] = dihedral_calculation_dihedral_mdtraj(t, top, 373)
#outward movement of TM7 --> good measure maybe position of CA ALA374 toward a stable point
#position of H8
#gprotein interactions
#ARG200?
distances[31] = 0
distances[32] = 0
distances[33] = 0
#distances[si, :, 27:28] = distance_calculation_atoms_mdtraj(t, top, "resSeq 306 and name OG" , "resSeq 358 and name OH")
#SER306
distances[34] = distance_calculation_atoms_mdtraj(t, top, "resSeq 305 and name CA" , "resSeq 186 and name CA")
distances[35] = distance_calculation_atoms_mdtraj(t, top, "resSeq 306 and name OG" , "resSeq 308 and name NZ")
#distances[36] = dihedral_calculation_dihedral_mdtraj(t, top, 306)
distances[36] = 0
distances[37] = distance_calculation_atoms_mdtraj(t, top, "resSeq 306 and name OG" , "protein and resSeq 302 and name O")
#TM6
distances[38] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 312 and name CA',  'resSeq 200 and name CA')
#YY
distances[39] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 290 and name CZ', 'resSeq 371 and name CZ')
#TM5 bulge
distances[40] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 275 and name CA', 'resSeq 357 and name CA')
#cvs
distances[41] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 145 and name CA', 'resSeq 372 and name CA')
distances[42] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 216 and name CA', 'resSeq 308 and name CA')
distances[43] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 372 and name CA', 'resSeq 216 and name CA')
distances[44] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 184 and name CA', 'resSeq 315 and name CA')
distances[45] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 307 and name CA', 'resSeq 294 and name CA')
#a100 dist
distances[46] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 123 and name CA', 'resSeq 373 and name CA')
distances[47] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 148 and name CA', 'resSeq 187 and name CA')
distances[48] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 192 and name CA', 'resSeq 219 and name CA')
distances[49] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 298 and name CA', 'resSeq 312 and name CA')
distances[50] = distance_calculation_atoms_mdtraj(t, top, 'resSeq 336 and name CA', 'resSeq 353 and name CA')
distances[51] = - 14.43 * distances[46]*10 - 7.62 * distances[47]*10 + 9.11 * distances[48]*10 - 6.32 * distances[49]*10 - 5.22 * distances[50]*10 + 278.88
np.savetxt('PAR1_spec_{}.txt'.format(args.name), distances)