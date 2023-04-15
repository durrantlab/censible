# Label is always 1
# Affinity is pK(affinity)
# then path to receptor gninatypes file
# then path to ligand gninatypes file
# like this:
# 1 5.886100 1b3l/1b3l_rec.gninatypes 1b3l/1b3l_crystal.gninatypes

import numpy as np
import glob
import os

# conda activate py39

# First get affinities
affinity_lines = open("1.pdbbind2020/refined-set/index/INDEX_refined_set.2020").readlines()
affinity_lines += open("1.pdbbind2020/refined-set/index/INDEX_general_PL.2020").readlines()

# Remove lines that start with "#"
affinity_lines = [line for line in affinity_lines if not line.startswith("#")]
# Remove empty lines
affinity_lines = [line for line in affinity_lines if line.strip() != ""]

# pdbid and affinity are in 1st and 4th column
affinity_dict = {line.split()[0]: line.split()[3] for line in affinity_lines}

# Map pdbids to their paths too
paths = glob.glob("1.pdbbind2020/refined-set/????") + glob.glob("1.pdbbind2020/v2020-other-PL/????")
paths = {os.path.basename(path): path for path in paths}

# Need to convert the values into pK(affinity)
# pK(affinity) = -log10(affinity)

pdbids_to_remove = []

for pdbid in affinity_dict:
    affinity = affinity_dict[pdbid]

    # Let's say ~ and = are the same
    affinity = affinity.replace("~", "=")

    # If affinity is ">" than something, throw it out. Affinity not precisely
    # defined.
    if ">" in affinity:
        # print(f"Affinity for {pdbid} is not in the right format: {affinity}")
        pdbids_to_remove.append(pdbid)
        continue

    affinity = affinity.replace("<=", "<")

    # If affinity is "<" than something, and that something is nM or pM
    # affinity, assume it is equal.
    if "<" in affinity and ("nM" in affinity or "pM" in affinity):
        affinity = affinity.replace("<", "=")

    # If affinity is "<" than something, and that something is uM, and the
    # something is < 1 (so really nM), assume it is equal.
    if "<" in affinity and "uM" in affinity and float(affinity.split("<")[1].split("uM")[0]) < 1:
        affinity = affinity.replace("<", "=")

    if not "=" in affinity:
        # print(f"Affinity for {pdbid} is not in the right format: {affinity}")
        pdbids_to_remove.append(pdbid)
        continue

    affinity = affinity.split("=")[1]
    affinity = affinity.replace("mM", "e-3")
    affinity = affinity.replace("uM", "e-6")
    affinity = affinity.replace("nM", "e-9")
    affinity = affinity.replace("pM", "e-12")
    affinity = affinity.replace("fM", "e-15")

    affinity_dict[pdbid] = -np.log10(float(affinity))

    # Store as string, 6 sig figs
    affinity_dict[pdbid] = f"{affinity_dict[pdbid]:.6f}"

for pdbid in pdbids_to_remove:
    del affinity_dict[pdbid]

g = open("all_centered.types", "w")
with open("all.types", "w") as f:
    for pdbid in affinity_dict:
        f.write(f"1 {affinity_dict[pdbid]} {paths[pdbid]}/{pdbid}_protein.gninatypes {paths[pdbid]}/{pdbid}_ligand.gninatypes\n")
        g.write(f"1 {affinity_dict[pdbid]} {paths[pdbid]}/{pdbid}_protein_centered.gninatypes {paths[pdbid]}/{pdbid}_ligand_centered.gninatypes\n")
g.close()