import json
import os
import random

# Load the file pdbid_bins.json
with open('pdbid_bins.json', 'r') as f:
    bin_data = json.load(f)

# Get the names of the pdbs with successfully added cen (smina) precalculated
# terms.
with open("all_cen.types") as f:
    all_cen_lines = f.readlines()
    types_lines = {os.path.basename(line.split()[-1].replace("_ligand.gninatypes", "")): line for line in all_cen_lines}

bin1 = bin_data["bins"][0]
bin2 = bin_data["bins"][1]
bin3 = bin_data["bins"][2]

bin1_types_lines = [types_lines[pdb] for pdb in bin1 if pdb in types_lines]
bin2_types_lines = [types_lines[pdb] for pdb in bin2 if pdb in types_lines]
bin3_types_lines = [types_lines[pdb] for pdb in bin3 if pdb in types_lines]

def make_types_file(train_types_lines, test_types_lines, idx, prefix="crystal"):
    train_filename = f"{prefix}train{idx}_cen.types"
    test_filename = f"{prefix}test{idx}_cen.types"

    with open(train_filename, 'w') as f:
        f.writelines(train_types_lines)
    
    with open(test_filename, 'w') as f:
        f.writelines(test_types_lines)

make_types_file(bin1_types_lines + bin2_types_lines, bin3_types_lines, 0)
make_types_file(bin1_types_lines + bin3_types_lines, bin2_types_lines, 1)
make_types_file(bin2_types_lines + bin3_types_lines, bin1_types_lines, 2)

# Also make random splits, to compare with the original implementation.

# shuffle all_cen_lines
random.shuffle(all_cen_lines)

# split into 3
random_split1 = all_cen_lines[:len(all_cen_lines)//3]
random_split2 = all_cen_lines[len(all_cen_lines)//3:2*len(all_cen_lines)//3]
random_split3 = all_cen_lines[2*len(all_cen_lines)//3:]

make_types_file(random_split1 + random_split2, random_split3, 0, "randomsplit")
make_types_file(random_split1 + random_split3, random_split2, 1, "randomsplit")
make_types_file(random_split2 + random_split3, random_split1, 2, "randomsplit")
