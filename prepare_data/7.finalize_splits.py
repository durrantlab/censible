import json
import os

# Load the file pdbid_bins.json
with open('pdbid_bins.json', 'r') as f:
    bin_data = json.load(f)

# Get the names of the pdbs with successfully added cen (smina) precalculated
# terms.
with open("all_cen.types") as f:
    lines = f.readlines()
    types_lines = {os.path.basename(line.split()[-1].replace("_ligand.gninatypes", "")): line for line in lines}

bin1 = bin_data["bins"][0]
bin2 = bin_data["bins"][1]
bin3 = bin_data["bins"][2]

bin1_types_lines = [types_lines[pdb] for pdb in bin1 if pdb in types_lines]
bin2_types_lines = [types_lines[pdb] for pdb in bin2 if pdb in types_lines]
bin3_types_lines = [types_lines[pdb] for pdb in bin3 if pdb in types_lines]

def make_types_file(train_types_lines, test_types_lines, idx):
    train_filename = f"crystaltrain{idx}_cen.types"
    test_filename = f"crystaltest{idx}_cen.types"

    with open(train_filename, 'w') as f:
        f.writelines(train_types_lines)
    
    with open(test_filename, 'w') as f:
        f.writelines(test_types_lines)

make_types_file(bin1_types_lines + bin2_types_lines, bin3_types_lines, 0)
make_types_file(bin1_types_lines + bin3_types_lines, bin2_types_lines, 1)
make_types_file(bin2_types_lines + bin3_types_lines, bin1_types_lines, 2)
