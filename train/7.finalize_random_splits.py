import json
import os
import random

# Get the names of the pdbs with successfully added cen (smina) precalculated
# terms.
with open("all_cen.types") as f:
    all_cen_lines = f.readlines()
    types_lines = {
        os.path.basename(line.split()[-1].replace("_ligand.mol2.ph7.gninatypes", "")): line
        for line in all_cen_lines
    }


def make_types_file(train_types_lines, test_types_lines, idx, prefix="randomsplit"):
    train_filename = f"{prefix}train{idx}_cen.types"
    test_filename = f"{prefix}test{idx}_cen.types"

    with open(train_filename, "w") as f:
        f.writelines(train_types_lines)

    with open(test_filename, "w") as f:
        f.writelines(test_types_lines)


# Make random splits

# shuffle all_cen_lines
random.shuffle(all_cen_lines)

# split into 3
random_split1 = all_cen_lines[: len(all_cen_lines) // 3]
random_split2 = all_cen_lines[len(all_cen_lines) // 3 : 2 * len(all_cen_lines) // 3]
random_split3 = all_cen_lines[2 * len(all_cen_lines) // 3 :]

make_types_file(random_split1 + random_split2, random_split3, 0, "randomsplit")
make_types_file(random_split1 + random_split3, random_split2, 1, "randomsplit")
make_types_file(random_split2 + random_split3, random_split1, 2, "randomsplit")
