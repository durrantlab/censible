# Use /home/jdurrant/DataC/miniconda3/envs/my-rdkit-env/bin/python3

# Make it so you can save to hdf5 files
import os
import h5py
import glob
import requests
import subprocess
import numpy as np
import hashlib
import json
import re


def create_initial_hd5_file():
    # If pdbbind2020.hdf5 doens't exist, make it.
    if os.path.exists("pdbbind2020.hdf5"):
        return

    index_files = [
        "1.pdbbind2020/refined-set/index/INDEX_refined_name.2020",
        "1.pdbbind2020/refined-set/index/INDEX_general_PL_name.2020",
    ]

    # Load the lines
    lines = []
    for index_file in index_files:
        with open(index_file, "r") as f:
            lines += f.readlines()

    # Remove lines that begin with '#'
    lines = [l for l in lines if l[0] != "#"]

    # Split the lines into a list of lists
    lines = [l.split() for l in lines]

    # Get the PDB IDs
    pdb_ids = [l[0] for l in lines]

    # Get the uniprot ids
    uniprot_ids = [l[2] for l in lines]

    data = {
        pdb_id: {"uniprot": uniprot_id}
        for pdb_id, uniprot_id in zip(pdb_ids, uniprot_ids)
    }
    dirs = glob.glob("1.pdbbind2020/refined-set/????") + glob.glob(
        "1.pdbbind2020/v2020-other-PL/????"
    )
    dirs = {os.path.basename(d): d for d in dirs}

    with h5py.File("pdbbind2020.hdf5", "w") as h5file:
        for pdb_id, value in data.items():
            pdb_group = h5file.create_group(pdb_id)
            pdb_group.create_dataset("uniprot", data=value["uniprot"])
            pdb_group.create_dataset("dir", data=dirs[pdb_id])

    h5file.close()


def add_pdb_author_chains(h5file):
    # Get the pdb chains
    for pdbid in h5file.keys():
        # If has chain key, skip
        if "auth_chains" in h5file[pdbid].keys():
            print(f"Skipping {pdbid} because it already has a auth_chains key.")
            continue
        print(f"Processing {pdbid} to find auth_chains...")

        # Load the pocket file to get the chains
        pocket_file = (
            h5file[pdbid]["dir"][()].decode("utf-8") + "/" + pdbid + "_pocket.pdb"
        )

        # Load the pocket file
        with open(pocket_file, "r") as pocket_f:
            pocket_lines = pocket_f.readlines()
            # Keep only lines that begin with 'ATOM' or 'HETATM'
            pocket_lines = [l for l in pocket_lines if l[:5] in ["ATOM ", "HETAT"]]
            # Remove lines with resname HOH
            pocket_lines = [l for l in pocket_lines if l[17:20] != "HOH"]
            chains = [l[21] for l in pocket_lines]
            # unique chains only
            chains = list(set(chains))
            chains = [c for c in chains if c != " "]
            # Sort the chains
            chains.sort()
            chains = "".join(chains)

            # Save chains list to hdf5 file
            h5file[pdbid].create_dataset("auth_chains", data=chains)


def get_remote_txt(url):
    # Make a hash from url that you could save to a filename

    hash_object = hashlib.md5(url.encode())
    hex_dig = hash_object.hexdigest()
    flnm = f"./cache/{hex_dig}.txt"

    if os.path.exists(flnm):
        # Load the file
        with open(flnm, "r") as f:
            txt = f.read()
    else:
        # Get the contents of the url
        r = requests.get(url)
        if r.status_code != 200:
            # Throw an error
            raise Exception(f"Error getting {url}. Status code: {r.status_code}")
        txt = r.text

        # Save to file
        with open(flnm, "w") as f:
            f.write(txt)

    return txt


def add_pdb_chains(h5file):
    # Unfortunately, the chains in the PDB file itself are the author chains,
    # but you need to pdb-assigned chains to use with the RCSB API. Also
    # unfortunately, these conversions are not present in the PDBBind-processed
    # PDB files. So we need to scrape the PDB website.

    for pdbid in h5file.keys():
        if "chains" in h5file[pdbid].keys():
            print(f"Skipping {pdbid} because it already has a chains key.")
            continue
        print(f"Processing {pdbid} to find chains...")

        auth_chains = h5file[pdbid]["auth_chains"][()].decode("utf-8")

        url = f"https://www.rcsb.org/structure/{pdbid.upper()}"
        html = get_remote_txt(url)

        # Keep only portion above ligand table (proteins)
        html = html.split('<div class="panel-title">Small Molecules</div>')[0]

        real_chains = ""

        for auth_chain in auth_chains:
            # Find the line with the chain
            # consider regex
            # macromolecule-entityId.{1,500}\>(.) \[auth L\]

            # Find it in the html
            rgx = r"macromolecule-entityId.{1,500}\>(.) \[auth " + auth_chain + r"\]"
            m = re.search(rgx, html)

            if m is None:
                # This means the two are the same
                real_chains += auth_chain
            else:
                # Get the first group
                real_chains += m.group(1)

        # Save chains list to hdf5 file
        h5file[pdbid].create_dataset("chains", data=real_chains)


def add_scop2(h5file):
    # Now get the scop2 family for each chain
    for pdbid in h5file.keys():
        # If has scop2_fam key, skip
        if "scop2_fams" in h5file[pdbid].keys():
            print(f"Skipping {pdbid} because it already has a scop2_fams key.")
            continue
        print(f"Processing {pdbid} to find scop2_fams...")

        # Get the contents of the url
        scop2_fam_ids = []
        for chain in h5file[pdbid]["chains"][()].decode("utf-8"):
            # NOTE: Assigning scope id wherever you can, but not always possible.
            try:
                url = f"https://data.rcsb.org/rest/v1/core/polymer_entity_instance/{pdbid.upper()}/{chain}"
                r = json.loads(get_remote_txt(url))

                if "rcsb_polymer_instance_annotation" not in r:
                    # This means that there is no scop2 annotation for this chain
                    # so we will skip it
                    print(
                        f"Skipping {pdbid} {chain} because there is no scop2 annotation."
                    )
                    continue

                # Get the scop2_fam
                scop2_fam = [
                    v
                    for v in r["rcsb_polymer_instance_annotation"]
                    if v["type"] == "SCOP2"
                ]

                if not scop2_fam:
                    print(
                        f"Skipping {pdbid} {chain} because there is no scop2 annotation."
                    )
                    continue

                if "annotation_lineage" not in scop2_fam[0]:
                    print(
                        f"Skipping {pdbid} {chain} because there is no scop2 annotation."
                    )
                    continue

                scop2_fam = scop2_fam[0]["annotation_lineage"]
                scop2_fam_id = [v for v in scop2_fam if v["depth"] == 3]

                if not scop2_fam_id:
                    print(
                        f"Skipping {pdbid} {chain} because there is no scop2 annotation."
                    )
                    continue

                if "id" not in scop2_fam_id[0]:
                    print(
                        f"Skipping {pdbid} {chain} because there is no scop2 annotation."
                    )
                    continue

                scop2_fam_id = scop2_fam_id[0]["id"]
                scop2_fam_ids.append(scop2_fam_id)
            except Exception:
                auth_chain = h5file[pdbid]["auth_chains"][()].decode("utf-8")
                print(f"Error: {pdbid} {chain} [auth={auth_chain}]: {url}")
                # import pdb; pdb.set_trace()
                continue

        # Unique
        scop2_fam_ids = list(set(scop2_fam_ids))
        scop2_fam_ids = ",".join(scop2_fam_ids)

        # Save to hdf5 file
        h5file[pdbid].create_dataset("scop2_fams", data=scop2_fam_ids)


def add_smiles(h5file):
    # Now add in the canonical smiles strings of the ligands
    for pdbid in h5file.keys():
        # Remove smiles (to reset)
        # if 'smiles' in f[pdbid].keys():
        #     del f[pdbid]['smiles']
        # continue

        # If has smiles key, skip
        if "smiles" in h5file[pdbid].keys():
            print(f"Skipping {pdbid} because it already has a smiles key.")
            continue
        print(f"Processing {pdbid} to find smiles...")

        d = h5file[pdbid]["dir"][()].decode("utf-8")
        flnm = f"{d}/{pdbid}_ligand.sdf"
        if not os.path.exists(flnm):
            print(f"Warning: {pdbid} has no ligand file.")
            continue

        cmd = f"/usr/bin/obabel -isdf {flnm} -osmi"

        # Run cmd and capture its output to a string
        p = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        smiles = (
            p.stdout.read().decode("utf-8").strip().split("converted\n")[1].split()[0]
        )
        print(smiles)

        # if "***" in smiles:
        #     print(flnm)
        #     import pdb; pdb.set_trace()

        # # Load the mol2 file into rdkit
        # mols = Chem.SDMolSupplier(flnm)

        # # Get the canonical smiles
        # try:
        #     smiles = Chem.MolToSmiles(mols[0], isomericSmiles=True)
        # except:
        #     print(f"Error: {pdbid} smiles error: {flnm}")
        #     continue

        # Save to hdf5 file
        h5file[pdbid].create_dataset("smiles", data=smiles)


create_initial_hd5_file()

h5file = h5py.File("pdbbind2020.hdf5", "a")

add_pdb_author_chains(h5file)
add_pdb_chains(h5file)
add_scop2(h5file)
add_smiles(h5file)


# Now place them in three separate bins. TODO: Check this.
pdbid_bins = [set([]), set([]), set([])]
smiles_bins = [[], [], []]
scop2_fams_bins = [[], [], []]


def add_to_bin(idx, pdbid, smiles, scop2_fams):
    global pdbid_bins, smiles_bins, scop2_fams_bins
    pdbid_bins[idx].add(pdbid)
    smiles_bins[idx].append(smiles)
    for scop2 in scop2_fams:
        scop2_fams_bins[idx].append(scop2)


pdbids = list(h5file.keys())

# sort pdbids, scop2_fams, and smiles by the length of scop2_fams
data = zip(
    pdbids,
    [
        set(h5file[pdbid]["scop2_fams"][()].decode("utf-8").split(","))
        for pdbid in pdbids
    ],
    [h5file[pdbid]["smiles"][()].decode("utf-8") for pdbid in pdbids],
)
data = sorted(data, key=lambda x: len(x[1]))

# First, position the easy ones.
for pdbid_idx, datum in enumerate(data):
    pdbid, scop2_fams, smiles = datum
    # scop2_fams = set(h5file[pdbid]['scop2_fams'][()].decode('utf-8').split(','))
    # smiles = h5file[pdbid]['smiles'][()].decode('utf-8')

    smiles_or_scop2_already_placed = False
    for i in range(3):
        if smiles in smiles_bins[i]:
            smiles_or_scop2_already_placed = True
            break

        if scop2_fams.intersection(scop2_fams_bins[i]):
            smiles_or_scop2_already_placed = True
            break

    if not smiles_or_scop2_already_placed:
        # No need to worry about redundancies, so just add it to the bin
        # with the fewest pdbids
        min_bin_idx = np.argmin([len(pdbid_bins[i]) for i in range(3)])
        add_to_bin(min_bin_idx, pdbid, smiles, list(scop2_fams))
        data[pdbid_idx] = None

# Remove the ones that were placed
data = [datum for datum in data if datum is not None]

# Now position the rest where you can
for pdbid_idx, datum in enumerate(data):
    pdbid, scop2_fams, smiles = datum

    # Get the indices of the bins that have the same smiles
    same_smiles_bin_idxs = [i for i in range(3) if smiles in smiles_bins[i]]

    # Get the indices of the bins that have the same scop2_fams
    same_scop2_fams_bin_idxs = [
        i for i in range(3) if scop2_fams.intersection(scop2_fams_bins[i])
    ]

    # Get the intersection of the two
    same_bin_idxs = set(same_smiles_bin_idxs).union(set(same_scop2_fams_bin_idxs))

    # If that set is greater than 1, then this one cannot be placed without a redundancy
    if len(same_bin_idxs) > 1:
        print(f"Warning: {pdbid} cannot be placed without a redundancy.")
        continue

    idx_to_use = same_bin_idxs.pop()
    add_to_bin(idx_to_use, pdbid, smiles, list(scop2_fams))

    # Remove this one from the list
    data[pdbid_idx] = None

# Remove the ones that were placed
data = [datum for datum in data if datum is not None]

h5file.close()

# Now, sanity checks
for i in range(3):
    print(f"Bin {i} has {len(pdbid_bins[i])} pdbids.")
    print(f"Bin {i} has {len(smiles_bins[i])} smiles.")
    print(f"Bin {i} has {len(scop2_fams_bins[i])} scop2_fams.")

# Make sure none of the pdbid_bins have overlap
for i in range(3):
    for j in range(i + 1, 3):
        print(
            f"Bin {i} and {j} have {len(pdbid_bins[i].intersection(pdbid_bins[j]))} pdbids in common."
        )

# Same for smiles
for i in range(3):
    for j in range(i + 1, 3):
        print(
            f"Bin {i} and {j} have {len(set(smiles_bins[i]).intersection(set(smiles_bins[j])))} smiles in common."
        )

# Some PDBs couldn't be placed without a redundancy
print(f"There are {len(data)} PDBs that couldn't be placed without a redundancy.")

# Save the pdbid bins, json format. Note that it's deterministic.
with open("pdbid_bins.json", "w") as f:
    json.dump(
        {
            "bin1Size": len(pdbid_bins[0]),
            "bin2Size": len(pdbid_bins[1]),
            "bin3Size": len(pdbid_bins[2]),
            "numToPlaced": len(data),
            "bins": [sorted(list(b)) for b in pdbid_bins],
        },
        f,
    )


# import pdb; pdb.set_trace()


# https://data.rcsb.org/rest/v1/core/polymer_entity_instance/4HHB/A
