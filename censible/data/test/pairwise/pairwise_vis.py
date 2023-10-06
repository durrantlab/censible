from typing import Tuple
import numpy as np
import os
import re


# NOTE: Smina type assignments 100% depend on protonation states. Confirmed with
# this project.

SMINA_EXEC = "/home/jdurrant/miniconda3/envs/censible/bin/smina"

TYPES_TO_RADIUS = {
    "Hydrogen": 0.000000,
    "PolarHydrogen": 0.000000,
    "AliphaticCarbonXSHydrophobe": 1.900000,
    "AliphaticCarbonXSNonHydrophobe": 1.900000,
    "AromaticCarbonXSHydrophobe": 1.900000,
    "AromaticCarbonXSNonHydrophobe": 1.900000,
    "Nitrogen": 1.800000,
    "NitrogenXSDonor": 1.800000,
    "NitrogenXSDonorAcceptor": 1.800000,
    "NitrogenXSAcceptor": 1.800000,
    "Oxygen": 1.700000,
    "OxygenXSDonor": 1.700000,
    "OxygenXSDonorAcceptor": 1.700000,
    "OxygenXSAcceptor": 1.700000,
    "Sulfur": 2.000000,
    "SulfurAcceptor": 2.000000,
    "Phosphorus": 2.100000,
    "Fluorine": 1.500000,
    "Chlorine": 1.800000,
    "Bromine": 2.000000,
    "Iodine": 2.200000,
    "Magnesium": 1.200000,
    "Manganese": 1.200000,
    "Zinc": 1.200000,
    "Calcium": 1.200000,
    "Iron": 1.200000,
    "GenericMetal": 1.200000,
}


class PDBParser:
    def __init__(self, filename):
        self.filename = filename
        self.atoms = []
        self.coordinates = []
        self.parse_file()

    def parse_file(self):
        with open(self.filename, "r") as f:
            atom_idx = 0
            for line in f:
                if "HOH" in line or "WAT" in line or "TIP" in line:
                    # No water molecules
                    continue

                if line.startswith(("ATOM", "HETATM")):
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    atom_data = {
                        "atom_idx": atom_idx,
                        "record": line[0:6].strip(),
                        "atom_num": int(line[6:11].strip()),
                        "atom_name": line[12:16].strip(),
                        "alt_loc": line[16].strip(),
                        "res_name": line[17:20].strip(),
                        "chain_id": line[21].strip(),
                        "res_num": int(line[22:26].strip()),
                        "iCode": line[26].strip(),
                        "occupancy": float(line[54:60].strip())
                        if line[54:60].strip()
                        else None,
                        "temp_factor": float(line[60:66].strip())
                        if line[60:66].strip()
                        else None,
                        "element": line[76:78].strip(),
                        "charge": line[78:80].strip(),
                        "type": "",
                        "type_pairs": {},
                    }
                    atom_idx = atom_idx + 1
                    self.atoms.append(atom_data)
                    self.coordinates.append([x, y, z])

        self.coordinates = np.array(self.coordinates)

        self.assign_atom_types()

        # Now keep only those atoms with types. Remove also the corresponding
        # entries in self.coordinates
        self.atoms = [a for a in self.atoms if a["type"] != ""]
        self.coordinates = self.coordinates[[a["atom_idx"] for a in self.atoms]]

    def total_gauss(self, type_pair: Tuple):
        type_pair = all_pair_to_censible_pair[type_pair]
        return sum(
            [
                a["type_pairs"][type_pair] if type_pair in a["type_pairs"] else 0
                for a in self.atoms
            ]
        )

    def save_pdb(self, filename, type_pairs_for_beta=None, beta_scale=1):
        with open(filename, "w") as f:
            for i, atom in enumerate(self.atoms):
                beta_val = 0
                if type_pairs_for_beta is not None:
                    type_pairs_for_beta = all_pair_to_censible_pair[type_pairs_for_beta]
                    if type_pairs_for_beta in atom["type_pairs"]:
                        beta_val = atom["type_pairs"][type_pairs_for_beta] * beta_scale
                f.write(self.make_pdb_line(i, beta_val) + "\n")

    def make_pdb_line(self, atom_idx, beta_val):
        atom = self.atoms[atom_idx]
        atom_name = atom["atom_name"]
        res_name = atom["res_name"]
        chain_id = atom["chain_id"]
        res_num = atom["res_num"]
        x, y, z = self.coordinates[atom_idx]
        atom_type = atom["type"]
        element = atom["element"]
        radius = TYPES_TO_RADIUS[atom_type]
        return f"ATOM  {atom_idx:5d} {atom_name:4s} {res_name:3s} {chain_id:1s}{res_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{beta_val:6.2f}          {element:2s}  {radius:6.3f}"

    def assign_atom_types(self):
        cmd = f"{SMINA_EXEC} --atom_terms test.txt -r {self.filename} -l {self.filename} --score_only"
        os.system(cmd)
        with open("test.txt") as f:
            lines = f.readlines()
        lines = [l for l in lines if "<" in l]
        types = []
        coords = []
        for line in lines:
            # Regular expression pattern to extract the type and coordinates
            pattern = r"(\w+)\s+<(-?[\d\.]+),(-?[\d\.]+),(-?[\d\.]+)>"
            match = re.search(pattern, line)
            if match:
                typ = match.group(1)
                if typ == "PolarHydrogen":
                    # We're ignoring hydrogens. CENsible doens't use them.
                    continue
                types.append(typ)
                coords.append(
                    [
                        float(match.group(2)),
                        float(match.group(3)),
                        float(match.group(4)),
                    ]
                )
            else:
                print("WARNING! No match: " + line)
        coords = np.array(coords)

        # For each coordinate in coords, I want to find the coordinate in
        # self.coordinates that is closest to it. Then I want to add the
        # corresponding type to the self.atoms entry for that closest match.
        for c, t in zip(coords, types):
            distances = np.linalg.norm(self.coordinates - c, axis=1)
            closest_idx = np.argmin(distances)
            self.atoms[closest_idx]["type"] = t

    def get_coordinates(self):
        return self.coordinates

    def get_atoms(self):
        return self.atoms

    def keep_only_near_coords(self, coords: np.array, cutoff=8):
        # Remove any entries in self.atoms that are not within cutoff of any
        # coordinate in coords. Also remove associated entries in
        # self.coordinates

        # Create an index list of atoms to keep
        to_keep = []

        for i, atom_coord in enumerate(self.coordinates):
            distances = np.linalg.norm(coords - atom_coord, axis=1)
            if np.any(distances < cutoff):
                to_keep.append(i)

        # Filter atoms and coordinates using the index list
        self.atoms = [self.atoms[i] for i in to_keep]
        self.coordinates = self.coordinates[to_keep]

    def get_all_atom_types(self) -> set:
        return {a["type"] for a in self.atoms}

    def add_atomic_gauss(self, atom_idx, atom_type_pair: Tuple, val: float):
        # Standardize the atom_type_pair order
        atom_type_pair = all_pair_to_censible_pair[atom_type_pair]
        if atom_type_pair not in self.atoms[atom_idx]["type_pairs"]:
            self.atoms[atom_idx]["type_pairs"][atom_type_pair] = val
        else:
            self.atoms[atom_idx]["type_pairs"][atom_type_pair] += val


def load_censible_output(filename):
    with open(filename) as f:
        lines = f.readlines()
    header = [l for l in lines if "gauss(o=0,_w=0.5,_c=8)" in l][0].split()
    precalc_smina_terms = [l for l in lines if "precalc_smina_term" in l][0].split()[1:]
    normalized_precalc_smina_terms = [
        l for l in lines if "normalized_precalc_smina_term" in l
    ][0].split()[1:]
    predicted_weights = [l for l in lines if "predicted_weight" in l][0].split()[1:]
    predicted_contributions = [l for l in lines if "predicted_contribution" in l][
        0
    ].split()[1:]

    scale_factors = [
        float(n) / float(p) if float(p) != 0 else 1
        for n, p in zip(normalized_precalc_smina_terms, precalc_smina_terms)
    ]

    data = {}
    for i, h in enumerate(header):
        data[h] = {
            "precalc_smina_term": float(precalc_smina_terms[i]),
            "normalized_precalc_smina_term": float(normalized_precalc_smina_terms[i]),
            "predicted_weight": float(predicted_weights[i]),
            "predicted_contribution": float(predicted_contributions[i]),
            "scale_factor": scale_factors[i],
        }
    return data


def type_pair_to_censible_key(atom_type_pair):
    atom_type_pair = all_pair_to_censible_pair[atom_type_pair]
    return f"atom_type_gaussian(t1={atom_type_pair[0]},t2={atom_type_pair[1]},o=0,_w=1,_c=8)"


censible_output = load_censible_output("test_out.tsv")

receptor = PDBParser("./1wdn_receptor.pdb.censible.converted.pdb")
ligand = PDBParser("./1wdn_ligand.pdb.censible.converted.pdb")
cutoff = 8

receptor.keep_only_near_coords(ligand.get_coordinates(), cutoff=cutoff)

# Now get all pairwise combinations of types
censible_pairings = [
    ["AliphaticCarbonXSHydrophobe", "AliphaticCarbonXSHydrophobe"],
    ["AliphaticCarbonXSHydrophobe", "AliphaticCarbonXSNonHydrophobe"],
    ["AliphaticCarbonXSHydrophobe", "AromaticCarbonXSHydrophobe"],
    ["AliphaticCarbonXSHydrophobe", "AromaticCarbonXSNonHydrophobe"],
    ["AliphaticCarbonXSHydrophobe", "Bromine"],
    ["AliphaticCarbonXSHydrophobe", "Chlorine"],
    ["AliphaticCarbonXSHydrophobe", "Fluorine"],
    ["AliphaticCarbonXSHydrophobe", "GenericMetal"],
    ["AliphaticCarbonXSHydrophobe", "Magnesium"],
    ["AliphaticCarbonXSHydrophobe", "Manganese"],
    ["AliphaticCarbonXSHydrophobe", "Nitrogen"],
    ["AliphaticCarbonXSHydrophobe", "NitrogenXSAcceptor"],
    ["AliphaticCarbonXSHydrophobe", "NitrogenXSDonor"],
    ["AliphaticCarbonXSHydrophobe", "NitrogenXSDonorAcceptor"],
    ["AliphaticCarbonXSHydrophobe", "OxygenXSAcceptor"],
    ["AliphaticCarbonXSHydrophobe", "OxygenXSDonorAcceptor"],
    ["AliphaticCarbonXSHydrophobe", "Phosphorus"],
    ["AliphaticCarbonXSHydrophobe", "Sulfur"],
    ["AliphaticCarbonXSHydrophobe", "Zinc"],
    ["AliphaticCarbonXSNonHydrophobe", "AliphaticCarbonXSNonHydrophobe"],
    ["AliphaticCarbonXSNonHydrophobe", "AromaticCarbonXSHydrophobe"],
    ["AliphaticCarbonXSNonHydrophobe", "AromaticCarbonXSNonHydrophobe"],
    ["AliphaticCarbonXSNonHydrophobe", "Bromine"],
    ["AliphaticCarbonXSNonHydrophobe", "Calcium"],
    ["AliphaticCarbonXSNonHydrophobe", "Chlorine"],
    ["AliphaticCarbonXSNonHydrophobe", "Fluorine"],
    ["AliphaticCarbonXSNonHydrophobe", "GenericMetal"],
    ["AliphaticCarbonXSNonHydrophobe", "Magnesium"],
    ["AliphaticCarbonXSNonHydrophobe", "Manganese"],
    ["AliphaticCarbonXSNonHydrophobe", "Nitrogen"],
    ["AliphaticCarbonXSNonHydrophobe", "NitrogenXSAcceptor"],
    ["AliphaticCarbonXSNonHydrophobe", "NitrogenXSDonor"],
    ["AliphaticCarbonXSNonHydrophobe", "NitrogenXSDonorAcceptor"],
    ["AliphaticCarbonXSNonHydrophobe", "OxygenXSAcceptor"],
    ["AliphaticCarbonXSNonHydrophobe", "OxygenXSDonorAcceptor"],
    ["AliphaticCarbonXSNonHydrophobe", "Phosphorus"],
    ["AliphaticCarbonXSNonHydrophobe", "Sulfur"],
    ["AliphaticCarbonXSNonHydrophobe", "Zinc"],
    ["AromaticCarbonXSHydrophobe", "AromaticCarbonXSHydrophobe"],
    ["AromaticCarbonXSHydrophobe", "AromaticCarbonXSNonHydrophobe"],
    ["AromaticCarbonXSHydrophobe", "Bromine"],
    ["AromaticCarbonXSHydrophobe", "Chlorine"],
    ["AromaticCarbonXSHydrophobe", "Fluorine"],
    ["AromaticCarbonXSHydrophobe", "Nitrogen"],
    ["AromaticCarbonXSHydrophobe", "NitrogenXSAcceptor"],
    ["AromaticCarbonXSHydrophobe", "NitrogenXSDonor"],
    ["AromaticCarbonXSHydrophobe", "NitrogenXSDonorAcceptor"],
    ["AromaticCarbonXSHydrophobe", "OxygenXSAcceptor"],
    ["AromaticCarbonXSHydrophobe", "OxygenXSDonorAcceptor"],
    ["AromaticCarbonXSHydrophobe", "Phosphorus"],
    ["AromaticCarbonXSHydrophobe", "Sulfur"],
    ["AromaticCarbonXSNonHydrophobe", "AromaticCarbonXSNonHydrophobe"],
    ["AromaticCarbonXSNonHydrophobe", "Bromine"],
    ["AromaticCarbonXSNonHydrophobe", "Chlorine"],
    ["AromaticCarbonXSNonHydrophobe", "Fluorine"],
    ["AromaticCarbonXSNonHydrophobe", "Nitrogen"],
    ["AromaticCarbonXSNonHydrophobe", "NitrogenXSAcceptor"],
    ["AromaticCarbonXSNonHydrophobe", "NitrogenXSDonor"],
    ["AromaticCarbonXSNonHydrophobe", "NitrogenXSDonorAcceptor"],
    ["AromaticCarbonXSNonHydrophobe", "OxygenXSAcceptor"],
    ["AromaticCarbonXSNonHydrophobe", "OxygenXSDonorAcceptor"],
    ["AromaticCarbonXSNonHydrophobe", "Phosphorus"],
    ["AromaticCarbonXSNonHydrophobe", "Sulfur"],
    ["Nitrogen", "Chlorine"],
    ["Nitrogen", "Fluorine"],
    ["Nitrogen", "Nitrogen"],
    ["Nitrogen", "NitrogenXSAcceptor"],
    ["Nitrogen", "NitrogenXSDonor"],
    ["Nitrogen", "NitrogenXSDonorAcceptor"],
    ["Nitrogen", "OxygenXSAcceptor"],
    ["Nitrogen", "OxygenXSDonorAcceptor"],
    ["Nitrogen", "Phosphorus"],
    ["Nitrogen", "Sulfur"],
    ["Nitrogen", "Zinc"],
    ["NitrogenXSAcceptor", "Chlorine"],
    ["NitrogenXSAcceptor", "Fluorine"],
    ["NitrogenXSAcceptor", "NitrogenXSAcceptor"],
    ["NitrogenXSAcceptor", "OxygenXSAcceptor"],
    ["NitrogenXSAcceptor", "OxygenXSDonorAcceptor"],
    ["NitrogenXSAcceptor", "Phosphorus"],
    ["NitrogenXSAcceptor", "Sulfur"],
    ["NitrogenXSAcceptor", "Zinc"],
    ["NitrogenXSDonor", "Bromine"],
    ["NitrogenXSDonor", "Chlorine"],
    ["NitrogenXSDonor", "Fluorine"],
    ["NitrogenXSDonor", "NitrogenXSAcceptor"],
    ["NitrogenXSDonor", "NitrogenXSDonor"],
    ["NitrogenXSDonor", "NitrogenXSDonorAcceptor"],
    ["NitrogenXSDonor", "OxygenXSAcceptor"],
    ["NitrogenXSDonor", "OxygenXSDonorAcceptor"],
    ["NitrogenXSDonor", "Phosphorus"],
    ["NitrogenXSDonor", "Sulfur"],
    ["NitrogenXSDonor", "Zinc"],
    ["NitrogenXSDonorAcceptor", "NitrogenXSAcceptor"],
    ["NitrogenXSDonorAcceptor", "OxygenXSAcceptor"],
    ["NitrogenXSDonorAcceptor", "OxygenXSDonorAcceptor"],
    ["NitrogenXSDonorAcceptor", "Sulfur"],
    ["NitrogenXSDonorAcceptor", "Zinc"],
    ["OxygenXSAcceptor", "Bromine"],
    ["OxygenXSAcceptor", "Chlorine"],
    ["OxygenXSAcceptor", "Fluorine"],
    ["OxygenXSAcceptor", "Magnesium"],
    ["OxygenXSAcceptor", "Manganese"],
    ["OxygenXSAcceptor", "OxygenXSAcceptor"],
    ["OxygenXSAcceptor", "Phosphorus"],
    ["OxygenXSAcceptor", "Sulfur"],
    ["OxygenXSAcceptor", "Zinc"],
    ["OxygenXSDonorAcceptor", "Bromine"],
    ["OxygenXSDonorAcceptor", "Chlorine"],
    ["OxygenXSDonorAcceptor", "Fluorine"],
    ["OxygenXSDonorAcceptor", "Magnesium"],
    ["OxygenXSDonorAcceptor", "OxygenXSAcceptor"],
    ["OxygenXSDonorAcceptor", "OxygenXSDonorAcceptor"],
    ["OxygenXSDonorAcceptor", "Phosphorus"],
    ["OxygenXSDonorAcceptor", "Sulfur"],
    ["OxygenXSDonorAcceptor", "Zinc"],
    ["Phosphorus", "Magnesium"],
    ["Sulfur", "Bromine"],
    ["Sulfur", "Chlorine"],
    ["Sulfur", "Fluorine"],
    ["Sulfur", "Phosphorus"],
    ["Sulfur", "Sulfur"],
    ["Sulfur", "Zinc"],
    ["AliphaticCarbonXSHydrophobe", "Calcium"],
    ["AliphaticCarbonXSHydrophobe", "Iodine"],
    ["AliphaticCarbonXSHydrophobe", "Iron"],
    ["AliphaticCarbonXSHydrophobe", "Oxygen"],
    ["AliphaticCarbonXSHydrophobe", "OxygenXSDonor"],
    ["AliphaticCarbonXSHydrophobe", "SulfurAcceptor"],
    ["AliphaticCarbonXSNonHydrophobe", "Iodine"],
    ["AliphaticCarbonXSNonHydrophobe", "Iron"],
    ["AliphaticCarbonXSNonHydrophobe", "Oxygen"],
    ["AliphaticCarbonXSNonHydrophobe", "OxygenXSDonor"],
    ["AliphaticCarbonXSNonHydrophobe", "SulfurAcceptor"],
    ["AromaticCarbonXSHydrophobe", "Calcium"],
    ["AromaticCarbonXSHydrophobe", "GenericMetal"],
    ["AromaticCarbonXSHydrophobe", "Iodine"],
    ["AromaticCarbonXSHydrophobe", "Iron"],
    ["AromaticCarbonXSHydrophobe", "Magnesium"],
    ["AromaticCarbonXSHydrophobe", "Manganese"],
    ["AromaticCarbonXSHydrophobe", "Oxygen"],
    ["AromaticCarbonXSHydrophobe", "OxygenXSDonor"],
    ["AromaticCarbonXSHydrophobe", "SulfurAcceptor"],
    ["AromaticCarbonXSHydrophobe", "Zinc"],
    ["AromaticCarbonXSNonHydrophobe", "Calcium"],
    ["AromaticCarbonXSNonHydrophobe", "GenericMetal"],
    ["AromaticCarbonXSNonHydrophobe", "Iodine"],
    ["AromaticCarbonXSNonHydrophobe", "Iron"],
    ["AromaticCarbonXSNonHydrophobe", "Magnesium"],
    ["AromaticCarbonXSNonHydrophobe", "Manganese"],
    ["AromaticCarbonXSNonHydrophobe", "Oxygen"],
    ["AromaticCarbonXSNonHydrophobe", "OxygenXSDonor"],
    ["AromaticCarbonXSNonHydrophobe", "SulfurAcceptor"],
    ["AromaticCarbonXSNonHydrophobe", "Zinc"],
    ["Bromine", "Bromine"],
    ["Bromine", "Calcium"],
    ["Bromine", "GenericMetal"],
    ["Bromine", "Iodine"],
    ["Bromine", "Iron"],
    ["Bromine", "Magnesium"],
    ["Bromine", "Manganese"],
    ["Bromine", "Zinc"],
    ["Calcium", "Calcium"],
    ["Calcium", "GenericMetal"],
    ["Calcium", "Iron"],
    ["Chlorine", "Bromine"],
    ["Chlorine", "Calcium"],
    ["Chlorine", "Chlorine"],
    ["Chlorine", "GenericMetal"],
    ["Chlorine", "Iodine"],
    ["Chlorine", "Iron"],
    ["Chlorine", "Magnesium"],
    ["Chlorine", "Manganese"],
    ["Chlorine", "Zinc"],
    ["Fluorine", "Bromine"],
    ["Fluorine", "Calcium"],
    ["Fluorine", "Chlorine"],
    ["Fluorine", "Fluorine"],
    ["Fluorine", "GenericMetal"],
    ["Fluorine", "Iodine"],
    ["Fluorine", "Iron"],
    ["Fluorine", "Magnesium"],
    ["Fluorine", "Manganese"],
    ["Fluorine", "Zinc"],
    ["GenericMetal", "GenericMetal"],
    ["Iodine", "Calcium"],
    ["Iodine", "GenericMetal"],
    ["Iodine", "Iodine"],
    ["Iodine", "Iron"],
    ["Iodine", "Magnesium"],
    ["Iodine", "Manganese"],
    ["Iodine", "Zinc"],
    ["Iron", "GenericMetal"],
    ["Iron", "Iron"],
    ["Magnesium", "Calcium"],
    ["Magnesium", "GenericMetal"],
    ["Magnesium", "Iron"],
    ["Magnesium", "Magnesium"],
    ["Magnesium", "Manganese"],
    ["Magnesium", "Zinc"],
    ["Manganese", "Calcium"],
    ["Manganese", "GenericMetal"],
    ["Manganese", "Iron"],
    ["Manganese", "Manganese"],
    ["Manganese", "Zinc"],
    ["Nitrogen", "Bromine"],
    ["Nitrogen", "Calcium"],
    ["Nitrogen", "GenericMetal"],
    ["Nitrogen", "Iodine"],
    ["Nitrogen", "Iron"],
    ["Nitrogen", "Magnesium"],
    ["Nitrogen", "Manganese"],
    ["Nitrogen", "Oxygen"],
    ["Nitrogen", "OxygenXSDonor"],
    ["Nitrogen", "SulfurAcceptor"],
    ["NitrogenXSAcceptor", "Bromine"],
    ["NitrogenXSAcceptor", "Calcium"],
    ["NitrogenXSAcceptor", "GenericMetal"],
    ["NitrogenXSAcceptor", "Iodine"],
    ["NitrogenXSAcceptor", "Iron"],
    ["NitrogenXSAcceptor", "Magnesium"],
    ["NitrogenXSAcceptor", "Manganese"],
    ["NitrogenXSAcceptor", "Oxygen"],
    ["NitrogenXSAcceptor", "OxygenXSDonor"],
    ["NitrogenXSAcceptor", "SulfurAcceptor"],
    ["NitrogenXSDonor", "Calcium"],
    ["NitrogenXSDonor", "GenericMetal"],
    ["NitrogenXSDonor", "Iodine"],
    ["NitrogenXSDonor", "Iron"],
    ["NitrogenXSDonor", "Magnesium"],
    ["NitrogenXSDonor", "Manganese"],
    ["NitrogenXSDonor", "Oxygen"],
    ["NitrogenXSDonor", "OxygenXSDonor"],
    ["NitrogenXSDonor", "SulfurAcceptor"],
    ["NitrogenXSDonorAcceptor", "Bromine"],
    ["NitrogenXSDonorAcceptor", "Calcium"],
    ["NitrogenXSDonorAcceptor", "Chlorine"],
    ["NitrogenXSDonorAcceptor", "Fluorine"],
    ["NitrogenXSDonorAcceptor", "GenericMetal"],
    ["NitrogenXSDonorAcceptor", "Iodine"],
    ["NitrogenXSDonorAcceptor", "Iron"],
    ["NitrogenXSDonorAcceptor", "Magnesium"],
    ["NitrogenXSDonorAcceptor", "Manganese"],
    ["NitrogenXSDonorAcceptor", "NitrogenXSDonorAcceptor"],
    ["NitrogenXSDonorAcceptor", "Oxygen"],
    ["NitrogenXSDonorAcceptor", "OxygenXSDonor"],
    ["NitrogenXSDonorAcceptor", "Phosphorus"],
    ["NitrogenXSDonorAcceptor", "SulfurAcceptor"],
    ["Oxygen", "Bromine"],
    ["Oxygen", "Calcium"],
    ["Oxygen", "Chlorine"],
    ["Oxygen", "Fluorine"],
    ["Oxygen", "GenericMetal"],
    ["Oxygen", "Iodine"],
    ["Oxygen", "Iron"],
    ["Oxygen", "Magnesium"],
    ["Oxygen", "Manganese"],
    ["Oxygen", "Oxygen"],
    ["Oxygen", "OxygenXSAcceptor"],
    ["Oxygen", "OxygenXSDonor"],
    ["Oxygen", "OxygenXSDonorAcceptor"],
    ["Oxygen", "Phosphorus"],
    ["Oxygen", "Sulfur"],
    ["Oxygen", "SulfurAcceptor"],
    ["Oxygen", "Zinc"],
    ["OxygenXSAcceptor", "Calcium"],
    ["OxygenXSAcceptor", "GenericMetal"],
    ["OxygenXSAcceptor", "Iodine"],
    ["OxygenXSAcceptor", "Iron"],
    ["OxygenXSAcceptor", "SulfurAcceptor"],
    ["OxygenXSDonor", "Bromine"],
    ["OxygenXSDonor", "Calcium"],
    ["OxygenXSDonor", "Chlorine"],
    ["OxygenXSDonor", "Fluorine"],
    ["OxygenXSDonor", "GenericMetal"],
    ["OxygenXSDonor", "Iodine"],
    ["OxygenXSDonor", "Iron"],
    ["OxygenXSDonor", "Magnesium"],
    ["OxygenXSDonor", "Manganese"],
    ["OxygenXSDonor", "OxygenXSAcceptor"],
    ["OxygenXSDonor", "OxygenXSDonor"],
    ["OxygenXSDonor", "OxygenXSDonorAcceptor"],
    ["OxygenXSDonor", "Phosphorus"],
    ["OxygenXSDonor", "Sulfur"],
    ["OxygenXSDonor", "SulfurAcceptor"],
    ["OxygenXSDonor", "Zinc"],
    ["OxygenXSDonorAcceptor", "Calcium"],
    ["OxygenXSDonorAcceptor", "GenericMetal"],
    ["OxygenXSDonorAcceptor", "Iodine"],
    ["OxygenXSDonorAcceptor", "Iron"],
    ["OxygenXSDonorAcceptor", "Manganese"],
    ["OxygenXSDonorAcceptor", "SulfurAcceptor"],
    ["Phosphorus", "Bromine"],
    ["Phosphorus", "Calcium"],
    ["Phosphorus", "Chlorine"],
    ["Phosphorus", "Fluorine"],
    ["Phosphorus", "GenericMetal"],
    ["Phosphorus", "Iodine"],
    ["Phosphorus", "Iron"],
    ["Phosphorus", "Manganese"],
    ["Phosphorus", "Phosphorus"],
    ["Phosphorus", "Zinc"],
    ["Sulfur", "Calcium"],
    ["Sulfur", "GenericMetal"],
    ["Sulfur", "Iodine"],
    ["Sulfur", "Iron"],
    ["Sulfur", "Magnesium"],
    ["Sulfur", "Manganese"],
    ["Sulfur", "SulfurAcceptor"],
    ["SulfurAcceptor", "Bromine"],
    ["SulfurAcceptor", "Calcium"],
    ["SulfurAcceptor", "Chlorine"],
    ["SulfurAcceptor", "Fluorine"],
    ["SulfurAcceptor", "GenericMetal"],
    ["SulfurAcceptor", "Iodine"],
    ["SulfurAcceptor", "Iron"],
    ["SulfurAcceptor", "Magnesium"],
    ["SulfurAcceptor", "Manganese"],
    ["SulfurAcceptor", "Phosphorus"],
    ["SulfurAcceptor", "SulfurAcceptor"],
    ["SulfurAcceptor", "Zinc"],
    ["Zinc", "Calcium"],
    ["Zinc", "GenericMetal"],
    ["Zinc", "Iron"],
    ["Zinc", "Zinc"],
]
all_pair_to_censible_pair = {}
for t1, t2 in censible_pairings:
    all_pair_to_censible_pair[(t1, t2)] = (t1, t2)
    all_pair_to_censible_pair[(t2, t1)] = (t1, t2)

all_pairwise_by_type_pair = {}

# Get all pairs of receptor/ligand atoms.
receptor_coords = receptor.get_coordinates()
ligand_coords = ligand.get_coordinates()
for ri, receptor_atom in enumerate(receptor.get_atoms()):
    receptor_coord = receptor_coords[ri]
    receptor_type = receptor_atom["type"]
    receptor_atom_radius = TYPES_TO_RADIUS[receptor_type]
    for li, ligand_atom in enumerate(ligand.get_atoms()):
        ligand_coord = ligand_coords[li]
        ligand_type = ligand_atom["type"]
        ligand_atom_radius = TYPES_TO_RADIUS[ligand_type]
        dist = np.linalg.norm(receptor_coord - ligand_coord)
        if dist > cutoff:
            continue

        # Calculate gaussian
        optimal_dist = receptor_atom_radius + ligand_atom_radius
        gauss = np.exp(-(dist - optimal_dist) ** 2)

        # Divide evenly between the two atoms of the pair
        gauss = 0.5 * gauss

        # Multiply by the scale factor
        pair_info = censible_output[
            type_pair_to_censible_key((receptor_type, ligand_type))
        ]
        gauss = gauss * pair_info["scale_factor"]

        # Multiple by the predicted weight to get the contribution.
        gauss = gauss * pair_info["predicted_weight"]

        # Add to the receptor atom
        receptor.add_atomic_gauss(ri, (receptor_type, ligand_type), gauss)

        # Add to the ligand atom
        ligand.add_atomic_gauss(li, (ligand_type, receptor_type), gauss)

# Now do a sanity check to make sure you've calculated Gaussian's correctly.
pair_sums = {}
for atom in receptor.get_atoms() + ligand.get_atoms():
    for pair in atom["type_pairs"]:
        if pair not in pair_sums:
            pair_sums[pair] = 0
        pair_sums[pair] += atom["type_pairs"][pair]
for pair in pair_sums:
    key = type_pair_to_censible_key(pair)
    censible_sum = censible_output[key]["predicted_contribution"]
    this_sum = round(pair_sums[pair], 5)
    # print(f"{this_sum} {censible_sum} {key}")

    # There are rounding errors, so allow for some wiggle room
    assert np.fabs(this_sum - censible_sum) <= 0.00004

pairs = [p for p in pair_sums.keys()]

# Sort by total_gauss
pairs = sorted(pairs, key=receptor.total_gauss, reverse=True)

for pair_idx, pair in enumerate(pairs):
    pair = all_pair_to_censible_pair[pair]
    receptor.save_pdb(
        f"{pair_idx + 1}.receptor_contribs_{pair[0]}_{pair[1]}.pdb",
        type_pairs_for_beta=pair,
        beta_scale=10,
    )
    ligand.save_pdb(
        f"{pair_idx + 1}.ligand_contribs_{pair[0]}_{pair[1]}.pdb",
        type_pairs_for_beta=pair,
        beta_scale=10,
    )

