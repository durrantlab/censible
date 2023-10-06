import os
import numpy as np
from .PDBParser import PDBParser
from .consts import TYPES_TO_RADIUS, to_censible_key, to_censible_pair


def _get_close_recep_lig_pairs(
    receptor: PDBParser, ligand: PDBParser, cutoff: float = 8.0
) -> list:
    """Get all pairs of receptor/ligand atoms that are within a cutoff distance.
    
    Args:
        receptor (PDBParser): The receptor.
        ligand (PDBParser): The ligand.
        cutoff (float): The cutoff distance.

    Returns:
        list: A list of dicts representing the pairs.
    """
    # Get all pairs of receptor/ligand atoms.
    pairs = []
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
            pairs.append(
                {
                    "receptor_idx": ri,
                    "ligand_idx": li,
                    "receptor_type": receptor_type,
                    "ligand_type": ligand_type,
                    "receptor_atom_radius": receptor_atom_radius,
                    "ligand_atom_radius": ligand_atom_radius,
                    "dist": dist,
                }
            )
    return pairs


def _add_gauss_vals_to_atoms(
    receptor: PDBParser, ligand: PDBParser, censible_output: dict, pairs: list
):
    """Adds the Gaussian values to the atoms.
    
    Args:
        receptor (PDBParser): The receptor.
        ligand (PDBParser): The ligand.
        censible_output (dict): The censible output dictionary.
        pairs (list): The pairs.
    """
    for pair in pairs:
        # Calculate gaussian
        optimal_dist = pair["receptor_atom_radius"] + pair["ligand_atom_radius"]
        gauss = np.exp(-(pair["dist"] - optimal_dist) ** 2)

        # Divide evenly between the two atoms of the pair
        gauss = 0.5 * gauss

        # Multiply by the scale factor
        pair_info = censible_output[
            to_censible_key((pair["receptor_type"], pair["ligand_type"]))
        ]
        gauss = gauss * pair_info["scale_factor"]

        # Multiple by the predicted weight to get the contribution.
        gauss = gauss * pair_info["predicted_weight"]

        # Add to the receptor atom
        receptor.add_atomic_gauss(
            pair["receptor_idx"], (pair["receptor_type"], pair["ligand_type"]), gauss
        )

        # Add to the ligand atom
        ligand.add_atomic_gauss(
            pair["ligand_idx"], (pair["ligand_type"], pair["receptor_type"]), gauss
        )


def assign_gauss_vals(receptor: PDBParser, ligand: PDBParser, censible_output: dict):
    """Assigns Gaussian values to the atoms.
    
    Args:
        receptor (PDBParser): The receptor.
        ligand (PDBParser): The ligand.
        censible_output (dict): The censible output dictionary.
    """
    cutoff = 8

    receptor.keep_only_near_coords(ligand.get_coordinates(), cutoff=cutoff)

    close_pairs = _get_close_recep_lig_pairs(receptor, ligand, cutoff=cutoff)

    _add_gauss_vals_to_atoms(receptor, ligand, censible_output, close_pairs)


def calc_gauss_sums_per_pair(receptor: PDBParser, ligand: PDBParser) -> dict:
    """Calculates the gauss sums per pair (over all atoms).

    Args:
        receptor (PDBParser): The receptor.
        ligand (PDBParser): The ligand.

    Returns:
        dict: A dictionary mapping each pair to its summed Gaussian value.
    """
    sums_per_pair = {}
    for atom in receptor.get_atoms() + ligand.get_atoms():
        for pair in atom["type_pairs"]:
            if pair not in sums_per_pair:
                sums_per_pair[pair] = 0
            sums_per_pair[pair] += atom["type_pairs"][pair]
    return sums_per_pair


def verify_summed_gauss(sums_per_pair: dict, censible_output: dict):
    """Verifies the summed Gaussian values against CENsible's output.
    
    Args:
        sums_per_pair (dict): The summed Gaussian values.
        censible_output (dict): The censible output dictionary.
    """
    # Now do a sanity check to make sure you've calculated Gaussian's correctly.
    for pair in sums_per_pair:
        key = to_censible_key(pair)
        censible_sum = censible_output[key]["predicted_contribution"]
        this_sum = round(sums_per_pair[pair], 5)
        # print(f"{this_sum} {censible_sum} {key}")

        # There are rounding errors, so allow for some wiggle room
        assert np.fabs(this_sum - censible_sum) <= 0.00004


def save_pdbs(
    receptor: PDBParser,
    ligand: PDBParser,
    sums_per_pair: dict,
    out_path: str,
    predicted_affinity: float,
    beta_scale=10,
):
    """Saves the PDBs, with per-atom gauss values in the beta column.

    Args:
        receptor (PDBParser): The receptor.
        ligand (PDBParser): The ligand.
        sums_per_pair (dict): The summed Gaussian values.
        out_path (str): The path to save the PDB to.
        predicted_affinity (float): The CENsible-predicted affinity.
        beta_scale (float): The value to scale the Gaussian value by.
    """

    # Y, Z reserved for receptor and ligand
    chain_ids_to_use = [
        char for char in "ABCDEFGHIJKLMNOPQRSTUVWXabcdefghijklmnopqrstuvwxyz"
    ]

    pairs = [p for p in sums_per_pair.keys()]

    # Sort by total_gauss
    pairs = sorted(pairs, key=receptor.total_gauss, reverse=True)

    header_remark = "REMARK CHAIN  PAIR                                                          CONTRIBUTION\n"
    pairs_pdb_txt = ""

    for pair_idx, pair in enumerate(pairs):
        pair = to_censible_pair(pair)
        total_gauss = receptor.total_gauss(pair) + ligand.total_gauss(pair)
        chain_id = chain_ids_to_use[pair_idx]
        chain_pair = f"{pair[0]} - {pair[1]}"
        header_remark += (
            f"REMARK {chain_id:5s}  {chain_pair:63s} {total_gauss:>10.5f}\n"
        )
        pairs_pdb_txt += f"REMARK ATOM-TYPE PAIR: {pair[0]} - {pair[1]}\n"
        pairs_pdb_txt += f"REMARK CONTRIBUTION TO SCORE: {total_gauss}\n"
        pairs_pdb_txt += receptor.get_pdb_text(
            type_pairs_for_beta=pair, beta_scale=beta_scale, chain_id=chain_id
        )
        pairs_pdb_txt += "TER\n"
        pairs_pdb_txt += ligand.get_pdb_text(
            type_pairs_for_beta=pair, beta_scale=beta_scale, chain_id=chain_id
        )
        pairs_pdb_txt += "TER\n\n"

    # Get the origianl receptor and ligand files
    receptor_pdb_txt = receptor.get_from_orig_pdb("Y")
    ligand_pdb_txt = ligand.get_from_orig_pdb("Z")

    pairs_pdb_txt = f"""REMARK CENSible OUTPUT

REMARK CENsible-predicted contributions associated with the 
REMARK atom_type_gaussian smina terms can be attributed to specific receptor
REMARK and ligand atoms. This PDB file includes the per-atom contributions in
REMARK the beta columns, scaled by {beta_scale}.

REMARK For reference, the CENsible score for this complex is {float(predicted_affinity):.5f}.
REMARK Summing the contributions associated with the atom_type_gaussian smina
REMARK terms shown below gives only {sum(sums_per_pair.values()):.5f}.

{header_remark.strip()}

REMARK RECEPTOR
{receptor_pdb_txt.strip()}
TER

REMARK LIGAND
{ligand_pdb_txt.strip()}
TER

{pairs_pdb_txt}
"""

    with open(out_path, "w") as f:
        f.write(pairs_pdb_txt)
    
    # For convenience sake, also save the receptor and ligand as a single file.
    with open(out_path.replace(".pdb", ".rec_lig_only.pdb"), "w") as f:
        f.write(receptor_pdb_txt + "\nTER\n" + ligand_pdb_txt + "\nTER\n")

