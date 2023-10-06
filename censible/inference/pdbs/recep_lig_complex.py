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
    beta_scale=10,
):
    """Saves the PDBs, with per-atom gauss values in the beta column.

    Args:
        receptor (PDBParser): The receptor.
        ligand (PDBParser): The ligand.
        sums_per_pair (dict): The summed Gaussian values.
        out_path (str): The path to save the PDBs to.
        beta_scale (float): The value to scale the Gaussian value by.
    """

    pairs = [p for p in sums_per_pair.keys()]

    # Sort by total_gauss
    pairs = sorted(pairs, key=receptor.total_gauss, reverse=True)

    for pair_idx, pair in enumerate(pairs):
        pair = to_censible_pair(pair)
        receptor.save_pdb(
            os.path.join(
                out_path, f"{pair_idx + 1}.receptor_contribs_{pair[0]}_{pair[1]}.pdb"
            ),
            type_pairs_for_beta=pair,
            beta_scale=beta_scale,
        )
        ligand.save_pdb(
            os.path.join(
                out_path, f"{pair_idx + 1}.ligand_contribs_{pair[0]}_{pair[1]}.pdb"
            ),
            type_pairs_for_beta=pair,
            beta_scale=beta_scale,
        )

