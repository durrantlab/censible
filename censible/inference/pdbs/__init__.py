import glob
import os
from .recep_lig_complex import (
    assign_gauss_vals,
    calc_gauss_sums_per_pair,
    save_pdbs,
    verify_summed_gauss,
)
from .PDBParser import PDBParser

# NOTE: Smina type assignments 100% depend on protonation states. Confirmed with
# this project.


def load_censible_output(censible_output_contents: str) -> dict:
    """Loads the CENsible output file.

    Args:
        censible_output_contents (str): The contents of the CENsible output
            file.

    Returns:
        dict: A dictionary mapping each term to its CENsible output.
    """
    lines = censible_output_contents.split("\n")
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


def _fix_path(path: str) -> str:
    """Fixes the path to point to the right structure file (with proper
    protonation state, etc.).

    Args:
        path (str): The path.

    Returns:
        str: The fixed path.
    """
    if ".censible.converted." in path:
        # It's already correct
        return path

    # It's not correct. Fix it.
    correct_paths = glob.glob(path + ".censible.converted.*")

    if len(correct_paths) == 0:
        raise Exception(f"Could not find the converted file for {path}")

    if len(correct_paths) > 1:
        raise Exception(f"Found multiple converted files for {path}")

    return correct_paths[0]


def save_pdbs_with_per_atom_gauss_vals_in_beta(
    censible_output_contents: str,
    predicted_affinity: float,
    smina_exec_path: str,
    obabel_exec_path: str,
    lig_path: str,
    rec_path: str,
    out_path: str,
):
    """Saves the PDBs, with per-atom gauss values in the beta columns.

    Args:
        censible_output_contents (str): The contents of the CENsible output
            file.
        predicted_affinity (float): The predicted affinity.
        smina_exec_path (str): The path to the smina executable.
        obabel_exec_path (str): The path to the obabel executable.
        lig_path (str): The path to the ligand PDB.
        rec_path (str): The path to the receptor PDB.
        out_path (str): The path to save the PDBs to.
    """
    rec_path = _fix_path(rec_path)
    lig_path = _fix_path(lig_path)

    receptor = PDBParser(rec_path, smina_exec_path, obabel_exec_path)
    ligand = PDBParser(lig_path, smina_exec_path, obabel_exec_path)

    censible_output = load_censible_output(censible_output_contents)
    assign_gauss_vals(receptor, ligand, censible_output)
    sums_per_pair = calc_gauss_sums_per_pair(receptor, ligand)
    verify_summed_gauss(sums_per_pair, censible_output)
    save_pdbs(
        receptor,
        ligand,
        sums_per_pair,
        out_path,
        predicted_affinity,
    )
