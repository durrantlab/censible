import argparse
import re
from censible.data.get_data_paths import data_file_path
import molgrid
import subprocess
import torch
from censible.CEN_model import CENet
import random
import os
import numpy as np
import tempfile


def is_numeric(s: str) -> bool:
    """Return a boolean representing if the string s is a numeric string.
    
    Args:
        s (str): A string.
        
    Returns:
        A boolean representing if the string s is a numeric string.
    """

    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$", s))


# def fix_receptor_structure(filename: str, obabel_exec: str) -> str:
#     """Fix the protein structure.

#     Args:
#         filename (str): The path to the protein file.
#         obabel_exec (str): The path to the open babel executable.

#     Returns:
#         A string representing the path to the temporary file.
#     """

#     if os.path.exists(f"{filename}.converted.pdb"):
#         # For receptor, if converted already exists, don't recreate it.
#         return f"{filename}.converted.pdb"

#     # PDBBind data (used for training) had only polar hydrogens on the receptor,
#     # with carboxylates deprotonated. Unfortunately, as best I can tell, keeping
#     # only polar hydrogen atoms isn't an option using command line open babel.
#     # So I will convert to PDBQT and then back to PDB.

#     # First, convert to PDBQT to get polar hydrogens (pH 7)
#     subprocess.check_output(
#         f"{obabel_exec} {filename} -O {filename}.converted.pdbqt -p 7 -xr", shell=True
#     )

#     # Now convert back to PDB to make sure there are no assigned atomic charges
#     # (important that smina does that).
#     subprocess.check_output(
#         f"{obabel_exec} {filename}.converted.pdbqt -O {filename}.converted.pdb",
#         shell=True,
#     )

#     # Clean up intermediate file
#     os.remove(f"{filename}.converted.pdbqt")

#     return f"{filename}.converted.pdb"


# def fix_ligand_structure(filename: str, obabel_exec: str) -> str:
#     """Fix the ligand structure.
    
#     Args:
#         filename (str): The path to the ligand file.
#         obabel_exec (str): The path to the open babel executable.
        
#     Returns:
#         A string representing the path to the temporary file.
#     """

#     if os.path.exists(f"{filename}.converted.pdb"):
#         return f"{filename}.converted.pdb"

#     # PDBBind data (used for training) had all hydrogens on the ligands.
#     # Carboxylates, phosphates were protonated, but amines were too (so not just
#     # neutral form). Note that we trained on the SDF files, not the MOL2 files.
#     # MOL2 files had deprotonated carboxylates. Very unfortunate inconsistency.
#     # But we will make the ligands look like the sdf files by protonating at pH
#     # 0. This isn't perfect (sulphonates still not protoanted), but at least
#     # brings the ligands closer to the sdf files used for training.

#     # Note also that, strangely, -p doesn't work when converting from a pdbqt
#     # file, even if the target file can included non-polar hydrogens. So I'll
#     # convert to pdb first.

#     subprocess.check_output(
#         f"{obabel_exec} {filename} -O {filename}.converted.tmp.pdb -d", shell=True
#     )

#     subprocess.check_output(
#         f"{obabel_exec} {filename}.converted.tmp.pdb -O {filename}.converted.pdb -p 0",
#         shell=True,
#     )

#     # Clean up intermediate file
#     os.remove(f"{filename}.converted.tmp.pdb")

#     return f"{filename}.converted.pdb"


def load_example(
    lig_path: str,
    rec_path: str,
    smina_exec_path: str,
    smina_ordered_terms_names: np.ndarray,
    obabel_exec_path: str,
) -> molgrid.molgrid.ExampleProvider:
    """Load an example from a ligand and receptor path.
    
    Args:
        lig_path (str): A string representing the path to the ligand.
        rec_path (str): A string representing the path to the receptor.
        smina_exec_path (str): A string representing the path to the smina 
            executable.
        smina_ordered_terms_names (np.ndarray): A numpy array of strings 
            representing the names of all the terms.
        obabel_exec_path (str): A string representing the path to the open babel
            executable.
            
    Returns:
        A molgrid ExampleProvider.
    """

    # Standardize the molecules to make them more like the training set.
    # lig_path = fix_ligand_structure(lig_path, obabel_exec_path)
    # rec_path = fix_receptor_structure(rec_path, obabel_exec_path)

    # get CEN terms for proper termset
    # this is my smina path i neglected to append it
    custom_scoring_path = data_file_path("custom_scoring.txt")
    cmd = f"{smina_exec_path} --custom_scoring {custom_scoring_path} --score_only -r {rec_path} -l {lig_path} --seed 42"
    smina_out = str(subprocess.check_output(cmd, shell=True)).split("\\n")

    # It's critical to make sure the order is correct (could change with new version of smina).
    actual_ordered_terms_names = [l for l in smina_out if l.startswith("## Name")][0][
        8:
    ].split()
    for t1, t2 in zip(actual_ordered_terms_names, smina_ordered_terms_names):
        assert t1 == t2, f"terms not in correct order: {t1} != {t2}"

    # Get the computed terms as a string.
    line_with_terms = [l for l in smina_out if l.startswith("##")][-1]
    all_smina_computed_terms = line_with_terms.split()

    # Keep only those terms in all_smina_computed_terms that are numeric
    # (meaning they contain -, numbers, e, and .).
    all_smina_computed_terms = [t for t in all_smina_computed_terms if is_numeric(t)]
    all_smina_computed_terms_str = " ".join(all_smina_computed_terms)

    smina_outfile = f"types_file_cen.{random.randint(0, 1000000000)}.tmp"
    with open(smina_outfile, "w") as smina_out_f:
        smina_out_f.write(
            f"{all_smina_computed_terms_str} "
            + rec_path  # .split("a/")[-1]
            + " "
            + lig_path  # .split("a/")[-1]
        )

    example = molgrid.ExampleProvider(
        # data_root can be any directory, I think.
        # data_root="./",
        default_batch_size=1,
        # add_hydrogens=False,
    )
    example.populate(smina_outfile)

    # Delete the temporary file.
    os.remove(smina_outfile)

    return example


# load in model -- from torch
def load_model(model_dir: str):
    """Load the model, the smina terms mask, and the smina term scales.
    
    Args:
        model_dir (str): The path to the model directory.
        
    Returns:
        A tuple containing the model and other important data for applying the
        model to an example.
    """

    model_path = model_dir + os.sep + "model.pt"
    smina_terms_mask_path = model_dir + os.sep + "which_precalc_terms_to_keep.npy"
    smina_term_scales_path = model_dir + os.sep + "precalc_term_scales.npy"

    ### get the single_example_terms -- one set of smina computed terms
    # load normalization term data
    smina_terms_mask = np.load(smina_terms_mask_path)
    norm_factors_masked = np.load(smina_term_scales_path)

    dims = (28, 48, 48, 48)
    model = CENet(dims, len(norm_factors_masked))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    smina_ordered_terms_path = data_file_path("smina_ordered_terms.txt")
    with open(smina_ordered_terms_path) as f:
        smina_ordered_terms_names = f.read().strip().split()

    return (model, smina_terms_mask, norm_factors_masked, smina_ordered_terms_names)


# apply model to test data
def apply(
    example_data: molgrid.molgrid.ExampleProvider,
    smina_terms_mask: np.ndarray,
    smina_norm_factors_masked: np.ndarray,
    model: CENet,
    device="cuda",
):
    """Apply the model to the test data.
    
    Args:
        example_data (molgrid.molgrid.ExampleProvider): The example data.
        smina_terms_mask (np.ndarray): A boolean array representing which terms
            to keep.
        smina_norm_factors_masked (np.ndarray): The normalization factors for
            the terms.
        model (CENet): The model.
        device (str): The device to use. Defaults to "cuda".
        
    Returns:
        A tuple containing the predicted affinity, weights, contributions, etc.
    """

    smina_norm_factors_masked = torch.from_numpy(smina_norm_factors_masked).to(device)

    smina_terms_mask_trch = torch.from_numpy(smina_terms_mask).to(device)

    # Create tensors to store the precalculated terms and the input voxels.
    all_smina_terms = torch.zeros(
        (1, example_data.num_labels()), dtype=torch.float32, device=device
    )
    input_voxel = torch.zeros(
        (1,) + (28, 48, 48, 48), dtype=torch.float32, device=device
    )

    # Get this batch (just one example)
    test_batch = example_data.next_batch()

    # Get this batch's labels and put them in all_precalc_terms. This is all
    # labels, not just the one's you'll use.
    test_batch.extract_labels(all_smina_terms)

    # Now get only those precalculated terms you'll use.
    smina_terms_masked = all_smina_terms[:, :][:, smina_terms_mask_trch]

    # Populate the input_voxel tensor with the one example. Note that not using
    # random_translation and random_rotation keywords. Thus, this is
    # deterministic. Unlike during training, when you do add random translation
    # and rotation.
    gm = molgrid.GridMaker()

    # gm.set_resolution(0.1)
    gm.forward(test_batch, input_voxel)

    # save_all_channels(input_voxel)

    # print(input_voxel[0,5,4,32])

    scaled_smina_terms_masked = smina_terms_masked * smina_norm_factors_masked

    # print(scaled_smina_terms_masked)

    # Run that through the model.
    model.to(device)
    predicted_affinity, weights_predict, contributions_predict = model(
        input_voxel, scaled_smina_terms_masked
    )

    # print(weights_predict)
    # Round below to nearest 0.00001
    # print(contributions_predict[0,:15].cpu().detach().numpy().round(5))
        #   ) #.sum(dim=0))

    # weighted_terms = coef_predict * scaled_smina_terms_masked

    # scaled_smina_terms_masked = scaled_smina_terms_masked.cpu().detach().numpy()
    smina_terms_masked = smina_terms_masked.cpu().detach().numpy()[0]
    # smina_norm_factors_masked = smina_norm_factors_masked.cpu().detach().numpy()
    weights_predict = weights_predict.cpu().detach().numpy()[0]
    contributions_predict = contributions_predict.cpu().detach().numpy()[0]

    return (
        predicted_affinity,
        weights_predict,
        contributions_predict,
        smina_terms_masked,
    )


# run the model on test example
def get_cmd_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: command line arguments
    """

    # Create argparser
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--ligpath",
        required=True,
        nargs="+",
        help="path to the ligand(s) (PDB or PDBQT format)",
    )

    parser.add_argument(
        "--recpath", required=True, help="path to the receptor (PDB or PDBQT format)"
    )

    parser.add_argument(
        "--smina_exec_path", required=True, help="path to the smina executable"
    )

    parser.add_argument(
        "--obabel_exec_path", required=True, help="path to the open babel executable"
    )

    # Use store_true
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="use cpu (uses cuda by default, if not specified)",
    )

    # Optional parameters
    parser.add_argument(
        "--model_dir",
        default=None,
        help="path to a directory containing files such as model.pt, which_precalc_terms_to_keep.npy, etc.",
    )

    parser.add_argument("--out", default="", help="path to save output tsv file")

    args = parser.parse_args()

    # Do some validation
    # Check if ligpath exists
    for ligpath in args.ligpath:
        if not os.path.exists(ligpath):
            raise FileNotFoundError(f"{ligpath} does not exist")

    # Check if recpath exists
    if not os.path.exists(args.recpath):
        raise FileNotFoundError(f"{args.recpath} does not exist")

    # Check if smina_exec_path exists
    if not os.path.exists(args.smina_exec_path):
        raise FileNotFoundError(f"{args.smina_exec_path} does not exist")

    # If model_dir is not provided, use the default model_dir
    if args.model_dir is None:
        args.model_dir = data_file_path(f"model_allcen2{os.sep}")

    # ligpath must end in .pdb or .pdbqt
    for ligpath in args.ligpath:
        if not ligpath.endswith(".pdb") and not ligpath.endswith(".pdbqt"):
            raise ValueError(
                f"{ligpath} must end in .pdb or .pdbqt. Otherwise, can't guarantee partial charges haven't been precalculated. It is critical that smina be allowed to calculate partial charges."
            )

    # Same for recpath
    if not args.recpath.endswith(".pdb") and not args.recpath.endswith(".pdbqt"):
        raise ValueError(
            f"{args.recpath} must end in .pdb or .pdbqt. Otherwise, can't guarantee partial charges haven't been precalculated. It is critical that smina be allowed to calculate partial charges."
        )

    return args

