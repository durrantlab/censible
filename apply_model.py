import molgrid

# from openbabel import pybel
# import json
import argparse
import torch
import numpy as np
import subprocess
import os
import re

from CEN_model import CENet

def is_numeric(s):
    """Return a boolean representing if the string s is a numeric string."""
    return bool(re.match(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$', s))

def load_example(
    lig_datapath,
    rec_datapath,
    which_precalc_terms_to_keep,
    precalc_term_scales,
    smina_exec_path,
):
    ### get the single_example_terms -- one set of smina computed terms
    # load normalization term data
    which_precalc_terms = np.load(which_precalc_terms_to_keep)
    norm_factors_to_keep = np.load(precalc_term_scales)

    # Get the path to the custom_scoring.txt file. It is in the same directory
    # as this python script.
    custom_scoring_path = os.path.dirname(os.path.realpath(__file__)) + os.sep + "custom_scoring.txt"

    # get CEN terms for proper termset
    # this is my smina path i neglected to append it
    cmd = f"{smina_exec_path} --custom_scoring {custom_scoring_path} --score_only -r {rec_datapath} -l {lig_datapath}"
    smina_out = str(subprocess.check_output(cmd, shell=True)).split("\\n")

    # It's critical to make sure the order is correct (could change with new version of smina).
    ordered_terms = [l for l in smina_out if l.startswith("## Name")][0][8:].split()
    smina_ordered_terms_path = os.path.dirname(os.path.realpath(__file__)) + os.sep + "smina_ordered_terms.txt"
    with open(smina_ordered_terms_path) as f:
        smina_ordered_terms = f.read().strip().split()
    for t1, t2 in zip(ordered_terms, smina_ordered_terms):
        assert t1 == t2, f"terms not in correct order: {t1} != {t2}"

    # Get the computed terms as a string.
    line_with_terms = [l for l in smina_out if l.startswith("##")][-1]
    all_smina_computed_terms = line_with_terms.split()

    # Keep only those terms in all_smina_computed_terms that are numeric
    # (meaning they contain -, numbers, e, and .).
    all_smina_computed_terms = [
        t for t in all_smina_computed_terms if is_numeric(t)
    ]

    all_smina_computed_terms_str = " ".join(
        all_smina_computed_terms
    )

    smina_outfile = "types_file_cen.tmp"
    with open(smina_outfile, "w") as smina_out_f:
        smina_out_f.write(
            f"{all_smina_computed_terms_str} "
            + rec_datapath # .split("a/")[-1]
            + " "
            + lig_datapath # .split("a/")[-1]
        )

    example = molgrid.ExampleProvider(
        # data_root can be any directory, I think.
        data_root="./",
        default_batch_size=1,
    )
    example.populate(smina_outfile)

    return (example, which_precalc_terms, norm_factors_to_keep)


# load in model -- from torch
def load_model(modelpath, num_terms):
    dims = (28, 48, 48, 48)
    model = CENet(dims, num_terms)
    model.load_state_dict(torch.load(modelpath))
    model.eval()

    return model


# apply model to test data
def test_apply(example_data, which_precalc_terms_to_keep, norm_factors, model):
    gm = molgrid.GridMaker()
    norm_factors = torch.from_numpy(norm_factors).to("cuda")
    which_precalc_terms_to_keep = torch.from_numpy(which_precalc_terms_to_keep).to(
        "cuda"
    )

    # Create tensors to store the precalculated terms and the input voxels.
    all_precalc_terms = torch.zeros(
        (1, example_data.num_labels()), dtype=torch.float32, device="cuda"
    )
    input_voxel = torch.zeros(
        (1,) + (28, 48, 48, 48), dtype=torch.float32, device="cuda"
    )

    # Get this batch (just one example)
    test_batch = example_data.next_batch()

    # Get this batch's labels and put them in all_precalc_terms. This is all
    # labels, not just the one's you'll use.
    test_batch.extract_labels(all_precalc_terms)

    # Now get only those precalculated terms you'll use.
    precalc_terms_to_use = all_precalc_terms[:, :][:, which_precalc_terms_to_keep]

    # Populate the input_voxel tensor with the one example.
    gm.forward(test_batch, input_voxel)

    model.to("cuda")

    # print("running model to predict")

    # Run that through the model.
    output, coef_predict, weighted_terms = model(
        input_voxel, precalc_terms_to_use * norm_factors
    )

    return (output, coef_predict, weighted_terms)


# run the model on test example
def get_cmd_args():
    # Create argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--ligpath", required=True, nargs='+', help="path to the ligand(s)")
    parser.add_argument("--recpath", required=True, help="path to the receptor")
    parser.add_argument(
        "--model_dir",
        default="./",
        help="path to a directory containing files such as model.pt, which_precalc_terms_to_keep.npy, etc.",
    )

    parser.add_argument("--smina_exec_path", help="path to the smina executable")
    parser.add_argument("--out_prefix", default="", help="prefix to use for saving")

    return parser.parse_args()


# Usage: python apply_model.py --ligpath <path to ligand file>
#           --recpath <path to receptor file>
#           --model_dir <path to model directory>
#           --smina_exec_path <path to smina executable>
#           --out_prefix <prefix to use for saving>

args = get_cmd_args()

for lig in args.ligpath:
    # load the data
    example, which_precalc_terms_to_keep, norm_factors = load_example(
        lig,
        args.recpath,
        args.model_dir + os.sep + "which_precalc_terms_to_keep.npy",
        args.model_dir + os.sep + "precalc_term_scales.npy",
        args.smina_exec_path,
    )

    # load the model
    model = load_model(args.model_dir + os.sep + "model.pt", len(norm_factors))

    # print("data and model loaded.")
    prediction, coef_predictions, weighted_ind_terms = test_apply(
        example, which_precalc_terms_to_keep, norm_factors, model
    )

    print(f"Affinity prediction: {str(round(float(prediction), 5))} {args.recpath} {lig}")

    # if args.out_prefix == "":
    #     prefix = (
    #         lig.split("/")[-1].split(".")[-2]
    #         + "_"
    #         + args.recpath.split("/")[-1].split(".")[-2]
    #     )
    # else:
    #     prefix = args.out_prefix

    if args.out_prefix != "":
        prefix = args.out_prefix
        np.savetxt(f"{prefix}_coeffs.txt", coef_predictions.cpu().detach().numpy())
        np.savetxt(f"{prefix}_weighted.txt", weighted_ind_terms.cpu().detach().numpy())

        print(f"Coefficients saved in: {prefix}_coeffs.txt")
        print(f"Weighted terms saved in: {prefix}_weighted.txt")
