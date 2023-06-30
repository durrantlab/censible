import molgrid

# from openbabel import pybel
# import json
import argparse
import torch
import numpy as np
import subprocess
import os
import re
import random

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
    which_precalc_terms_mask = np.load(which_precalc_terms_to_keep)
    norm_factors_to_keep = np.load(precalc_term_scales)

    # Get the path to the custom_scoring.txt file. It is in the same directory
    # as this python script.
    custom_scoring_path = os.path.dirname(os.path.realpath(__file__)) + os.sep + "custom_scoring.txt"

    # get CEN terms for proper termset
    # this is my smina path i neglected to append it
    cmd = f"{smina_exec_path} --custom_scoring {custom_scoring_path} --score_only -r {rec_datapath} -l {lig_datapath}"
    smina_out = str(subprocess.check_output(cmd, shell=True)).split("\\n")

    # It's critical to make sure the order is correct (could change with new version of smina).
    ordered_terms_names = [l for l in smina_out if l.startswith("## Name")][0][8:].split()
    smina_ordered_terms_path = os.path.dirname(os.path.realpath(__file__)) + os.sep + "smina_ordered_terms.txt"
    with open(smina_ordered_terms_path) as f:
        smina_ordered_terms_names = f.read().strip().split()
    for t1, t2 in zip(ordered_terms_names, smina_ordered_terms_names):
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

    smina_outfile = "types_file_cen." + str(random.randint(0, 1000000000)) + ".tmp"
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

    # Delete the temporary file.
    os.remove(smina_outfile)

    ordered_terms_names_to_keep = np.array(ordered_terms_names)[which_precalc_terms_mask]

    return (example, which_precalc_terms_mask, norm_factors_to_keep, ordered_terms_names_to_keep)


# load in model -- from torch
def load_model(modelpath, num_terms):
    dims = (28, 48, 48, 48)
    model = CENet(dims, num_terms)
    model.load_state_dict(torch.load(modelpath))
    model.eval()

    return model


# apply model to test data
def test_apply(example_data, which_precalc_terms_mask, precalc_terms_to_use_norm_factors, model):
    precalc_terms_to_use_norm_factors = torch.from_numpy(precalc_terms_to_use_norm_factors).to("cuda")

    which_precalc_terms_to_keep_torch = torch.from_numpy(which_precalc_terms_mask).to(
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
    precalc_terms_to_use = all_precalc_terms[:, :][:, which_precalc_terms_to_keep_torch]

    # Populate the input_voxel tensor with the one example. Note that not using
    # random_translation and random_rotation keywords. Thus, this is
    # deterministic. Unlike during training, when you do add random translation
    # and rotation.
    gm = molgrid.GridMaker()
    gm.forward(test_batch, input_voxel)

    # print("running model to predict")

    scaled_precalc_terms_to_use = precalc_terms_to_use * precalc_terms_to_use_norm_factors

    # Run that through the model.
    model.to("cuda")
    predicted_affinity, weights_predict, contributions_predict = model(
        input_voxel, scaled_precalc_terms_to_use
    )

    # weighted_terms = coef_predict * scaled_precalc_terms_to_use

    # scaled_precalc_terms_to_use = scaled_precalc_terms_to_use.cpu().detach().numpy()
    precalc_terms_to_use = precalc_terms_to_use.cpu().detach().numpy()
    # precalc_terms_to_use_norm_factors = precalc_terms_to_use_norm_factors.cpu().detach().numpy()
    weights_predict = weights_predict.cpu().detach().numpy()
    contributions_predict = contributions_predict.cpu().detach().numpy()

    return (predicted_affinity, weights_predict, contributions_predict, precalc_terms_to_use)

def get_numeric_val(s: str, varname: str) -> str:
    # v is a number, so only digits, +/-, and .
    num_regex = "([e0-9\.\+\-]+)"
    v = re.search(varname + "=" + num_regex, s)
    v = v.group(1) if v is not None else "?"
    return v


def full_term_description(term: str) -> str:
    # Given that making explainable scoring functions is the goal, good to
    # provide more complete description of the terms. This function tries to do
    # so semi-programatically.

    # TODO: Good to have David review this?

    desc = ""

    # Many terms share variables in common. Let's extract those first.
    o = get_numeric_val(term, "o")
    _w = get_numeric_val(term, "_w")
    _c = get_numeric_val(term, "_c")
    g = get_numeric_val(term, "g")
    _b = get_numeric_val(term, "_b")

    if term.startswith("atom_type_gaussian("):
        # Looks like atom_type_gaussian(t1=AliphaticCarbonXSHydrophobe,t2=AliphaticCarbonXSNonHydrophobe,o=0,_w=1,_c=8)
        # Extract t1, t2, o, _w, _c using regex.
        t1 = re.search("t1=(.*?),", term)
        t1 = t1.group(1) if t1 is not None else None

        t2 = re.search("t2=(.*?),", term)
        t2 = t2.group(1) if t2 is not None else None

        desc = f"adjacent atoms: {t1}-{t2}; offset: {o}; gaussian width: {_w}; distance cutoff: {_c}"
    elif term.startswith("gauss("):
        desc = f"sterics (vina): offset: {o}; gaussian width: {_w}; distance cutoff: {_c}; see PMC3041641"
    elif term.startswith("repulsion("):
        desc = f"repulsion (vina): offset: {o}; distance cutoff: {_c}; see PMC3041641"
    elif term.startswith("hydrophobic("):
        desc = f"hydrophobic (vina): good-distance cutoff: {g}; bad-distance cutoff: {_b}; distance cutoff: {_c}; see PMC3041641"
    elif term.startswith("non_hydrophobic("):
        desc = f"non-hydrophobic: good-distance cutoff: {g}; bad-distance cutoff: {_b}; distance cutoff: {_c}; see ???"
    elif term.startswith("vdw("):
        i = get_numeric_val(term, "i")
        _j = get_numeric_val(term, "_j")
        _s = get_numeric_val(term, "_s")
        _ = get_numeric_val(term, "_\^")

        desc = f"vdw: Lennard-Jones exponents (AutoDock 4): {i}, {_j}; smoothing: {_s}; cap: {_}; distance cutoff: {_c}; see PMID17274016"

    elif term.startswith("non_dir_h_bond("):
        desc = f"non-directional hydrogen bond (vina): good-distance cutoff: {g}; bad-distance cutoff: {_b}; distance cutoff: {_c}; see PMC3041641"
    
    elif term.startswith("non_dir_anti_h_bond_quadratic"):
        desc = f"mimics repulsion between polar atoms that can't hydrogen bond: offset: {o}; distance cutoff: {_c}; see ???"
    elif term.startswith("non_dir_h_bond_lj("):
        _ = get_numeric_val(term, "_\^")
        desc = f"10-12 Lennard-Jones potential (AutoDock 4) : {o}; cap: {_}; distance cutoff: {_c}; see PMID17274016"
    elif term.startswith("acceptor_acceptor_quadratic("):
        desc = "quadratic potential (see repulsion) between two acceptor atoms: offset: {o}; distance cutoff: {_c}; see ???"
    elif term.startswith("donor_donor_quadratic("):
        desc = "quadratic potential (see repulsion) between two donor atoms: offset: {o}; distance cutoff: {_c}; see ???"
    elif term.startswith("ad4_solvation("):
        dsig = get_numeric_val(term, "d-sigma")
        sq = get_numeric_val(term, "_s/q")

        desc = f"desolvation (AutoDock 4): d-sigma: {dsig}; _s/q: {sq}; distance cutoff: {_c}; see PMID17274016"
    elif term.startswith("electrostatic("):
        i = get_numeric_val(term, "i")
        _ = get_numeric_val(term, "_\^")
        desc = f"electrostatics (AutoDock 4): distance exponent: {i}; cap: {_}; distance cutoff: {_c}; see PMID17274016"
    elif term == "num_heavy_atoms":
        desc = "number of heavy atoms"
    elif term == "num_tors_add":
        desc = "loss of torsional entropy upon binding (AutoDock 4); see PMID17274016"
    elif term == "num_hydrophobic_atoms":
        desc = "number of hydrophobic atoms"
    elif term == "ligand_length":
        desc = "lenght of the ligand"
    elif term in ["num_tors_sqr", "num_tors_sqrt"]:
        desc = "meaning uncertain"  # TODO: Ask David?
    else:
        desc = "error: unknown term"

    return desc


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
    parser.add_argument("--out", default="", help="path to save output tsv file")

    return parser.parse_args()


# Usage: python apply_model.py --ligpath <path to ligand file>
#           --recpath <path to receptor file>
#           --model_dir <path to model directory>
#           --smina_exec_path <path to smina executable>
#           --out <path to save output tsv file>

args = get_cmd_args()

for lig in args.ligpath:
    # load the data
    example, which_precalc_terms_mask, norm_factors_to_keep, ordered_terms_names_to_keep = load_example(
        lig,
        args.recpath,
        args.model_dir + os.sep + "which_precalc_terms_to_keep.npy",
        args.model_dir + os.sep + "precalc_term_scales.npy",
        args.smina_exec_path,
    )

    # load the model
    model = load_model(args.model_dir + os.sep + "model.pt", len(norm_factors_to_keep))

    # print("data and model loaded.")
    predicted_affinity, weights_predict, contributions_predict, precalc_terms_to_use = test_apply(
        example, which_precalc_terms_mask, norm_factors_to_keep, model
    )

    tsv_output = f"receptor\t{args.recpath}\n"
    tsv_output += f"ligand\t{lig}\n\n"
    tsv_output += f"predicted_affinity\t{str(round(float(predicted_affinity), 5))}\n\n"

    print(tsv_output)

    if args.out != "":
        # If specifying an output file, provide additional information and save.
        tsv_output += "\t" + "\t".join(ordered_terms_names_to_keep) + "\n"

        tsv_output += "\t" + "\t".join(
            [full_term_description(t) for t in ordered_terms_names_to_keep]
        ) + "\n"

        for name in ordered_terms_names_to_keep:
            full_term_description(name)


        # TODO: Some require [0], others don't. Why? Seems disorganized.
        tsv_output += "precalc_smina_terms\t" + "\t".join(
            [str(round(x, 5)) for x in precalc_terms_to_use[0]]
        ) + "\n"
        
        # tsv_output += "Precalc-term normalization scales\t" + "\t".join(
        #     [str(round(x, 5)) for x in norm_factors_to_keep]
        # ) + "\n"
        
        # import pdb ;pdb.set_trace()
        tsv_output += "normalized_precalc_smina_terms\t" + "\t".join(
            [str(round(x, 5)) for x in precalc_terms_to_use[0] * norm_factors_to_keep]
        ) + "\n"
        tsv_output += "predicted_weights\t" + "\t".join([str(round(x, 5)) for x in weights_predict[0]]) + "\n"
        tsv_output += "predicted_contributions\t" + "\t".join(
            [str(round(x, 5)) for x in contributions_predict[0]]
        ) + "\n"


        with open(args.out, "w") as f:
            # Report the receptor/ligand:
            f.write(tsv_output)

    # if args.out_prefix == "":
    #     prefix = (
    #         lig.split("/")[-1].split(".")[-2]
    #         + "_"
    #         + args.recpath.split("/")[-1].split(".")[-2]
    #     )
    # else:
    #     prefix = args.out_prefix

    # if args.out_prefix != "":
    #     prefix = args.out_prefix
    #     np.savetxt(f"{prefix}_coeffs.txt", weights_predict.cpu().detach().numpy())
    #     np.savetxt(f"{prefix}_weighted.txt", contributions_predict.cpu().detach().numpy())

    #     print(f"Coefficients saved in: {prefix}_coeffs.txt")
    #     print(f"Weighted terms saved in: {prefix}_weighted.txt")
