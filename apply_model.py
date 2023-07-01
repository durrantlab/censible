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
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$", s))


def load_example(
    lig_path: str,
    rec_path: str,
    smina_exec_path: str,
    smina_terms_mask,
    smina_ordered_terms_names,
):
    # get CEN terms for proper termset
    # this is my smina path i neglected to append it
    cmd = f"{smina_exec_path} --custom_scoring {custom_scoring_path} --score_only -r {rec_path} -l {lig_path}"
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

    smina_outfile = "types_file_cen." + str(random.randint(0, 1000000000)) + ".tmp"
    with open(smina_outfile, "w") as smina_out_f:
        smina_out_f.write(
            f"{all_smina_computed_terms_str} "
            + rec_path  # .split("a/")[-1]
            + " "
            + lig_path  # .split("a/")[-1]
        )

    example = molgrid.ExampleProvider(
        # data_root can be any directory, I think.
        data_root="./",
        default_batch_size=1,
    )
    example.populate(smina_outfile)

    # Delete the temporary file.
    os.remove(smina_outfile)

    return example


# load in model -- from torch
def load_model(
    model_path: str, smina_terms_mask_path: str, smina_term_scales_path: str
):
    ### get the single_example_terms -- one set of smina computed terms
    # load normalization term data
    smina_terms_mask = np.load(smina_terms_mask_path)
    norm_factors_masked = np.load(smina_term_scales_path)

    dims = (28, 48, 48, 48)
    model = CENet(dims, len(norm_factors_masked))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Get the path to the custom_scoring.txt file. It is in the same directory
    # as this python script.
    custom_scoring_path = (
        os.path.dirname(os.path.realpath(__file__)) + os.sep + "custom_scoring.txt"
    )

    smina_ordered_terms_path = (
        os.path.dirname(os.path.realpath(__file__)) + os.sep + "smina_ordered_terms.txt"
    )
    with open(smina_ordered_terms_path) as f:
        smina_ordered_terms_names = f.read().strip().split()

    return (
        model,
        smina_terms_mask,
        norm_factors_masked,
        custom_scoring_path,
        smina_ordered_terms_names,
    )


# apply model to test data
def test_apply(example_data, smina_terms_mask, smina_norm_factors_masked, model):
    smina_norm_factors_masked = torch.from_numpy(smina_norm_factors_masked).to("cuda")

    smina_terms_mask_trch = torch.from_numpy(smina_terms_mask).to("cuda")

    # Create tensors to store the precalculated terms and the input voxels.
    all_smina_terms = torch.zeros(
        (1, example_data.num_labels()), dtype=torch.float32, device="cuda"
    )
    input_voxel = torch.zeros(
        (1,) + (28, 48, 48, 48), dtype=torch.float32, device="cuda"
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
    gm.forward(test_batch, input_voxel)

    # print("running model to predict")

    scaled_smina_terms_masked = smina_terms_masked * smina_norm_factors_masked

    # Run that through the model.
    model.to("cuda")
    predicted_affinity, weights_predict, contributions_predict = model(
        input_voxel, scaled_smina_terms_masked
    )

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
    parser.add_argument(
        "--ligpath", required=True, nargs="+", help="path to the ligand(s)"
    )
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

# load the model
model, smina_terms_mask, norm_factors_masked, custom_scoring_path, smina_ordered_terms_names = load_model(
    args.model_dir + os.sep + "model.pt",
    args.model_dir + os.sep + "which_precalc_terms_to_keep.npy",
    args.model_dir + os.sep + "precalc_term_scales.npy",
)

tsv_output = ""

bar = "=====================================\n"

print("")

for lig_path in args.ligpath:
    terminal_output = ""

    # Load the data. TODO: One ligand at a time here for simplicity's sake.
    # Could batch to improve speed, I think.
    example = load_example(
        lig_path,
        args.recpath,
        args.smina_exec_path,
        smina_terms_mask,
        smina_ordered_terms_names,
    )

    predicted_affinity, weights_predict, contributions_predict, smina_terms_masked = test_apply(
        example, smina_terms_mask, norm_factors_masked, model
    )

    terminal_output += f"receptor\t{args.recpath}\n"
    terminal_output += f"ligand\t{lig_path}\n\n"
    terminal_output += f"predicted_affinity\t{str(round(float(predicted_affinity), 5))}\n"

    print(terminal_output)

    if args.out != "":
        print("See " + args.out + " for predicted weights and contributions.")
    else:
        print("WARNING: No output file specified (--out). Not saving weights and contributions.")

    print("\n" + bar)

    tsv_output += terminal_output

    if args.out != "":
        # If you're going to print out the specific terms, you need to get the
        # names of only those in the mask.
        smina_ordered_terms_names_masked = np.array(smina_ordered_terms_names)[
            smina_terms_mask
        ]

        # If specifying an output file, provide additional information and save.
        tsv_output += "\t" + "\t".join(smina_ordered_terms_names_masked) + "\n"

        tsv_output += (
            "\t"
            + "\t".join(
                [full_term_description(t) for t in smina_ordered_terms_names_masked]
            )
            + "\n"
        )

        for name in smina_ordered_terms_names_masked:
            full_term_description(name)

        tsv_output += (
            "precalc_smina_terms\t"
            + "\t".join([str(round(x, 5)) for x in smina_terms_masked])
            + "\n"
        )

        # tsv_output += "Precalc-term normalization scales\t" + "\t".join(
        #     [str(round(x, 5)) for x in norm_factors_masked]
        # ) + "\n"

        # import pdb ;pdb.set_trace()
        tsv_output += (
            "normalized_precalc_smina_terms\t"
            + "\t".join(
                [
                    str(round(x, 5))
                    for x in smina_terms_masked * norm_factors_masked
                ]
            )
            + "\n"
        )
        tsv_output += (
            "predicted_weights\t"
            + "\t".join([str(round(x, 5)) for x in weights_predict])
            + "\n"
        )
        tsv_output += (
            "predicted_contributions\t"
            + "\t".join([str(round(x, 5)) for x in contributions_predict])
            + "\n\n"
        )

        tsv_output += bar

if args.out != "":
    with open(args.out, "w") as f:
        # Report the receptor/ligand:
        f.write(tsv_output)