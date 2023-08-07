from _training import train_single_fold
from _outputs import generate_outputs
import numpy as np
import json
import argparse
import torch
import datetime

# Published performance for this model on this set is 1.5 RMSE and 0.7 Pearson's
# R, so we are pretty close (could train longer).
# from published_model import Net

from CEN_model import CENet
from _preprocess import preprocess

# import py3Dmol
import os

params = [
    {
        "name": "epochs",
        "val": 250,  # 400,
        "description": "Number of epochs to train for.",
    },
    {"name": "fold_num", "val": 0, "description": "Which fold to train on."},
    {"name": "batch_size", "val": 25, "description": "Batch size."},
    {"name": "lr", "val": 0.01, "description": "Learning rate."},
    {
        "name": "step_size",
        "val": 80,
        "description": "Step size for learning rate decay.",
    },
    {
        "name": "prefix",
        "val": "randomsplit",
        "description": "Prefix for the input types files.",  # TODO: Correct description?
    },
    {
        "name": "termtypes",
        "val": "all",
        "description": "Which terms to use. Can be 'all', 'smina', or 'gaussian'.",
    },
    {
        "name": "data_dir",
        "val": "./prepare_data/",
        "description": "Directory where the data is stored.",
    },
]
# "prefix": "crystal",

# Create argparser with same args as params
parser = argparse.ArgumentParser()
for value in params:
    # parser.add_argument("--" + key, type=type(value), default=value)
    parser.add_argument(
        "--" + value["name"],
        type=type(value["val"]),
        default=value["val"],
        help=value["description"] + " Default: " + str(value["val"]),
    )
args = parser.parse_args()
params = vars(args)

# Make sure termtypes is valid
if params["termtypes"] not in ["all", "smina", "gaussian"]:
    raise ValueError("termtypes must be 'all', 'smina', or 'gaussian'")

print(params)

orig_dir = os.getcwd() + os.sep

# change working directory to "./prepare_data/"
os.chdir(params["data_dir"])

# which_precalc_terms_to_keep is a boolean array, True if a given feature is worth
# keeping, False otherwise. term_names is a list of all the term names.
which_precalc_terms_to_keep, term_names, precalc_term_scales = preprocess(
    params["termtypes"]
)

print("Preprocessing done.")

# Train the model
(
    model,
    test_labels,
    test_results,
    gninatypes_filenames,
    test_mses,
    test_ames,
    test_pearsons,
    training_losses,
    test_coefs_predict_lst,
    test_weighted_terms_lst,
    which_precalc_terms_to_keep,
    precalc_term_scales_to_keep,
) = train_single_fold(
    CENet, which_precalc_terms_to_keep, params, term_names, precalc_term_scales
)

# Save model
# Create the directory if it doesn't exist
if not os.path.exists(f"{orig_dir}imgs"):
    os.mkdir(f"{orig_dir}imgs")

# Get the current date and time as a string, in a format that can be a filename
now = datetime.datetime.now()
now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

# Define and create the save directory
save_dir = orig_dir + "imgs/" + now_str + "/"
os.mkdir(save_dir)

# Make report subdirectory
report_subdir = save_dir + "report/"
os.mkdir(report_subdir)

# Save the model
torch.save(model.state_dict(), save_dir + "model.pt")

# Save a boolean list of which precalculated terms are used in this model
np.save(save_dir + "which_precalc_terms_to_keep.npy", which_precalc_terms_to_keep)
with open(report_subdir + "which_precalc_terms_to_keep.txt", "w") as f:
    f.write(str(which_precalc_terms_to_keep))

# Save the names of the retained terms
with open(report_subdir + "term_names.txt", "w") as f:
    f.write(
        str(
            [
                term_names[i]
                for i in range(len(term_names))
                if which_precalc_terms_to_keep[i]
            ]
        )
    )

# Save the normalization factors applied to the retained precalculated terms.
np.save(save_dir + "precalc_term_scales.npy", precalc_term_scales_to_keep.cpu())
with open(report_subdir + "precalc_term_scales.txt", "w") as f:
    f.write(str(precalc_term_scales_to_keep))

# Save weights and predictions. TODO: Where is this used?
np.save(
    report_subdir + "weights_and_predictions.npy",
    np.hstack([np.array(test_coefs_predict_lst).squeeze(), test_results[:, None]]),
)
# np.save("predictions.npy",results)

generate_outputs(
    report_subdir,
    training_losses,
    test_labels,
    test_results,
    gninatypes_filenames,
    test_pearsons,
    test_coefs_predict_lst,
    test_weighted_terms_lst,
    which_precalc_terms_to_keep,
    term_names,
    params,
)
