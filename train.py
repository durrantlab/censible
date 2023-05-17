from _training import train_single_fold
from _graphs import generate_graphs
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
        "val": 250, # 400,
        "description": "Number of epochs to train for."
    },
    {
        "name": "fold_num",
        "val": 0,
        "description": "Which fold to train on."
    },
    {
        "name": "batch_size",
        "val": 25,
        "description": "Batch size."
    },
    {
        "name": "lr",
        "val": 0.01,
        "description": "Learning rate."
    },
    {
        "name": "step_size",
        "val": 80,
        "description": "Step size for learning rate decay."
    },
    {
        "name": "prefix",
        "val": "randomsplit",
        "description": "Prefix for the output files."  # TODO: Correct description?
    },
    {
        "name": "termtypes",
        "val": "all",
        "description": "Which terms to use. Can be 'all', 'smina', or 'gaussian'."
    },
    {
        "name": "data_dir",
        "val": "./prepare_data/",
        "description": "Directory where the data is stored."
    },
]
# "prefix": "crystal",

# Create argparser with same args as params
parser = argparse.ArgumentParser()
for value in params:
    # parser.add_argument("--" + key, type=type(value), default=value)
    parser.add_argument("--" + value["name"], type=type(value["val"]), default=value["val"], help=value["description"])
args = parser.parse_args()
params = vars(args)

print(params)

orig_dir = os.getcwd() + os.sep

# change working directory to "./prepare_data/"
os.chdir(params["data_dir"])

# which_precalc_terms_to_keep is a boolean array, True if a given feature is worth
# keeping, False otherwise. term_names is a list of all the term names.
which_precalc_terms_to_keep, term_names, precalc_term_scale_factors = preprocess(params["termtypes"])

print('Preprocessing done.')
# This keeps only the smina terms (not gaussian terms)
# which_precalc_terms_to_keep[24:] = False

# Train the model
(
    model,
    tmp_labels,
    results,
    test_mses,
    ames,
    pearsons,
    training_losses,
    coefs_predict_lst,
    contributions_lst,
    which_precalc_terms_to_keep,
    precalc_term_scale_factors_updated
) = train_single_fold(
    CENet,
    which_precalc_terms_to_keep,
    params,
    term_names,
    precalc_term_scale_factors
)

# Save model
# Create the directory if it doesn't exist
if not os.path.exists(orig_dir + "imgs"):
    os.mkdir(orig_dir + "imgs")

# Get the current date and time as a string, in a format that can be a filename
now = datetime.datetime.now()
now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

# Define and create the save directory
save_dir = orig_dir + "imgs/" + now_str + "/"
os.mkdir(save_dir)

torch.save(model.state_dict(), save_dir + "model.pt")
with open(save_dir + "which_precalc_terms_to_keep.txt", "w") as f:
    f.write(str(which_precalc_terms_to_keep))
with open(save_dir + "precalc_term_scale_factors.txt", "w") as f:
    f.write(str(precalc_term_scale_factors_updated))
with open(save_dir + "term_names.txt", "w") as f:
    f.write(str([term_names[i] for i in range(len(term_names)) if which_precalc_terms_to_keep[i]]))

# Save weights and predictions
np.save("weights_and_predictions.npy",np.hstack([np.array(coefs_predict_lst).squeeze(),results[:,None]]))
#np.save("predictions.npy",results)

generate_graphs(
    save_dir,
    training_losses,
    tmp_labels,
    results,
    pearsons,
    coefs_predict_lst,
    contributions_lst,
    which_precalc_terms_to_keep,
    term_names,
    params
)
