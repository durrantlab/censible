from _training import train_single_fold
from _graphs import generate_graphs
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

params = {
    "epochs": 250, # 400,
    "fold_num": 0,
    "batch_size": 25,
    "lr": 0.01,
    "step_size": 80,
    # "prefix": "crystal",
    "prefix": "randomsplit",
}

# Create argparser with same args as params
parser = argparse.ArgumentParser()
for key, value in params.items():
    parser.add_argument("--" + key, type=type(value), default=value)
args = parser.parse_args()
params = vars(args)

print(params)

orig_dir = os.getcwd() + os.sep

# change working directory to "./prepare_data/"
os.chdir("./prepare_data/")

# which_precalc_terms_to_keep is a boolean array, True if a given feature is worth
# keeping, False otherwise. term_names is a list of all the term names.
which_precalc_terms_to_keep, term_names, precalc_term_scale_factors = preprocess()

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
import pdb; pdb.set_trace()

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
