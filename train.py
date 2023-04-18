from _training import train_single_fold
from _graphs import generate_graphs
import json

# Published performance for this model on this set is 1.5 RMSE and 0.7 Pearson's
# R, so we are pretty close (could train longer).
# from published_model import Net

from CEN_model import CENet
from _preprocess import preprocess

# import py3Dmol
import os

params = {
    "epochs": 250,  # 400,
    "fold_num": 0,
    "batch_size": 25,
    "lr": 0.01,
    "step_size": 80,
    # "prefix": "crystal",
    "prefix": "randomsplit",
}

save_dir = os.getcwd() + os.sep

# change working directory to "./data/cen/"
# os.chdir("./data/cen/")
os.chdir("./prepare_data/")

# allct = np.loadtxt("all_cen.types", max_rows=5, dtype=str)

# which_precalc_terms_to_keep is a boolean array, True if a given feature is worth
# keeping, False otherwise. term_names is a list of all the term names.
which_precalc_terms_to_keep, term_names = preprocess()

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
    which_precalc_terms_to_keep
) = train_single_fold(
    CENet,
    which_precalc_terms_to_keep,
    params,
    term_names
    # epochs=params["epochs"],
    # fold_num=params["fold_num"]
    # use_ligands=True,
    # lr=0.0001
)

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
