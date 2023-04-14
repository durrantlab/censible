from _training import train_single_fold
from _graphs import generate_graphs

# Published performance for this model on this set is 1.5 RMSE and 0.7 Pearson's
# R, so we are pretty close (could train longer).
# from published_model import Net

from CEN_model import CENet
from _preprocess import preprocess

# import py3Dmol
import os

save_dir = os.getcwd() + os.sep

# change working directory to "./data/cen/"
# os.chdir("./data/cen/")
os.chdir("./prepare_data/")

# allct = np.loadtxt("all_cen.types", max_rows=5, dtype=str)

# which_precalc_terms_to_keep is a boolean array, True if a given feature is worth
# keeping, False otherwise. term_names is a list of all the term names.
which_precalc_terms_to_keep, term_names = preprocess()

# Train the model
(
    model,
    labels,
    results,
    test_mses,
    ames,
    pearsons,
    training_losses,
    coefs_predict_lst,
    contributions_lst,
) = train_single_fold(
    CENet,
    which_precalc_terms_to_keep,
    epochs=400,
    # use_ligands=True,
    # lr=0.0001
)

generate_graphs(
    save_dir,
    training_losses,
    labels,
    results,
    pearsons,
    coefs_predict_lst,
    contributions_lst,
    which_precalc_terms_to_keep,
    term_names,
)
