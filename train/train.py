# add ../ to the path
import sys

sys.path.append("..")

from censible.training import get_args, train_single_fold, validate_params
from censible.outputs import save_outputs
from censible.CEN_model import CENet
from censible.preprocess import preprocess

params = get_args()
params = validate_params(params)

# which_precalc_terms_to_keep is a boolean array, True if a given feature is
# worth keeping, False otherwise. term_names is a list of all the term names.
which_precalc_terms_to_keep, term_names, precalc_term_scales = preprocess(
    params["termtypes"], params["data_dir"]
)

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

# Save the model, reports, etc.
save_outputs(
    model,
    training_losses,
    test_labels,
    test_results,
    gninatypes_filenames,
    test_pearsons,
    test_coefs_predict_lst,
    test_weighted_terms_lst,
    which_precalc_terms_to_keep,
    precalc_term_scales_to_keep,
    term_names,
    params,
)
