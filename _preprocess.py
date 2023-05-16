import molgrid
import torch
import numpy as np

RARE_TERM_RATIO_CUTOFF = 0.01

def preprocess(termtypes):
    # Some atomic interactions are nonexistent or rare and should be ignored.
    # Calculate statistics for each term.

    # Get the names of all the terms
    term_names = []
    for line in open("allterms.txt"):
        line = line.rstrip()
        if line:
            # I accidentally left a blank line in the middle of the file
            term_names.append(line.split()[1])
    term_names = np.array(term_names)

    # Get all examples (grids)
    all_examples = molgrid.ExampleProvider(
        ligmolcache="lig.molcache2",
        recmolcache="rec.molcache2",
        iteration_scheme=molgrid.IterationScheme.LargeEpoch,
        default_batch_size=1,
    )
    all_examples.populate("all_cen.types")

    # Get the experimentally measured affinities as well as the other terms.
    all_affinities = []
    all_terms = []
    for batch in all_examples:
        example = batch[0]  # batch size on
        affinity = example.labels[0]
        terms = np.array(example.labels[1:])
        all_affinities.append(affinity)
        all_terms.append(terms)

    all_affinities = np.array(all_affinities)
    all_terms = np.array(all_terms)

    # for t in range(349):
    #     if allterms[1][t] != 0:
    #         print(str(t) + "  " + str(termnames[t]) + ": " + str(allterms[1][t]))

    which_precalc_terms_to_keep = remove_rare_terms(all_terms, termtypes)

    precalc_term_scale_factors = normalize_terms(all_terms, which_precalc_terms_to_keep)
    np.save(termtypes+"_normalization_factors.npy", precalc_term_scale_factors)

    return which_precalc_terms_to_keep, term_names, precalc_term_scale_factors

def remove_rare_terms(all_terms: np.ndarray, termtypes: str, which_precalc_terms_to_keep: np.ndarray = None):
    global RARE_TERM_RATIO_CUTOFF

    # If which_precalc_terms_to_keep is not provided, then it is calculated here. All trues.
    if which_precalc_terms_to_keep is None:
        which_precalc_terms_to_keep = np.ones(all_terms.shape[1], dtype=bool)

    num_examples = all_terms.shape[0]
    min_examples_permitted: int = int(num_examples * RARE_TERM_RATIO_CUTOFF)

    # Find the terms that are not just all zeros. which_precalc_terms_to_keep is a boolean
    # array, True if the term should be retained, false otherwise.
    cnts = np.count_nonzero(all_terms, axis=0)
    to_keep = cnts > min_examples_permitted

    # Update which_precalc_terms_to_keep
    which_precalc_terms_to_keep = np.logical_and(
        which_precalc_terms_to_keep, to_keep
    )

    if termtypes == "smina":
        which_precalc_terms_to_keep[:24] = False
    elif termtypes == "gaussian":
        which_precalc_terms_to_keep[24:] = False

    num_terms_kept = np.sum(which_precalc_terms_to_keep == True)
    print("Number of terms retained: " + str(num_terms_kept))
    print("Number of terms removed: " + str(np.sum(to_keep == False)))

    return which_precalc_terms_to_keep

def normalize_terms(all_terms, which_precalc_terms_to_keep):
    # TODO: Need to implement ability tosave values in factors and load them
    # back in for inference.

    MAX_VAL_AFTER_NORM = 1.0

    # Normalize the columns so the values go between 0 and 1. 
    precalc_term_scale_factors = np.zeros(all_terms.shape[1])
    for i in range(all_terms.shape[1]):
        col = all_terms[:, i]
        max_abs = np.max(np.abs(col))
        precalc_term_scale_factors[i] = 1.0

        if max_abs > 0:
            precalc_term_scale_factors[i] = MAX_VAL_AFTER_NORM * 1.0 / max_abs

    # Save factors
    # np.save("batch_labels.jdd.npy", batch_labels[:, 1:][:, goodfeatures])
    # np.save("factors.jdd.npy", factors)

    # To turn off this modification entirely, uncomment out below line. Sets all
    # factors to 1.
    # factors[:] = 1

    # Check to make sure between -1 and 1
    # batch_labels_tmp = batch_labels[:, 1:][:, goodfeatures] * factors
    # print(
    #     float(batch_labels_tmp.max()), 
    #     float(batch_labels_tmp.min())
    # )

    # Convert factors to a tensor
    precalc_term_scale_factors = torch.from_numpy(precalc_term_scale_factors).float().to(device="cuda")
    #precalc_term_scale_factors = torch.from_numpy(precalc_term_scale_factors).float()

    return precalc_term_scale_factors
