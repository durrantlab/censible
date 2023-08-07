from typing import List
import molgrid
import torch
import numpy as np

RARE_TERM_RATIO_CUTOFF = 0.01


def preprocess(termtypes: str, data_dir: str):
    """Preprocesses the data.
    
    Args:
        termtypes (str): A string representing the path to the termtypes file.
            Can be 'all', 'smina', or 'gaussian'.
        data_dir (str): A string representing the path to the data directory.

    Returns:
        A tuple containing:
            which_precalc_terms_to_keep: A numpy array of booleans indicating
                which precalculated terms to keep.
            term_names: A numpy array of strings, where each string represents
                the name of a term in the precalculated terms.
            precalc_term_scales: A tensor of floats representing the scale
                factors for each term.
    """

    if data_dir[-1] != "/":
        data_dir += "/"

    # Some atomic interactions are nonexistent or rare and should be ignored.
    # Calculate statistics for each term.

    # Get the names of all the terms. NOTE: You can't use allterms.txt for this,
    # because smina reorders the terms when it creates the vector for each
    # protein/ligand complex. So strange.
    term_names = []
    for line in open(f"{data_dir}smina_ordered_terms.txt"):
        line = line.rstrip()
        term_names.append(line)
    term_names = np.array(term_names)

    # Get all examples (grids). These aren't used for training in the end. Just
    # used here to get the terms, scale factors, etc. The are reloaded in
    # _training.py.
    all_examples = molgrid.ExampleProvider(
        ligmolcache=f"{data_dir}lig.molcache2",
        recmolcache=f"{data_dir}rec.molcache2",
        iteration_scheme=molgrid.IterationScheme.LargeEpoch,
        default_batch_size=1,
    )
    all_examples.populate(f"{data_dir}all_cen.types")

    # Get the experimentally measured affinities as well as the other terms.
    # all_affinities = []  # NOTE: Affinity not used here. Data reloaded (per
    # fold) in _training.py.
    all_terms = []
    for batch in all_examples:
        example = batch[0]  # batch size one
        # affinity = example.labels[0]
        # all_affinities.append(affinity)
        terms = np.array(example.labels[1:])  # Not affinity
        all_terms.append(terms)

    # all_affinities = np.array(all_affinities)
    all_terms = np.array(all_terms)

    # for t in range(349):
    #     if allterms[1][t] != 0:
    #         print(str(t) + "  " + str(termnames[t]) + ": " + str(allterms[1][t]))

    which_precalc_terms_to_keep = remove_rare_terms(all_terms, termtypes, term_names)

    precalc_term_scales = get_precalc_term_scales(all_terms, which_precalc_terms_to_keep)

    return which_precalc_terms_to_keep, term_names, precalc_term_scales


def remove_rare_terms(
    all_terms: np.ndarray,
    termtypes: str,
    term_names: np.ndarray,
    which_precalc_terms_to_keep: np.ndarray = None,
) -> np.ndarray:
    """Removes rare terms from the data.
    
    Args:
        all_terms (np.ndarray): A 2D numpy array representing all the terms for
            all examples.
        termtypes (str): A string representing the path to the termtypes file.
            Can be 'all', 'smina', or 'gaussian'.
        term_names (np.ndarray): A numpy array of strings representing the 
            names of all the terms.
        which_precalc_terms_to_keep (np.ndarray): A boolean array representing
            which terms to keep. If None, then it will be created rather than
            updated.

    Returns:
        A boolean array representing which terms to keep.
    """

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
    which_precalc_terms_to_keep = np.logical_and(which_precalc_terms_to_keep, to_keep)

    # It seems ad4_solvation(d-sigma=3.6,_s/q=0.01097,_c=8) is included twice.
    # Let's turn off the second instance. Hackish.
    idx = np.array(
        [
            i
            for i in range(len(term_names))
            if "ad4_solvation(d-sigma=3.6,_s/q=0.01097,_c=8)" in term_names[i]
        ]
    )
    assert len(idx) == 2
    which_precalc_terms_to_keep[idx[-1]] = False

    if termtypes == "smina":
        # Get the indexes of the term_names that contain the string "atom_type_gaussian"
        idx = np.array(
            [i for i in range(len(term_names)) if "atom_type_gaussian" in term_names[i]]
        )
        which_precalc_terms_to_keep[idx] = False
    elif termtypes == "gaussian":
        # Get the indexes of the term_names that do not contain the string "atom_type_gaussian"
        idx = np.array(
            [
                i
                for i in range(len(term_names))
                if "atom_type_gaussian" not in term_names[i]
            ]
        )
        which_precalc_terms_to_keep[idx] = False

    print(f"Number of terms retained: {np.sum(which_precalc_terms_to_keep == True)}")
    print(f"Number of terms removed: {np.sum(which_precalc_terms_to_keep == False)}")

    return which_precalc_terms_to_keep


def get_precalc_term_scales(
    all_terms: np.ndarray, which_precalc_terms_to_keep: np.ndarray
) -> torch.Tensor:
    """Gets the scales for each term. The scales will normalize values when
    multipled.
    
    Args:
        all_terms (np.ndarray): A 2D numpy array representing all the terms for
            all examples.
        which_precalc_terms_to_keep (np.ndarray): A boolean array representing
            which terms to keep.
    
    Returns:
        A 1D tensor representing the scales for each term.
    """

    MAX_VAL_AFTER_NORM = 1.0

    # Scales will normalize values when multipled.
    precalc_term_scales = np.zeros(all_terms.shape[1])
    for i in range(all_terms.shape[1]):
        col = all_terms[:, i]
        max_abs = np.max(np.abs(col))
        precalc_term_scales[i] = 1.0

        if max_abs > 0:
            precalc_term_scales[i] = MAX_VAL_AFTER_NORM * 1.0 / max_abs

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
    precalc_term_scales = (
        torch.from_numpy(precalc_term_scales).float().to(device="cuda")
    )
    # precalc_term_scales = torch.from_numpy(precalc_term_scales).float()

    return precalc_term_scales