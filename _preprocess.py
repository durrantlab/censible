import molgrid
import torch
import numpy as np

def preprocess():
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
    # labels = torch.tensor(all_examples.num_labels(), dtype=torch.float32)

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

    # Find the terms that are not just all zeros. which_precalc_terms_to_keep is a boolean
    # array, True if the term should be retained, false otherwise.
    cnts = np.count_nonzero(all_terms, axis=0)
    # goodfeatures = cnts > 4000  # JDD: To keep only a few terms. Performs just as well.
    which_precalc_terms_to_keep = cnts > 0

    num_terms_kept = np.sum(which_precalc_terms_to_keep == True)
    print("Number of terms retained: " + str(num_terms_kept))

    return which_precalc_terms_to_keep, term_names