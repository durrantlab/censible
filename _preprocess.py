import molgrid
import torch
import numpy as np

def preprocess():
    # Some atomic interactions are nonexistent or rare and should be ignored.
    # Calculate statistics for each term.

    termnames = []
    for line in open("allterms.txt"):
        line = line.rstrip()
        if line:
            # I accidentally left a blank line in the middle of the file
            termnames.append(line.split()[1])
    termnames = np.array(termnames)

    allex = molgrid.ExampleProvider(
        ligmolcache="lig.molcache2",
        recmolcache="rec.molcache2",
        iteration_scheme=molgrid.IterationScheme.LargeEpoch,
        default_batch_size=1,
    )
    allex.populate("all_cen.types")
    labels = torch.tensor(allex.num_labels(), dtype=torch.float32)
    allaffinity = []
    allterms = []
    for batch in allex:
        ex = batch[0]  # batch size on
        affinity = ex.labels[0]
        terms = np.array(ex.labels[1:])
        allaffinity.append(affinity)
        allterms.append(terms)

    allaffinity = np.array(allaffinity)
    allterms = np.array(allterms)

    # for t in range(349):
    #     if allterms[1][t] != 0:
    #         print(str(t) + "  " + str(termnames[t]) + ": " + str(allterms[1][t]))

    cnts = np.count_nonzero(allterms, axis=0)
    # goodfeatures = cnts > 4000  # JDD: To keep only a few terms. Performs just as well.
    goodfeatures = cnts > 0

    num_terms_kept = np.sum(goodfeatures == True)
    print("Number of terms retained: " + str(num_terms_kept))

    return goodfeatures, termnames