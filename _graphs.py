import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import json

def _weights_heatmap(coefs_predict_lst, which_precalc_terms_to_keep, termnames, save_dir):
    plt.clf()
    NUM_EXAMPLES_TO_PICK = 100
    np.random.shuffle(coefs_predict_lst)
    ccweights = np.array(coefs_predict_lst[:NUM_EXAMPLES_TO_PICK])
    num_terms_kept = np.sum(which_precalc_terms_to_keep == True)
    ccweights = np.reshape(ccweights, (NUM_EXAMPLES_TO_PICK, num_terms_kept))
    header = [h.replace(",", "_") for h in termnames[which_precalc_terms_to_keep]]

    # Save ccweights to csv file, using the values in header as the column names
    np.savetxt(
        save_dir + "a_few_weights.csv",
        ccweights,
        delimiter=",",
        header=",".join(header),
        fmt="%.8f",
    )

    # Scale the columns of the ccweights so that they are z scores
    ccweights = (ccweights - ccweights.mean(axis=0)) / ccweights.std(axis=0)

    # Scale the columns of ccweights so they go from 0 to 1
    # min_col = np.min(ccweights, axis=0)
    # span_col = np.max(ccweights, axis=0) - min_col
    # ccweights = (ccweights - min_col) / span_col

    sns.heatmap(ccweights)
    plt.xlabel("Weights")
    plt.ylabel("Prot/Lig Complexes")
    plt.savefig(save_dir + "a_few_weights.png")

def _contributions_heatmap(contributions_lst, goodfeatures, termnames, save_dir):

    plt.clf()
    NUM_EXAMPLES_TO_PICK = 100
    np.random.shuffle(contributions_lst)
    contribs = np.array(contributions_lst[:NUM_EXAMPLES_TO_PICK])
    num_terms_kept = np.sum(goodfeatures == True)
    contribs = np.reshape(contribs, (NUM_EXAMPLES_TO_PICK, num_terms_kept))
    header = [h.replace(",", "_") for h in termnames[goodfeatures]]

    # Save ccweights to csv file, using the values in header as the column names
    np.savetxt(
        save_dir + "a_few_contributions.csv",
        contribs,
        delimiter=",",
        header=",".join(header),
        fmt="%.8f",
    )

    sns.heatmap(contribs)
    plt.xlabel("Contributions")
    plt.ylabel("Prot/Lig Complexes")
    plt.savefig(save_dir + "a_few_contributions.png")


def generate_graphs(
    save_dir,
    losses,
    labels,
    results,
    pearsons,
    coefs_predict_lst,
    contributions_lst,
    which_precalc_terms_to_keep,
    termnames,
    params
):
    # Losses per batch
    plt.plot(losses)
    plt.plot(
        # moving average
        range(99, len(losses)),
        np.convolve(losses, np.ones(100) / 100, mode="valid"),
    )
    plt.ylim(0, 8)
    plt.savefig(save_dir + "loss_per_batch__train.png")

    # Clear plot and start over
    plt.clf()
    plt.plot(pearsons)
    plt.savefig(save_dir + "pearsons_per_epoch__test.png")

    # Predictions vs. reality
    plt.clf()
    j = sns.jointplot(x=labels, y=results)
    plt.suptitle("R = %.2f" % pearsons[-1])
    plt.savefig(save_dir + "label_vs_predict_final__test.png")

    # Show some representative weights. Should be similar across proteins, but
    # not identical.
    _weights_heatmap(coefs_predict_lst, which_precalc_terms_to_keep, termnames, save_dir)

    _contributions_heatmap(contributions_lst, which_precalc_terms_to_keep, termnames, save_dir)

    # Save params as json
    with open(save_dir + "params.json", "w") as f:
        json.dump(params, f, indent=4)

    # Save the term names
    # with open(save_dir + "termnames.txt", "w") as f:
    #     f.write("\n".join(termnames[which_precalc_terms_to_keep]))

