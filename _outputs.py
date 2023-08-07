from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import json
import csv


def _weights_heatmap(
    coefs_predict_lst: list[np.ndarray],
    which_precalc_terms_to_keep: np.ndarray,
    termnames: np.ndarray,
    save_dir: str,
    gninatypes_filenames: list[tuple[str, str]],
):
    """Saves a heatmap of the weights.
    
    Args:
        coefs_predict_lst: A list of numpy arrays, where each numpy array
            represents the weights of a model.
        which_precalc_terms_to_keep: A numpy array of booleans indicating which
            precalculated terms to keep.
        termnames: A numpy array of strings, where each string represents the
            name of a term in the precalculated terms.
        save_dir: A string representing the directory to save the heatmap to.
        gninatypes_filenames: A list of 2-tuples, where each 2-tuple represents
            the gninatypes filenames for a protein and ligand, respectively.
    """

    # Save all weights
    header = ["protein_gnina_types", "ligand_gnina_types"] + [
        h.replace(",", "_") for h in termnames[which_precalc_terms_to_keep]
    ]
    ccweights = np.array(coefs_predict_lst)
    ccweights = np.reshape(ccweights, (ccweights.shape[0], -1)).tolist()

    # Now add back in filenames to each item
    for i in range(len(ccweights)):
        ccweights[i].insert(0, gninatypes_filenames[i][1])
        ccweights[i].insert(0, gninatypes_filenames[i][0])

    # Save ccweights to csv file, using the values in header as the column names
    with open(f"{save_dir}weights.csv", "w") as f:
        _save_to_csv(f, header, ccweights)
    NUM_EXAMPLES_TO_PICK = 100
    np.random.shuffle(ccweights)
    ccweights = ccweights[:NUM_EXAMPLES_TO_PICK]

    # Save ccweights to csv file, using the values in header as the column names
    with open(f"{save_dir}a_few_weights.csv", "w") as f:
        _save_to_csv(f, header, ccweights)
    # Scale the columns of the ccweights so that they are z scores
    ccweights = np.array([ccweights[i][2:] for i in range(len(ccweights))])
    ccweights = (ccweights - ccweights.mean(axis=0)) / ccweights.std(axis=0)

    # Scale the columns of ccweights so they go from 0 to 1
    # min_col = np.min(ccweights, axis=0)
    # span_col = np.max(ccweights, axis=0) - min_col
    # ccweights = (ccweights - min_col) / span_col

    plt.clf()
    sns.heatmap(ccweights)
    plt.xlabel("Weights")
    plt.ylabel("Prot/Lig Complexes")
    plt.savefig(f"{save_dir}a_few_weights.png")


def _save_to_csv(f: Any, header: list[str], vals: list[list[Any]]):
    """Save ccweights to csv file, using the values in header as the column names
    
    Args:
        f: A file-like object.
        header: A list of strings, where each string represents the name of a
            column.
        vals: A list of lists with the values (rows). Can be combination of strings
            and numbers.
    """

    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(vals)


def _contributions_heatmap(
    contributions_lst: list[np.ndarray],
    which_precalc_terms_to_keep: np.ndarray,
    termnames: np.ndarray,
    save_dir: str,
    gninatypes_filenames: list[tuple[str, str]],
):
    """Saves a heatmap of the contributions.
    
    Args:
        contributions_lst: A list of numpy arrays, where each numpy array
            represents the contributions for a given example.
        which_precalc_terms_to_keep: A numpy array of booleans indicating which
            precalculated terms to keep.
        termnames: A numpy array of strings, where each string represents the
            name of a term in the precalculated terms.
        save_dir: A string representing the directory to save the heatmap to.
        gninatypes_filenames: A list of 2-tuples, where each 2-tuple represents
            the gninatypes filenames for a protein and ligand, respectively.
    """

    # save all contributions
    header = ["protein_gnina_types", "ligand_gnina_types"] + [
        h.replace(",", "_") for h in termnames[which_precalc_terms_to_keep]
    ]

    # Save ccweights to csv file, using the values in header as the column names
    contribs = np.array(contributions_lst)
    contribs = np.reshape(contribs, (contribs.shape[0], -1)).tolist()

    # Now add back in filenames to each item
    for i in range(len(contribs)):
        contribs[i].insert(0, gninatypes_filenames[i][1])
        contribs[i].insert(0, gninatypes_filenames[i][0])

    with open(f"{save_dir}contributions.csv", "w") as f:
        _save_to_csv(f, header, contribs)
    NUM_EXAMPLES_TO_PICK = 100
    np.random.shuffle(contribs)
    contribs = contribs[:NUM_EXAMPLES_TO_PICK]

    # Save ccweights to csv file, using the values in header as the column names
    with open(f"{save_dir}a_few_contributions.csv", "w") as f:
        _save_to_csv(f, header, contribs)
    contribs = np.array([contribs[i][2:] for i in range(len(contribs))])

    plt.clf()
    sns.heatmap(contribs)
    plt.xlabel("Contributions")
    plt.ylabel("Prot/Lig Complexes")
    plt.savefig(f"{save_dir}a_few_contributions.png")


def generate_outputs(
    save_dir: str,
    losses: list[float],
    labels: np.ndarray,  # Numeric labels
    results: np.ndarray,  # Predictions
    gninatypes_filenames: list[tuple[str, str]],
    pearsons: list[float],
    coefs_predict_lst: list[np.ndarray],
    contributions_lst: list[np.ndarray],
    which_precalc_terms_to_keep: np.ndarray,
    termnames: np.ndarray,
    params: dict,
):
    """Generates outputs for the model.
    
    Args:
        save_dir: A string representing the directory to save the outputs to.
        losses: A list of floats, where each float represents the loss for a
            batch.
        labels: A numpy array of floats, where each float represents the label
            (experimentally measured affinity) for a protein/ligand complex.
        results: A numpy array of floats, where each float represents the
            prediction for a protein/ligand complex.
        gninatypes_filenames: A list of 2-tuples, where each 2-tuple represents
            the gninatypes filenames for a protein and ligand, respectively.
        pearsons: A list of floats, where each float represents the Pearson's
            correlation coefficient for a test set.
        coefs_predict_lst: A list of numpy arrays, where each numpy array
            represents the coefficients for a given example.
        contributions_lst: A list of numpy arrays, where each numpy array
            represents the contributions for a given example.
        which_precalc_terms_to_keep: A numpy array of booleans indicating which
            precalculated terms to keep.
        termnames: A numpy array of strings, where each string represents the
            name of a term in the precalculated terms.
        params: A Params object.
    """

    # Losses per batch
    plt.plot(losses)
    plt.plot(
        # moving average
        range(99, len(losses)),
        np.convolve(losses, np.ones(100) / 100, mode="valid"),
    )
    plt.ylim(0, 8)
    plt.savefig(f"{save_dir}loss_per_batch__train.png")

    # Clear plot and start over
    plt.clf()
    plt.plot(pearsons)
    plt.savefig(f"{save_dir}pearsons_per_epoch__test.png")

    # Predictions vs. reality
    plt.clf()
    j = sns.jointplot(x=labels, y=results)
    plt.suptitle("R = %.2f" % pearsons[-1])
    plt.savefig(f"{save_dir}label_vs_predict_final__test.png")

    # Show some representative weights. Should be similar across proteins, but
    # not identical.
    _weights_heatmap(
        coefs_predict_lst,
        which_precalc_terms_to_keep,
        termnames,
        save_dir,
        gninatypes_filenames,
    )

    _contributions_heatmap(
        contributions_lst,
        which_precalc_terms_to_keep,
        termnames,
        save_dir,
        gninatypes_filenames,
    )

    # Save params as json
    with open(f"{save_dir}params.json", "w") as f:
        json.dump(params, f, indent=4)

    # Save the term names
    # with open(save_dir + "termnames.txt", "w") as f:
    #     f.write("\n".join(termnames[which_precalc_terms_to_keep]))

    # Save pearsons as csv
    np.savetxt(f"{save_dir}pearsons.csv", pearsons, delimiter=",", fmt="%.8f")
