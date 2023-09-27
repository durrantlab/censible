"""
Utilities to visualize/save model outputs (e.g., contributions, term weights).

The module use the Seaborn library and integrates with the PyTorch framework for
saving the trained model states.
"""

from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import json
import csv
import torch


def _weights_heatmap(
    coefs_predict_lst: list[np.ndarray],
    which_precalc_terms_to_keep: np.ndarray,
    term_names: np.ndarray,
    save_dir: str,
    gninatypes_filenames: list[tuple[str, str]],
):
    """Save a heatmap of the weights.
    
    Args:
        coefs_predict_lst (list[np.ndarray]): A list of numpy arrays, where
            each numpy array represents the weights of a model.
        which_precalc_terms_to_keep (np.ndarray): A numpy array of booleans
            indicating which precalculated terms to keep.
        term_names (np.ndarray): A numpy array of strings, where each string
            represents the name of a term in the precalculated terms.
        save_dir (str): A string representing the directory to save the
            heatmap to.
        gninatypes_filenames (list[tuple[str, str]]): A list of 2-tuples, where
            each 2-tuple represents the gninatypes filenames for a protein and
            ligand, respectively.
    """
    # Save all weights
    header = ["protein_gnina_types", "ligand_gnina_types"] + [
        h.replace(",", "_") for h in term_names[which_precalc_terms_to_keep]
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
    """Save ccweights to csv file, using header values as column names.
    
    Args:
        f (Any): A file-like object.
        header (list[str]): A list of strings, where each string represents
            the name of a column.
        vals (list[list[Any]]): A list of lists with the values (rows). Can be
            combination of strings and numbers.
    """
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(vals)


def _contributions_heatmap(
    contributions_lst: list[np.ndarray],
    which_precalc_terms_to_keep: np.ndarray,
    term_names: np.ndarray,
    save_dir: str,
    gninatypes_filenames: list[tuple[str, str]],
):
    """Save a heatmap of the contributions.
    
    Args:
        contributions_lst (list[np.ndarray]): A list of numpy arrays, where
            each numpy array represents the contributions for a given example.
        which_precalc_terms_to_keep (np.ndarray): A numpy array of booleans
            indicating which precalculated terms to keep.
        term_names (np.ndarray): A numpy array of strings, where each string
            represents the name of a term in the precalculated terms.
        save_dir (str): A string representing the directory to save the heatmap
            to.
        gninatypes_filenames (list[tuple[str, str]]): A list of 2-tuples, where
            each 2-tuple represents the gninatypes filenames for a protein and
            ligand, respectively.
    """
    # save all contributions
    header = ["protein_gnina_types", "ligand_gnina_types"] + [
        h.replace(",", "_") for h in term_names[which_precalc_terms_to_keep]
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


def _get_output_dir(params: dict) -> str:
    """Get the output directory for the model.
    
    Args:
        params (dict): A dictionary of parameters.
    
    Returns:
        str: A string representing the output directory for the model.
    """
    # Get the output directory, using default if necessary
    outdir = params.get("out_dir", "./outputs/")

    # If directory exists, make sure it is a directory
    if os.path.exists(outdir) and not os.path.isdir(outdir):
        raise ValueError(f"outdir {outdir} is not a directory")

    # Create the directory if it doesn't exist
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # Get the current date and time as a string, in a format that can be a filename
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Define and create the save subdirectory
    outdir = f"{outdir}/{now_str}/"
    os.mkdir(outdir)

    # Make report subdirectory
    report_subdir = f"{outdir}/report/"
    os.mkdir(report_subdir)

    return outdir


def _save_model(
    model: "CENet",
    results: dict,
    coefs_predict_lst: list[np.ndarray],
    which_precalc_terms_to_keep: np.ndarray,
    precalc_term_scales_to_keep: np.ndarray,
    term_names: np.ndarray,
    outdir: str,
    report_subdir: str,
):
    """Save the model and associated files.
    
    Args:
        model (CENet): The model to save.
        results (dict): A dictionary of results.
        coefs_predict_lst (list[np.ndarray]): A list of numpy arrays, where
            each numpy array represents the coefficients for a given example.
        which_precalc_terms_to_keep (np.ndarray): A numpy array of booleans
            indicating which precalculated terms to keep.
        precalc_term_scales_to_keep (np.ndarray): A numpy array of floats
            indicating the scale of each precalculated term to keep.
        term_names (np.ndarray): A numpy array of strings, where each string
            represents the name of a term in the precalculated terms.
        outdir (str): A string representing the output directory for the model.
        report_subdir (str): A string representing the output directory for the
            report.
    """
    # Save the model
    torch.save(model.state_dict(), f"{outdir}model.pt")

    # Save a boolean list of which precalculated terms are used in this model
    np.save(f"{outdir}which_precalc_terms_to_keep.npy", which_precalc_terms_to_keep)
    with open(f"{report_subdir}which_precalc_terms_to_keep.txt", "w") as f:
        f.write(str(which_precalc_terms_to_keep))

    # Save the names of the retained terms
    with open(f"{report_subdir}term_names.txt", "w") as f:
        f.write(
            str(
                [
                    term_names[i]
                    for i in range(len(term_names))
                    if which_precalc_terms_to_keep[i]
                ]
            )
        )

    # Save the normalization factors applied to the retained precalculated terms.
    np.save(f"{outdir}precalc_term_scales.npy", precalc_term_scales_to_keep.cpu())
    with open(f"{report_subdir}precalc_term_scales.txt", "w") as f:
        f.write(str(precalc_term_scales_to_keep))

    # Save weights and predictions. TODO: Where is this used?
    np.save(
        f"{report_subdir}weights_and_predictions.npy",
        np.hstack([np.array(coefs_predict_lst).squeeze(), results[:, None]]),
    )
    # np.save("predictions.npy",results)


def save_outputs(
    model: "CENet",
    losses: list[float],
    labels: np.ndarray,  # Numeric labels
    results: np.ndarray,  # Predictions
    gninatypes_filenames: list[tuple[str, str]],
    pearsons: list[float],
    coefs_predict_lst: list[np.ndarray],
    contributions_lst: list[np.ndarray],
    which_precalc_terms_to_keep: np.ndarray,
    precalc_term_scales_to_keep: np.ndarray,
    term_names: np.ndarray,
    params: dict,
):
    """Generate outputs for the model.
    
    Args:
        model (CENet): A CENet model to save.
        losses (list[float]): A list of floats, where each float represents the
            loss for a batch.
        labels (np.ndarray): A numpy array of floats, where each float
            represents the label (experimentally measured affinity) for a
            protein/ligand complex.
        results (np.ndarray): A numpy array of floats, where each float
            represents the prediction for a protein/ligand complex.
        gninatypes_filenames (list[tuple[str, str]]): A list of 2-tuples, where
            each 2-tuple represents the gninatypes filenames for a protein and
            ligand, respectively.
        pearsons (list[float]): A list of floats, where each float represents
            the Pearson's correlation coefficient for a test set.
        coefs_predict_lst (list[np.ndarray]): A list of numpy arrays, where
            each numpy array represents the coefficients for a given example.
        contributions_lst (list[np.ndarray]): A list of numpy arrays, where
            each numpy array represents the contributions for a given example.
        which_precalc_terms_to_keep (np.ndarray): A numpy array of booleans
            indicating which precalculated terms to keep.
        precalc_term_scales_to_keep (np.ndarray): A numpy array of floats,
            where each float represents the scale of a precalculated term.
        term_names (np.ndarray): A numpy array of strings, where each string
            represents the name of a term in the precalculated terms.
        params (dict): A Params object.
    """
    outdir = _get_output_dir(params)
    report_subdir = f"{outdir}report/"

    # Save the model and associated files
    _save_model(
        model,
        results,
        coefs_predict_lst,
        which_precalc_terms_to_keep,
        precalc_term_scales_to_keep,
        term_names,
        outdir,
        report_subdir,
    )

    # Losses per batch
    plt.plot(losses)
    plt.plot(
        # moving average
        range(99, len(losses)),
        np.convolve(losses, np.ones(100) / 100, mode="valid"),
    )
    plt.ylim(0, 8)
    plt.savefig(f"{report_subdir}loss_per_batch__train.png")

    # Clear plot and start over
    plt.clf()
    plt.plot(pearsons)
    plt.savefig(f"{report_subdir}pearsons_per_epoch__test.png")

    # Predictions vs. reality
    plt.clf()
    j = sns.jointplot(x=labels, y=results)
    plt.suptitle("R = %.2f" % pearsons[-1])
    plt.savefig(f"{report_subdir}label_vs_predict_final__test.png")

    # Show some representative weights. Should be similar across proteins, but
    # not identical.
    _weights_heatmap(
        coefs_predict_lst,
        which_precalc_terms_to_keep,
        term_names,
        report_subdir,
        gninatypes_filenames,
    )

    _contributions_heatmap(
        contributions_lst,
        which_precalc_terms_to_keep,
        term_names,
        report_subdir,
        gninatypes_filenames,
    )

    # Save params as json
    with open(f"{report_subdir}params.json", "w") as f:
        json.dump(params, f, indent=4)

    # Save the term names
    # with open(save_dir + "term_names.txt", "w") as f:
    #     f.write("\n".join(term_names[which_precalc_terms_to_keep]))

    # Save pearsons as csv
    np.savetxt(f"{report_subdir}pearsons.csv", pearsons, delimiter=",", fmt="%.8f")
