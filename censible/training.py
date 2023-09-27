"""
This module provides utilities for training and evaluating the CENet model.

It includes functions to load datasets, train the model on a single fold, and
apply various pre-processing and setup steps necessary for training and
evaluation. The module focuses on handling molecular data, specifically with
respect to terms, weights, and contributions.
"""

import argparse
from typing import Any, Tuple
from censible.debug import grid_channel_to_xyz_file
import molgrid
import torch
import torch.optim as optim
import os
import numpy as np
from scipy.stats import pearsonr
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# from _debug import grid_channel_to_xyz_file


def load_split(
    types_filename: str, batch_size: int, data_dir: str, is_training_set: bool = False
) -> Tuple[molgrid.molgrid.ExampleProvider, Any]:
    """Load the data from the types file.
    
    Args:
        types_filename (str): A string representing the path to the types file.
        batch_size (int): An integer representing the batch size.
        data_dir (str): A string representing the path to the data directory.
        is_training_set (bool): A boolean representing whether the data is a
            training set. Defaults to False.

    Returns:
        A tuple of the molgrid.ExampleProvider and the gninatypes filenames. If
        it's a training set, then the gninatypes filenames will be None.
        Otherwise, it will be a list of tuples of the gninatypes filenames
        (receptor, ligand).
    """
    types_filename = f"{data_dir}/{types_filename}"

    # You need to keep track of the ligand and receptor filenames
    if not is_training_set:
        # It's a testing set, so there will be no shuffle. Keep track of the
        # gninatypes filenames for reporting.
        gninatypes_filenames = []
        with open(types_filename, "r") as f:
            for line in f:
                receptor_file, ligand_file = line.strip().split()[-2:]
                gninatypes_filenames.append((receptor_file, ligand_file))
    else:
        # Testing set. So there will be a shuffle. No point in keeping track of
        # ordered gninatypes filenames.
        gninatypes_filenames = None

    kwargs = {
        "ligmolcache": f"{data_dir}/lig.molcache2",
        "recmolcache": f"{data_dir}/rec.molcache2",
        "iteration_scheme": molgrid.IterationScheme.LargeEpoch,
    }

    if is_training_set:
        # It's a training set. Shuffle.
        kwargs["shuffle"] = True
        kwargs["default_batch_size"] = batch_size
        kwargs["stratify_min"] = 3  # TODO: What do these mean?
        kwargs["stratify_max"] = 10
        kwargs["stratify_step"] = 1
        kwargs["stratify_pos"] = 0
    else:
        # It's a testing set. No shuffling.
        kwargs["default_batch_size"] = 1

    # Create a training dataset, which has access to all receptor and ligand grids.
    dataset = molgrid.ExampleProvider(**kwargs)

    # Indicate that the training set will only use those grids in a given file,
    # not all grids.
    dataset.populate(types_filename)
    # train_dataset.populate("all_cen.types")

    return dataset, gninatypes_filenames


def train_single_fold(
    Net: "CENet",
    which_precalc_terms_to_keep: np.ndarray,
    params: dict,
    term_names: np.ndarray,
    precalc_term_scales: torch.Tensor,
) -> tuple:
    """Train a single fold of the CEN model.
    
    Args:
        Net (CENet): The CEN model.
        which_precalc_terms_to_keep (np.ndarray): A boolean array indicating
            which terms to keep.
        params (dict): A dictionary of parameters.
        term_names (np.ndarray): A numpy array of term names.
        precalc_term_scales (torch.Tensor): A tensor of term scales.

    Returns:
        A tuple containing information about the results, etc.
    """
    # TODO: Divide this monster function into subfunctions

    # The main object. See
    # https://gnina.github.io/libmolgrid/python/index.html#the-gridmaker-class
    gmaker = molgrid.GridMaker()  # use defaults

    # Create a training dataset, which has access to all receptor and ligand grids.
    # train_dataset = molgrid.ExampleProvider(
    #     ligmolcache="lig.molcache2",
    #     recmolcache="rec.molcache2",
    #     iteration_scheme=molgrid.IterationScheme.LargeEpoch,
    #     shuffle=True,
    #     # default_batch_size=1
    #     default_batch_size=params["batch_size"],
    #     stratify_min=3,  # TODO: What do these mean?
    #     stratify_max=10,
    #     stratify_step=1,
    #     stratify_pos=0,
    # )

    # Indicate that the training set will only use those grids in a given file,
    # not all grids.
    # train_dataset.populate(params["prefix"] + ("train%d_cen.types" % params["fold_num"]))
    ## train_dataset.populate("all_cen.types")

    train_dataset, _ = load_split(
        params["prefix"] + ("train%d_cen.types" % params["fold_num"]),
        params["batch_size"],
        params["data_dir"],
        is_training_set=True,
    )

    # Similarly create a testing dataset.
    # test_dataset = molgrid.ExampleProvider(
    #     ligmolcache="lig.molcache2",
    #     recmolcache="rec.molcache2",
    #     iteration_scheme=molgrid.IterationScheme.LargeEpoch,
    #     default_batch_size=1,
    # )
    # test_dataset.populate(params["prefix"] + ("test%d_cen.types" % params["fold_num"]))
    test_dataset, test_gninatypes_filenames = load_split(
        params["prefix"] + ("test%d_cen.types" % params["fold_num"]),
        1,
        params["data_dir"],
        is_training_set=False,
    )

    # Note that alllabels doesn't contain all the labels, but is simply a tensor
    # where a given batch's labels will be placed.
    all_labels_for_training = torch.zeros(
        (params["batch_size"], train_dataset.num_labels()),
        dtype=torch.float32,
        device="cuda",
    )
    single_label_for_testing = torch.zeros(
        (1, train_dataset.num_labels()), dtype=torch.float32, device="cuda"
    )

    # Now that you've defined the train and test datasets, further identify
    # terms that are all zeros in the training set. Effectively remove those as
    # well. This is especially important because otherwise, there could be some
    # terms that are all zeros in the training set, but not in the testing set.
    # So the training process puts no constraints on those terms, inevitably
    # leading to bad results when evaluating on the test set.
    tmp_labels = []
    for train_batch in train_dataset:
        train_batch.extract_labels(all_labels_for_training)
        tmp_labels.append(all_labels_for_training.cpu().numpy()[:, 1:])
    train_dataset.reset()

    # Update precalc_term_scales to include only the terms that are
    # actually used.
    precalc_term_scales_to_keep = precalc_term_scales[which_precalc_terms_to_keep]

    # Create tensors to hold the inputs.
    dims = gmaker.grid_dimensions(train_dataset.num_types())
    tensor_shape = (params["batch_size"],) + dims  # shape of batched input
    input_tensor_for_training = torch.zeros(
        tensor_shape, dtype=torch.float32, device="cuda"
    )
    single_input_for_testing = torch.zeros(
        (1,) + dims, dtype=torch.float32, device="cuda"
    )

    # Create the model.
    nterms = np.count_nonzero(which_precalc_terms_to_keep)
    model = Net(dims, nterms).to("cuda")
    model.apply(weights_init)
    # Setup optimizer and scheduler for training.
    optimizer = optim.SGD(
        model.parameters(), lr=params["lr"], weight_decay=0.0001, momentum=0.9
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=params["step_size"], gamma=0.1
    )

    # Keep track of various metrics as training progresses.
    training_losses = []
    test_mses = []
    test_ames = []
    test_pearsons = []
    coefs_all = []

    for epoch_idx in range(params["epochs"]):
        # Loop through the batches (25 examples each)
        # print("epoch", epoch_idx)

        # train_dataset is exhausted at this point for some reason, but .reset()
        # doesn't seem to work. Run optimizer step to avoid error, but would be
        # good to get to the bottom of why reset doesn't work here. As is,
        # skipping first generation (effectively).
        optimizer.step()

        for train_batch in train_dataset:
            # print("    batch", batch_idx)
            train_batch.extract_labels(all_labels_for_training)
            affinity_label_for_training = all_labels_for_training[:, 0]  # affinity only

            # Keep only ones that are which_precalc_terms_to_keep (see _preprocess.py)
            precalculated_terms = all_labels_for_training[:, 1:][
                :, which_precalc_terms_to_keep
            ]

            # Scale so never outside of -1 to 1
            precalculated_terms = (
                precalc_term_scales_to_keep * precalculated_terms
            )  # JDD

            # Get the grid (populates input_tensor_for_training)
            gmaker.forward(
                train_batch,
                input_tensor_for_training,
                random_translation=2,
                random_rotation=True,
            )

            # # TODO: Debug below
            # gmaker.set_resolution(0.1)
            # for channel in range(len(input_tensor_for_training[0])):
            #     xyz = grid_channel_to_xyz_file(input_tensor_for_training[0][channel])
            #     with open("tmp" + str(channel) + ".xyz", "w") as f:
            #         f.write(xyz)
                
            # grid_channel_to_xyz_file(input_tensor[0][0])
            # print(batch_idx)
            # print("")
            # grid_channel_to_xyz_file(input_tensor[21][0])
            # break

            # Get the output for this batch. output[0] is output tensor.
            # output[1] is None for some reason. Note that weighted_terms is the
            # pre-calculated terms times the coefficients.
            output, coef_predict, weighted_terms = model(
                input_tensor_for_training, precalculated_terms
            )

            training_loss = F.smooth_l1_loss(
                output.flatten(), affinity_label_for_training.flatten()
            )
            training_loss.backward()

            # print(loss)

            # clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            training_losses.append(float(training_loss))

        scheduler.step()

        # Evaluate the performance on test set, given that you just finished one
        # epoch of training.
        with torch.no_grad():
            test_results = []
            test_labels = []
            test_coefs_predict_lst = []
            test_weighted_terms_lst = []

            for batch_index, test_batch in enumerate(test_dataset):
                # Get this batch's labels
                test_batch.extract_labels(single_label_for_testing)

                # Populate the single_input_for_testing tensor with an example.
                gmaker.forward(test_batch, single_input_for_testing)

                # Run that through the model.
                output, coef_predict, weighted_terms = model(
                    single_input_for_testing,
                    single_label_for_testing[:, 1:][:, which_precalc_terms_to_keep]
                    * precalc_term_scales_to_keep,  # JDD
                )

                if coef_predict is not None:
                    # There is a prediction, so copy the coeficients and
                    # contributions for later display.
                    test_coefs_predict_lst.append(coef_predict.detach().cpu().numpy())
                    test_weighted_terms_lst.append(
                        weighted_terms.detach().cpu().numpy()
                    )

                # Record measured and predicted affinities
                test_results.append(output.detach().cpu().numpy())
                test_labels.append(
                    single_label_for_testing[:, 0].detach().cpu().numpy()
                )

            # Collect the testing statistics.
            test_results = np.array(test_results).flatten()
            test_labels = np.array(test_labels).flatten()
            val_rmse = np.sqrt(np.mean((test_results - test_labels) ** 2))
            if np.isinf(val_rmse):
                val_rmse = 1000
            val_ame = np.mean(np.abs(test_results - test_labels))

            pearson_coeff = pearsonr(test_results, test_labels)[0]
            print("Validation", epoch_idx, val_rmse, val_ame, pearson_coeff)
            test_mses.append(val_rmse)
            test_ames.append(val_ame)
            test_pearsons.append(pearson_coeff)

    return (
        model,
        test_labels,
        test_results,
        test_gninatypes_filenames,
        test_mses,
        test_ames,
        test_pearsons,
        training_losses,
        test_coefs_predict_lst,
        test_weighted_terms_lst,
        which_precalc_terms_to_keep,
        precalc_term_scales_to_keep,
    )


def weights_init(m: "AvgPool3d"):
    """Initialize weights of the model."""
    if isinstance(m, (nn.Conv3d, nn.Linear)):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0)


def get_args() -> dict:
    """Get arguments from the command line.
    
    Returns:
        args: A dictionary of the arguments from the command line.
    """
    params_info = [
        {
            "name": "epochs",
            "val": 250,  # 400,
            "description": "Number of epochs to train for.",
        },
        {"name": "fold_num", "val": 0, "description": "Which fold to train on."},
        {"name": "batch_size", "val": 25, "description": "Batch size."},
        {"name": "lr", "val": 0.01, "description": "Learning rate."},
        {
            "name": "step_size",
            "val": 80,
            "description": "Step size for learning rate decay.",
        },
        {
            "name": "prefix",
            "val": "randomsplit",
            "description": "Prefix for the input types files.",  # TODO: Correct description?
        },
        {
            "name": "termtypes",
            "val": "all",
            "description": "Which terms to use. Can be 'all', 'smina', or 'gaussian'.",
        },
        {
            "name": "data_dir",
            # "val": "./train/",
            "val": "./",
            "description": "Directory where the data is stored.",
        },
        {
            "name": "out_dir",
            "val": "./outputs/",
            "description": "Directory where the outputs are saved.",
        },
    ]

    # Create argparser with same args as params
    parser = argparse.ArgumentParser()
    for value in params_info:
        parser.add_argument(
            "--" + value["name"],
            type=type(value["val"]),
            default=value["val"],
            help=value["description"] + " Default: " + str(value["val"]),
        )
    args = parser.parse_args()
    return vars(args)  # Returns as dict


def validate_params(params: dict) -> dict:
    """Validate parameters. Also adjust parameter to ensure all works properly.
    
    Args:
        params: A dictionary of the parameters.
        
    Returns:
        params: The validated parameters.
    """
    # Make sure termtypes is valid
    if params["termtypes"] not in ["all", "smina", "gaussian"]:
        raise ValueError("termtypes must be 'all', 'smina', or 'gaussian'")

    # All directories should be absolute
    params["data_dir"] = os.path.abspath(params["data_dir"])
    params["out_dir"] = os.path.abspath(params["out_dir"])

    # The data_dir must exist
    if not os.path.isdir(params["data_dir"]):
        raise ValueError("data_dir does not exist")

    # Note that out_dir will be created later, so doesn't need to exist at this
    # point.

    print(params)

    return params

