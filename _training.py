import molgrid
import torch
import torch.optim as optim
# from _debug import grid_channel_to_xyz_file
import numpy as np
from scipy.stats import pearsonr
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from _preprocess import remove_rare_terms

def train_single_fold(
    Net,
    which_precalc_terms_to_keep,
    params,
    term_names
    # fold_num=0,
    # batch_size=25,
    # lr=0.01,
    # epochs=100,
    # step_size=80,
):
    # The main object. See
    # https://gnina.github.io/libmolgrid/python/index.html#the-gridmaker-class
    # TODO: No grid_center parameter here?
    gmaker = molgrid.GridMaker()  # use defaults

    # Create a training dataset, which has access to all receptor and ligand grids.
    train_dataset = molgrid.ExampleProvider(
        ligmolcache="lig.molcache2",
        recmolcache="rec.molcache2",
        shuffle=True,
        iteration_scheme=molgrid.IterationScheme.LargeEpoch,
        # default_batch_size=1
        default_batch_size=params["batch_size"],
        stratify_min=3,  # TODO: What do these mean?
        stratify_max=10,
        stratify_step=1,
        stratify_pos=0,
    )

    # Indicate that the training set will only use those grids in a given file,
    # not all grids.
    train_dataset.populate(params["prefix"] + ("train%d_cen.types" % params["fold_num"]))
    # train_dataset.populate("all_cen.types")

    # Get num labels in train_dataset
    # num_labels = train_dataset.num_labels()

    # Similarly create a testing dataset.
    test_dataset = molgrid.ExampleProvider(
        ligmolcache="lig.molcache2",
        recmolcache="rec.molcache2",
        iteration_scheme=molgrid.IterationScheme.LargeEpoch,
        default_batch_size=1,
    )
    test_dataset.populate(params["prefix"] + ("test%d_cen.types" % params["fold_num"]))

    # Note that alllabels doesn't contain all the labels, but is simply a tensor
    # where a given batch's labels will be placed.
    all_labels_for_training = torch.zeros(
        (params["batch_size"], train_dataset.num_labels()), dtype=torch.float32, device="cuda"
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
    for batch_index, train_batch in enumerate(train_dataset):
        train_batch.extract_labels(all_labels_for_training)
        tmp_labels.append(all_labels_for_training.cpu().numpy()[:, 1:])
    which_precalc_terms_to_keep = remove_rare_terms(np.vstack(tmp_labels), which_precalc_terms_to_keep)
    train_dataset.reset()

    precalc_term_scale_factors = jdd_normalize_inputs(train_dataset, which_precalc_terms_to_keep)  # JDD

    import pdb; pdb.set_trace()

    # Instead, precalc_term_scale_factors is 1 for all terms.
    precalc_term_scale_factors = np.ones((1, which_precalc_terms_to_keep.sum()))
    precalc_term_scale_factors = torch.from_numpy(precalc_term_scale_factors).to("cuda").float()



    # Create tensors to hold the inputs.
    dims = gmaker.grid_dimensions(train_dataset.num_types())
    tensor_shape = (params["batch_size"],) + dims  # shape of batched input
    input_tensor_for_training = torch.zeros(tensor_shape, dtype=torch.float32, device="cuda")
    single_input_for_testing = torch.zeros((1,) + dims, dtype=torch.float32, device="cuda")

    # Create the model.
    nterms = np.count_nonzero(which_precalc_terms_to_keep)
    model = Net(dims, nterms).to("cuda")
    model.apply(weights_init)

    # Setup optimizer and scheduler for training.
    optimizer = optim.SGD(model.parameters(), lr=params["lr"], weight_decay=0.0001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params["step_size"], gamma=0.1)

    # Keep track of various metrics as training progresses.
    training_losses = []
    test_mses = []
    test_ames = []
    test_pearsons = []
    coefs_all = []

    for epoch_idx in range(params["epochs"]):
        # Loop through the batches (25 examples each)
        cnt = 0
        # import pdb; pdb.set_trace()
        while True:
            try:
                training_finished = False
                for batch_idx, train_batch in enumerate(train_dataset):
                    train_batch.extract_labels(all_labels_for_training)
                    affinity_label_for_training = all_labels_for_training[:, 0]  # affinity only

                    # Keep only ones that are which_precalc_terms_to_keep (see _preprocess.py)
                    precalculated_terms = all_labels_for_training[:, 1:][:, which_precalc_terms_to_keep]

                    # Scale so never outside of -1 to 1
                    precalculated_terms = precalc_term_scale_factors * precalculated_terms  # JDD

                    # print(float(smina_terms.max()), float(smina_terms.min()))
                    # import pdb; pdb.set_trace()

                    cnt += precalculated_terms.size()[0]

                    # Get the grid (populates input_tensor_for_training)
                    gmaker.forward(
                        train_batch, input_tensor_for_training, random_translation=2, random_rotation=True
                    )

                    # grid_channel_to_xyz_file(input_tensor[0][0])
                    # print(batch_idx)
                    # print("")
                    # grid_channel_to_xyz_file(input_tensor[21][0])
                    # break

                    # Get the output for this batch. output[0] is output tensor.
                    # output[1] is None for some reason. Note that weighted_terms is the
                    # pre-calculated terms times the coefficients.
                    output, coef_predict, weighted_terms = model(input_tensor_for_training, precalculated_terms)

                    # if batch_idx == 0:
                    #     output[0].detach_().cpu().numpy()[:-5]

                    # Print the output, after bringing it to cpu
                    # print(output.cpu().detach().numpy().T[0,:5])

                    training_loss = F.smooth_l1_loss(output.flatten(), affinity_label_for_training.flatten())
                    training_loss.backward()

                    # print(loss)

                    # clip gradients
                    nn.utils.clip_grad_norm_(model.parameters(), 1)

                    optimizer.step()

                    training_losses.append(float(training_loss))
                    training_finished = True
                    break
                if training_finished:
                    break
            except StopIteration:
                print("Reseting database...")
                train_dataset.reset()  # Reset the dataset for the next epoch

        # So you evaluate on test set after each epoch
        # print(cnt)

        # TODO: Note that cnt above is not the same as train_dataset.size(). I
        # checked the vectors in train_dataset, and there are many repeats. If
        # you consider the unique values, it does give a number very close to
        # train_dataset.size(). Need to get to bottom of this. Perhaps
        # duplicates as part of a balancing procedure?

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
                    single_label_for_testing[:, 1:][:, which_precalc_terms_to_keep] * precalc_term_scale_factors  # JDD
                )

                if coef_predict is not None:
                    # There is a prediction, so copy the coeficients and
                    # contributions for later display.
                    test_coefs_predict_lst.append(coef_predict.detach().cpu().numpy())
                    test_weighted_terms_lst.append(weighted_terms.detach().cpu().numpy())


                # Record measured and predicted affinities
                test_results.append(output.detach().cpu().numpy())
                test_labels.append(single_label_for_testing[:, 0].detach().cpu().numpy())

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
        test_mses,
        test_ames,
        test_pearsons,
        training_losses,
        test_coefs_predict_lst,
        test_weighted_terms_lst,
        which_precalc_terms_to_keep
    )

def jdd_normalize_inputs(train_dataset, which_precalc_terms_to_keep):  # JDD
    # TODO: Need to implement ability tosave values in factors and load them
    # back in for inference.

    MAX_VAL_AFTER_NORM = 1.0

    # Get all the labels into a numpy array
    batch = train_dataset.next_batch(train_dataset.size())
    batch_labels = np.array(
        [[v for v in item.labels] for item in batch]
    )

    # Normalize the columns so the values go between 0 and 1
    precalc_term_scale_factors = np.zeros(batch_labels.shape[1])
    for i in range(batch_labels.shape[1]):
        col = batch_labels[:, i]
        max_abs = np.max(np.abs(col))
        precalc_term_scale_factors[i] = 1.0

        if max_abs > 0:
            precalc_term_scale_factors[i] = MAX_VAL_AFTER_NORM * 1.0 / max_abs

    # Note that first column is affinity. No need to normalize that. Just save
    # normalization factors on smina terms.
    precalc_term_scale_factors = precalc_term_scale_factors[1:]

    # Also good to keep only those that are goodfeatures.
    precalc_term_scale_factors = precalc_term_scale_factors[which_precalc_terms_to_keep]

    # Save factors
    # np.save("batch_labels.jdd.npy", batch_labels[:, 1:][:, goodfeatures])
    # np.save("factors.jdd.npy", factors)

    # import pdb; pdb.set_trace()

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

    train_dataset.reset()

    return precalc_term_scale_factors

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0)
