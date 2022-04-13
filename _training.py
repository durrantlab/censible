import molgrid
import torch
import torch.optim as optim
from _debug import grid_channel_to_xyz_file
import numpy as np
from scipy.stats import pearsonr
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def train_single_fold(
    Net, goodfeatures, fold_num=0, batch_size=25, lr=0.01, epochs=100, step_size=80
):
    gmaker = molgrid.GridMaker()  # use defaults
    train_dataset = molgrid.ExampleProvider(
        ligmolcache="lig.molcache2",
        recmolcache="rec.molcache2",
        shuffle=True,
        iteration_scheme=molgrid.IterationScheme.LargeEpoch,
        default_batch_size=batch_size,
        stratify_min=3,
        stratify_max=10,
        stratify_step=1,
        stratify_pos=0,
    )
    train_dataset.populate("crystaltrain%d_cen.types" % fold_num)

    test_dataset = molgrid.ExampleProvider(
        ligmolcache="lig.molcache2",
        recmolcache="rec.molcache2",
        iteration_scheme=molgrid.IterationScheme.LargeEpoch,
        default_batch_size=1,
    )
    test_dataset.populate("crystaltest%d_cen.types" % fold_num)

    dims = gmaker.grid_dimensions(train_dataset.num_types())
    tensor_shape = (batch_size,) + dims  # shape of batched input
    input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device="cuda")
    single_input = torch.zeros((1,) + dims, dtype=torch.float32, device="cuda")

    nterms = np.count_nonzero(goodfeatures)
    model = Net(dims, nterms).to("cuda")
    model.apply(weights_init)

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    alllabels = torch.zeros(
        (batch_size, train_dataset.num_labels()), dtype=torch.float32, device="cuda"
    )
    single_label = torch.zeros(
        (1, train_dataset.num_labels()), dtype=torch.float32, device="cuda"
    )

    losses = []
    mses = []
    ames = []
    pearsons = []
    coefs_all = []

    for epoch_idx in range(epochs):
        # Loop through the batches (25 examples each)
        for batch_idx, batch in enumerate(train_dataset):
            batch.extract_labels(alllabels)
            label = alllabels[:, 0]  # affinity only
            smina_terms = alllabels[:, 1:][:, goodfeatures]

            # Get the grid (input_tensor)
            gmaker.forward(
                batch, input_tensor, random_translation=2, random_rotation=True
            )

            # grid_channel_to_xyz_file(input_tensor[0][0])
            # print(batch_idx)
            # print("")
            # grid_channel_to_xyz_file(input_tensor[21][0])
            # break

            # Get the output for this batch. output[0] is output tensor.
            # output[1] is None for some reason.
            output, coef_predict, contributions = model(input_tensor, smina_terms)

            # if batch_idx == 0:
            #     output[0].detach_().cpu().numpy()[:-5]

            # Print the output, after bringing it to cpu
            # print(output.cpu().detach().numpy().T[0,:5])

            loss = F.smooth_l1_loss(output.flatten(), label.flatten())
            loss.backward()

            # print(loss)

            # clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            losses.append(float(loss))

        # So you evaluate on test set after each epoch

        scheduler.step()

        # eval performance on test set
        with torch.no_grad():
            results = []
            testlabels = []
            coefs_predict_lst = []
            contributions_lst = []
            for batch_index, batch in enumerate(test_dataset):
                batch.extract_labels(single_label)
                gmaker.forward(batch, single_input)

                output, coef_predict, contributions = model(single_input, single_label[:, 1:][:, goodfeatures])
                if coef_predict is not None:
                    coefs_predict_lst.append(coef_predict.detach().cpu().numpy())
                    contributions_lst.append(contributions.detach().cpu().numpy())

                    # if batch_index in [0, 1, 2, 3]:
                        # print(coef_predict.detach().cpu().numpy()[0,:5])
                        # first_channel = single_input[0][0]
                        # print(batch_index)
                        # grid_channel_to_xyz_file(first_channel)
                        # print("========")

                results.append(output.detach().cpu().numpy())
                testlabels.append(single_label[:, 0].detach().cpu().numpy())

            results = np.array(results).flatten()
            labels = np.array(testlabels).flatten()
            val_rmse = np.sqrt(np.mean((results - labels) ** 2))
            if np.isinf(val_rmse):
                val_rmse = 1000
            val_ame = np.mean(np.abs(results - labels))
            pearson_coeff = pearsonr(results, labels)[0]
            print("Validation", epoch_idx, val_rmse, val_ame, pearson_coeff)
            mses.append(val_rmse)
            ames.append(val_ame)
            pearsons.append(pearson_coeff)

    return model, labels, results, mses, ames, pearsons, losses, coefs_predict_lst, contributions_lst


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
