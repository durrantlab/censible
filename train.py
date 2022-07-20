import sys, re, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from _training import train_single_fold
from _graphs import generate_graphs

# Published performance for this model on this set is 1.5 RMSE and 0.7 Pearson's
# R, so we are pretty close (could train longer).
from published_model import Net

from CEN_model import CENet
import molgrid
from _preprocess import preprocess

# import py3Dmol
from scipy.stats import pearsonr
import os

orig_dir = os.getcwd() + os.sep

# change working directory to "./data/cen/"
os.chdir("./data/cen/")

# allct = np.loadtxt("all_cen.types", max_rows=5, dtype=str)

goodfeatures, termnames = preprocess()

model, labels, results, mses, ames, pearsons, losses, coefs_predict_lst, contributions_lst = train_single_fold(
    CENet, goodfeatures, epochs=400, use_ligands=True,
    # lr=0.0001
)

generate_graphs(orig_dir, losses, labels, results, pearsons, coefs_predict_lst, contributions_lst, goodfeatures, termnames)