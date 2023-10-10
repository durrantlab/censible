# CENsible 1.0

## Introduction

CENsible uses deep-learning context explanation networks (CENs) to predict
small-molecule binding affinities. Rather than predict a binding affinity
directly, it predicts the contributions of pre-calculated terms to the overall
affinity, thus providing interpretable output. CENsible insights are useful for
subsequent lead optimization.

We release CENsible under the terms of the [GNU General Public License
v3.0](https://github.com/durrantlab/censible/blob/main/LICENSE.md). The git
repository is available at
[https://github.com/durrantlab/censible](https://github.com/durrantlab/censible).
A [Google Colab is also
available](https://durrantlab.pitt.edu/apps/censible/web/) for easy testing.

These instructions describe how to use CENsible for inference (prediction). If
you wish to train your own CENsible model, see `train/README.md` for some tips.

## Installation

### 1. Install Python

We recommend using anaconda python to create a new environment. We have tested
CENsible on Python 3.9.16.

```bash
conda create -n censible python=3.9.16
conda activate censible
```

### 2. Clone the Repository

```bash
git clone https://github.com/durrantlab/censible.git
cd censible
```

### 3. Install Dependencies

If you wish only to use CENsible for inference (prediction), install the
dependencies in the `requirements_predict.txt` file:

```bash
pip install -r requirements_predict.txt
```

**NOTE:** If you don't have CUDA installed on your system, you may need to edit
the `requirements_predict.txt` file to install the CPU version of PyTorch. If
so, run the censible script using the `--use_cpu` flag (see below).

### 4. Install _smina_

CENsible uses _smina_ to calculate the pre-calculated terms. Visit the [smina
website](https://sourceforge.net/projects/smina/) to download the latest
version.

You can also install _smina_ using Anaconda:

```bash
conda install -c conda-forge smina
```

We used the Oct 15 2019 version of _smina_ (based onAutoDock Vina 1.1.2) to
calculate terms for testing and training, though we expect other versions will
work equally well.

### 5. Install _Open Babel_

CENsible uses _Open Babel_ to standardize the user-provided protein and
small-molecule files. Visit the [Open Babel
repository](https://github.com/openbabel/openbabel) to download the latest
version.

You can also install _Open Babel_ using Anaconda:

```bash
conda install -c conda-forge openbabel
```

We used Open Babel 3.0.0 (Mar 11 2020) for training and testing. We rescored the
virtual screens described in our manuscript using Babel 3.1.0 (Oct 28 2022),
which also worked well.

### 6. Test the CENsible Installation

To test the installation, run the following command:

```bash
./test_predict.sh
```

**NOTE:** This script assumes _smina_ and _obaebl_ are in your PATH.
Additionally, if you have installed a version of Pytorch that does not support
CUDA, you must edit the `test_predict.sh` file to add the `--use_cpu` flag.

## Usage

### Simple Use

Here is a simple example of how to use CENsible for inference (prediction):

```bash
python predict.py --ligpath censible/data/test/1wdn_ligand.mol2 \
                  --recpath censible/data/test/1wdn_receptor.pdb \
                  --smina_exec_path /usr/local/bin/smina \
                  --obabel_exec_path /usr/local/bin/obabel
```

**NOTE:** You should replace the `--ligpath` and `--recpath` arguments with the
path to your ligand and receptor files, respectively. You should also replace
the `--smina_exec_path` and `--obabel_exec_path` arguments with the paths to
your smina and obabel executables, respectively.

### Saving CENsible Weights

In the above simple example, CENsible only outputs the predicted affinity. If
you also wish to output CENsible's predicted weights (as well as other
information used to calculate the final score), use the `--tsv_out` flag:

```bash
python predict.py --ligpath censible/data/test/1wdn_ligand.mol2 \
                  --recpath censible/data/test/1wdn_receptor.pdb \
                  --smina_exec_path /usr/local/bin/smina \
                  --obabel_exec_path /usr/local/bin/obabel \
                  --tsv_out test_out.tsv
```

CENsible will output affinity to `test_out.tsv`, an Excel-compatible
tab-delimited file. It will also output the following additional information:

- A text description of the pre-calculated terms the model uses.
- The pre-calculated terms themselves (calculated using _smina_).
- The pre-calculated terms after scaling/normalization.
- The weights the model assigns to each pre-calculated term.
- The predicted contribution of each pre-calculated term to the overall affinity
  (i.e., the product of the normalized pre-calculated term and its weight).

**NOTE:** The final affinity is the sum of the predicted contributions.

### Saving Per-Atom Gaussian Steric Terms to a PDB File

CENsible can also output the per-atom contributions associated with _smina_'s
Gaussian steric (`atom_type_gaussian`) terms to a PDB file. Use the `--pdb_out`
flag. For example:

```bash
python predict.py --ligpath censible/data/test/1wdn_ligand.mol2 \
                  --recpath censible/data/test/1wdn_receptor.pdb \
                  --smina_exec_path /usr/local/bin/smina \
                  --obabel_exec_path /usr/local/bin/obabel \
                  --tsv_out test_out.tsv \
                  --pdb_out test_out.pdb
```

The per-atom terms are placed in the beta column. See the `HEADER` fields of the
output PDB file for additional useful information.

### Using Other CENsible Models

CENsible comes with a pre-trained model (described in the accompanying
manuscript, see `censible/data/model_allcen3/`). If you wish to use your own
model, specify the path to the model directory using the `--model_dir` flag:

```bash
python predict.py --ligpath censible/data/test/1wdn_ligand.mol2 \
                  --recpath censible/data/test/1wdn_receptor.pdb \
                  --smina_exec_path /usr/local/bin/smina \
                  --obabel_exec_path /usr/local/bin/obabel \
                  --model_dir ./my_model_dir/ \
                  --tsv_out test_out.tsv
```

The model directory should contain the following files:

- `model.pt`: The trained model.
- `precalc_term_scales.npy`: The pre-calculated term scales.
- `which_precalc_terms_to_keep.npy`: The pre-calculated terms the model uses.
