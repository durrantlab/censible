# CENsible 1.0

## Introduction

CENsible uses context explanation networks (CENs) to predict small-molecule
binding affinities. Rather than predict a binding affinity directly, it predicts
the contributions of pre-calculated terms to the overall affinity, thus
providing interpretable output. CENsible insights are useful for subsequent lead
optimization.

We release CENsible under the terms of the [GNU General Public License
v3.0](https://github.com/durrantlab/censible/blob/main/LICENSE.md). The git
repository is available at
[https://github.com/durrantlab/censible](https://github.com/durrantlab/censible).
A working Google Colab demo is available at **URL HERE**.

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
so, run the script using the `--use_cpu` flag (see below).

### 4. Install _smina_

CENsible uses _smina_ to calculate the pre-calculated terms. Visit the [smina
repository](https://sourceforge.net/projects/smina/) to download the latest
version.

As of August 10, 2023, you can install _smina_ using Anaconda:

```bash
conda install -c conda-forge smina
```

### 5. Test the CENsible Installation

To test the installation, run the following command:

```bash
./test_predict.sh
```

**NOTE:** This script assumes _smina_ is in your PATH. Additionally, if you have
installed a version of Pytorch that does not support CUDA, you must edit the
`test_predict.sh` file to add the `--use_cpu` flag.

## Usage

### Simple Use

Here is a simple example of how to use CENsible for inference (prediction):

```bash
python predict.py --ligpath censible/data/test/1wdn_ligand.mol2 \
                  --recpath censible/data/test/1wdn_receptor.pdb \
                  --smina_exec_path /usr/local/bin/smina
```

**NOTE:** You should replace the `--ligpath` and `--recpath` arguments with the
path to your ligand and receptor files, respectively. You should also replace
the `--smina_exec_path` argument with the path to your smina executable.

### Saving CENsible Weights

In the above simple example, CENsible only outputs the predicted affinity. If
you also wish to output CENsible's predicted weights (as well as other
information used to calculate the final score), use the `--out` flag:

```bash
python predict.py --ligpath censible/data/test/1wdn_ligand.mol2 \
                  --recpath censible/data/test/1wdn_receptor.pdb \
                  --smina_exec_path /usr/local/bin/smina \
                  --out test_out.tsv
```

### Using Other CENsible Models

CENsible comes with a pre-trained model (described in the accompanying
manuscript, see `censible/data/model_allcen/`). If you wish to use your own
model, specify the path to the model directory using the `--model_dir` flag:

```bash
python predict.py --ligpath censible/data/test/1wdn_ligand.mol2 \
                  --recpath censible/data/test/1wdn_receptor.pdb \
                  --smina_exec_path /usr/local/bin/smina \
                  --model_dir ./my_model_dir/ \
                  --out test_out.tsv
```

The model directory should contain the following files:

-   `model.pt`: The trained model.
-   `precalc_term_scales.npy`: The pre-calculated term scales.
-   `which_precalc_terms_to_keep.npy`: The pre-calculated terms the model uses.

## CENsible Output

### Without the `--out` Flag

If you do not use the `--out` flag, CENsible will only report the predicted
affinity. For example:

```text
CENsible 1.0

receptor  censible/data/test/1wdn_receptor.pdb
ligand  censible/data/test/1wdn_ligand.mol2
model /mnt/Data/jdurrant/cenet/censible/data/model_allcen/

score:  5.81088 (1.55 µM)

WARNING: No output file specified (--out). Not saving weights and contributions.

=====================================
```

### With the `--out` Flag

If the user specifies the `--out` flag, CENsible will output the same
information to the specified Excel-compatible tab-delimited file. Additionally,
it will output the following:

- A text description of the pre-calculated terms the model uses.
- The pre-calculated terms themselves (calculated using _smina_).
- The pre-calculated terms after scaling/normalization.
- The weights the model assigns to each pre-calculated term.
- The predicted contribution of each pre-calculated term to the overall affinity
  (i.e., the product of the normalized pre-calculated term and its weight).

**NOTE:** The final affinity is the sum of the predicted contributions.
