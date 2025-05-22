<h1 align="center">CENsible</h1>

<p align="center">
    <a href="https://github.com/durrantlab/censible/releases">
        <img src="https://img.shields.io/github/v/release/durrantlab/censible" alt="GitHub release (latest by date)">
    </a>
    <a href="https://github.com/durrantlab/censible/blob/main/LICENSE" target="_blank">
        <img src="https://img.shields.io/github/license/durrantlab/censible" alt="License">
    </a>
    <a href="https://github.com/durrantlab/censible/" target="_blank">
        <img src="https://img.shields.io/github/repo-size/durrantlab/censible" alt="GitHub repo size">
    </a>
</p>

CENsible uses deep-learning context explanation networks (CENs) to predict
small-molecule binding affinities. Rather than predict a binding affinity
directly, it predicts the contributions of pre-calculated terms to the overall
affinity, thus providing interpretable output. CENsible insights are useful for
subsequent lead optimization.

## Installation

We recommend using [pixi](https://pixi.sh/latest/) to reproduce out Python virtual environments.
This will painlessly handle installing the appropriate Python, PyTorch, and CUDA versions.
Once you have pixi installed and the [repository cloned](https://github.com/durrantlab/censible), all you have to do is run the following command.

```bash
pixi install
```

Once that is finished, you can activate the new Python virtual environment.

```bash
pixi shell
```

To test the installation, run the following command.

```bash
./test_predict.sh
```

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

## Citation

If you use CENsible in your work, please cite the following literature.

Bhatt, R., Koes, D. R., & Durrant, J. D. (2024). CENsible: interpretable insights into small-molecule binding with Context Explanation Networks. *Journal of Chemical Information and Modeling, 64*(12), 4651-4660. DOI: [10.1021/acs.jcim.4c00825](https://doi.org/10.1021/acs.jcim.4c00825)

## License

This project is released under the GPL-3.0-only License as specified in `LICENSE.md`.
