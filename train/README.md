# README

This directory contains scripts to train a CENsible model. We provide these
scripts as a convenience and for reproducibility's sake. You may need to modify
them to work on your computer system.

## Instructions

### `1.downloads.md`

First, you must download the pdbbind 2020 dataset. Place this data in these
directories:

- `./pdbbind2020/refined-set/`
- `./pdbbind2020/v2020-other-PL/`

You should also download the `smina` executable. See `./1.downloads.md` for more
details.

### `2.make_regular_types_file.py`

```bash
python3 2.make_regular_types_file.py
```

This Python script processes the affinity data associated with the PDBbind
protein-ligand complexes and saves it in a standardized format (`all.types`). It
reads the affinity data from the dataset, processes or removes affinities that
are not precisely defined, and converts all affinity values to a standard
logarithmic scale, pK(affinity).

### `3.add_smina_types.py`

```bash
python3 3.add_smina_types.py
```

This Python script further processes the `all.types` file by using `smina` to
add custom (pre-calculated) scoring terms to each entry. The `allterms.txt` file
contains the specific scoring terms. The script writes the results to a new file
named `all_cen.types`.

### `4.make_gninatypes.sh`

```bash
./4.make_gninatypes.sh
```

This bash script uses the `gninatyper` tool to process the protein and ligand
files in the `pdbbind/` directory. For each `.pdb` protein file and `.sdf`
ligand file in the `pdbbind/` directory, `gninatyper` runs to create the
corresponding `.gninatypes` file, effectively converting all identified protein
and ligand files into their `.gninatypes` representation.

### `5.make_molcache2.sh`

```bash
./5.make_molcache2.sh
```

This script downloads the `create_caches2.py` script from GitHub and runs it to
process the `all_cen.types` file. It creates the `rec.molcache2` and
`lig.molcache2` cache files for quick access to the protein and ligand
representations.

### `6.finalize_splits.py`

```bash
python3 6.finalize_splits.py
```

This Python script divides the entries in the `all_cen.types` file into three
folds. It randomly shuffles the lines from `all_cen.types` and divides them into
three roughly equal parts. The script creates three distinct pairs of training
and testing datasets from these parts; each pair consists of two combined parts
for training and the remaining part for testing. It saves the three resultant
datasets (folds) to the disk.

### `7.get_smina_ordered_terms.sh`

```bash
./7.get_smina_ordered_terms.sh
```

This bash script uses the `smina.static` executable with a custom scoring option
(see `allterms.txt`) to score a specific protein-ligand complex. It filters the
`smina` output to extract the order of the `smina` terms and saves the ordered
list to a file named `smina_ordered_terms.txt`.

### `8.train.sh`

```bash
./8.train.sh
```

This bash script uses the `train.py` script to train a CENsible model.
