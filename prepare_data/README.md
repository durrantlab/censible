# README

This directory contains scripts to download and prepare data for the cenet
training.

## Instructions

First, you must download the pdbbind 2020 dataset. This data should be placed in
these directories:

* `./1.pdbbind2020/refined-set/`
* `./1.pdbbind2020/v2020-other-PL/`

Second, run the `2.make_three_fold_divisions.py` script. This scripts generates
three independent subsets of the data, where similar proteins and identical
ligands are not repeated in the same subset. Not all protein/ligand complexes
will be placed. The three subsets are stored in `pdbid_bins.json`.

Third, run the `3.make_regular_types_file.py` script. This script creates the
`all.types` file, which includes the experimental affinity of each
protein/ligand complex, as well as the locations of the receptor and ligand
`gninatypes` files (not yet created).

Fourth, run the `4.add_smina_types.py` script. This will use the `smina` tool to
add additional information about each complex. The output will be the
`all_cen.types` files. Note that you will need to install `smina` separately,
and will need to edit the path to `smina` in the `4.add_smina_types.py` script.

Fifth, run the `5.make_gninatypes.sh`. This will create the gninatypes files for
each protein and ligand. You'll need to install `gninatyper` separately.

Sixth, run the `6.make_molcache2.sh` script. This will create the `molcache2`
files for the proteins and ligands. 