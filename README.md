# README: CENsible - Predicting Small-Molecule Binding Affinities

## Introduction

CENsible uses context explanation networks (CENs) to predict small-molecule
binding affinities. Rather than predict the binding affinity directly, it
predicts the contributions of pre-calculated terms to the overall affinity,
providing an interpretable output. These insights are useful for subsequent lead
optimization.

## Prerequisites

1. **Install Python:**

We recommend using anaconda python to create a new environment. We have tested
CENsible on Python 3.9.16.

```bash
conda create -n censible python=3.9.16
conda activate censible
```

2. **Install Dependencies**

If you wish only to use CENsible for inference (prediction), install the
dependencies in the `requirements_predict.txt` file:

```bash
pip install -r requirements_predict.txt
```

3. SMINA executable, which is used for molecular docking simulations.

## Instructions to Use

1. **Clone the Repository (if applicable)**:

    ```python
    git clone <repository_link>
    cd <repository_directory>
    ```

2. **Prepare your Input Data**:
    - Have your ligand(s) and receptor files ready. The ligand(s) paths are
      expected as a list (even if there's only one), while the receptor path is
      a single file.

3. **Set up the Model**:
    - Ensure you have the trained model and other required files in a directory.
      By default, CENsible looks for `model.pt` and other files in the current
      directory. If you have them in another directory, you'll need to specify
      that using the `--model_dir` option.

4. **Run CENsible**:
    Use the command below, replacing placeholders with the appropriate paths:

    ```python
    python predict.py --ligpath <path_to_ligand1> <path_to_ligand2> ... --recpath <path_to_receptor> --model_dir <path_to_model_directory> --smina_exec_path <path_to_smina_executable> --out <path_to_output_tsv>
    ```

   Example:

   ```python
   python predict.py --ligpath ./ligands/ligand1.mol2 ./ligands/ligand2.mol2 --recpath ./receptor.pdb --model_dir ./models/ --smina_exec_path /usr/local/bin/smina --out ./output/results.tsv
   ```

5. **Check the Output**:
   - After running the script, if you provided an output path using `--out`, the
     results will be saved as a TSV file at the specified location.
   - If not, the results will be displayed in the terminal.

6. **Interpret the Results**:
   - The TSV output contains the predicted affinity for each ligand-receptor
     pair, followed by term-wise contributions.
   - Use these insights for subsequent lead optimization in drug design.

## Support

For any issues or questions, please refer to the FAQ section in the
documentation or reach out to the CENsible team.

---

We hope you find CENsible beneficial for your research and drug design
endeavors!
