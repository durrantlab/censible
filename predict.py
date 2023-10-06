"""
This module provides utilities for inferencing using the 'censible' package.

It primarily functions to process ligand data, apply the model, and generate
output in TSV (Tab-Separated Values) format.
"""

from censible.inference.inference import get_cmd_args, load_example, load_model, apply
from censible.inference.pdbs import save_pdbs_with_per_atom_gauss_vals_in_beta
from censible.inference.tsv_out_writer import TSVWriter

args = get_cmd_args()

# load the model
(model, smina_terms_mask, norm_factors_masked, smina_ordered_terms_names) = load_model(
    args.model_dir
)

print("")

all_tsv_output = ""

for lig_idx, lig_path in enumerate(args.ligpath):
    tsv_writer = TSVWriter(args, lig_path)

    # Load the data. TODO: One ligand at a time here for simplicity's sake.
    # Could batch to improve speed, I think.
    example = load_example(
        lig_path,
        args.recpath,
        args.smina_exec_path,
        smina_ordered_terms_names,
        args.obabel_exec_path,
    )

    (
        predicted_affinity,
        weights_predict,
        contributions_predict,
        smina_terms_masked,
    ) = apply(
        example,
        smina_terms_mask,
        norm_factors_masked,
        model,
        "cpu" if args.use_cpu else "cuda",
    )

    print("\n")
    tsv_writer.generate_summary(predicted_affinity)

    tsv_writer.generate_terms_weights_contributions(
        smina_ordered_terms_names,
        smina_terms_mask,
        smina_terms_masked,
        norm_factors_masked,
        weights_predict,
        contributions_predict,
    )

    all_tsv_output += tsv_writer.content

    if args.pdb_out != "":
        if lig_idx == 0:
            save_pdbs_with_per_atom_gauss_vals_in_beta(
                all_tsv_output,
                predicted_affinity,
                args.smina_exec_path,
                args.obabel_exec_path,
                lig_path,
                args.recpath,
                args.pdb_out,
            )
        else:
            print(
                "WARNING (--pdb_out): Only the first receptor/ligand complex will be saved as PDB files."
            )

if args.tsv_out != "":
    with open(args.tsv_out, "w") as f:
        # Report the receptor/ligand:
        f.write(all_tsv_output)
