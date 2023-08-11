from censible.inference.inference import get_cmd_args, load_example, load_model, apply
from censible.inference.tsv_out_writer import TSVWriter

args = get_cmd_args()

# load the model
(model, smina_terms_mask, norm_factors_masked, smina_ordered_terms_names) = load_model(
    args.model_dir
)

print("")

all_tsv_output = ""

for lig_path in args.ligpath:
    tsv_writer = TSVWriter(args, lig_path)

    # Load the data. TODO: One ligand at a time here for simplicity's sake.
    # Could batch to improve speed, I think.
    example = load_example(
        lig_path, args.recpath, args.smina_exec_path, smina_ordered_terms_names
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

if args.out != "":
    with open(args.out, "w") as f:
        # Report the receptor/ligand:
        f.write(all_tsv_output)
