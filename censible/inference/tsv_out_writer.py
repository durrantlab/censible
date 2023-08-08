import argparse
import numpy as np
import torch
from censible.inference.term_descriptions import full_term_description


class TSVWriter:
    bar = "=====================================\n"
    tsv_output = ""
    args = None
    summary = ""
    terms_weights_contributions = ""

    def __init__(self, args: argparse.Namespace, lig_path: str):
        """Initialize the TSVWriter.

        Args:
            args (argparse.Namespace): The arguments from argparse.
        """

        self.args = args
        self.lig_path = lig_path

    def generate_summary(self, predicted_affinity: torch.Tensor):
        summary = f"receptor\t{self.args.recpath}\n"
        summary += f"ligand\t{self.lig_path}\n"
        summary += f"model\t{self.args.model_dir}\n\n"
        summary += f"predicted_affinity\t{round(float(predicted_affinity), 5)}\n"

        if self.args.out != "":
            summary += f"\nSee {self.args.out} for predicted weights and contributions."
        else:
            summary += "\nWARNING: No output file specified (--out). Not saving weights and contributions."

        summary += "\n\n" + self.bar + "\n"

        self.summary = summary

        print(summary)

    def generate_terms_weights_contributions(
        self,
        smina_ordered_terms_names,
        smina_terms_mask,
        smina_terms_masked,
        norm_factors_masked,
        weights_predict,
        contributions_predict,
    ):
        if self.args.out == "":
            # If not specifying an output file, don't bother with the rest.
            return

        # If you're going to print out the specific terms, you need to get the
        # names of only those in the mask.
        smina_ordered_terms_names_masked = np.array(smina_ordered_terms_names)[
            smina_terms_mask
        ]

        # If specifying an output file, provide additional information and save.
        tsv_output = "\t" + "\t".join(smina_ordered_terms_names_masked) + "\n"

        tsv_output += (
            "\t"
            + "\t".join(
                [full_term_description(t) for t in smina_ordered_terms_names_masked]
            )
            + "\n"
        )

        for name in smina_ordered_terms_names_masked:
            full_term_description(name)

        tsv_output += (
            "precalc_smina_terms\t"
            + "\t".join([str(round(x, 5)) for x in smina_terms_masked])
            + "\n"
        )

        # tsv_output += "Precalc-term normalization scales\t" + "\t".join(
        #     [str(round(x, 5)) for x in norm_factors_masked]
        # ) + "\n"

        tsv_output += (
            "normalized_precalc_smina_terms\t"
            + "\t".join(
                [str(round(x, 5)) for x in smina_terms_masked * norm_factors_masked]
            )
            + "\n"
        )
        tsv_output += (
            "predicted_weights\t"
            + "\t".join([str(round(x, 5)) for x in weights_predict])
            + "\n"
        )
        tsv_output += (
            "predicted_contributions\t"
            + "\t".join([str(round(x, 5)) for x in contributions_predict])
            + "\n\n"
        )

        tsv_output += self.bar

        self.terms_weights_contributions = tsv_output

    @property
    def content(self):
        return self.summary + self.terms_weights_contributions

