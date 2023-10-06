"""
This module provides a TSVWriter class.

The class creates Tab-Separated Values (TSV) output for molecular data,
specifically with respect to terms, weights, and contributions in a format
suitable for visualization and analysis.
"""

import os
import argparse
from typing import List
import numpy as np
import torch
from censible.inference.term_descriptions import full_term_description


class TSVWriter:
    """A class for writing the TSV output."""

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
        """Generate the summary for the TSV output.
        
        Args:
            predicted_affinity (torch.Tensor): The predicted affinity.
        """
        summary = "CENsible 1.0\n\n"
        summary += f"receptor: {self.args.recpath}\n"
        summary += f"ligand:   {self.lig_path}\n"
        summary += f"model:    {self.args.model_dir}\n\n"

        predicted_affinity = float(predicted_affinity)

        # affinity is -log10(Kd). Convert back to Kd, using fM, pM, nM, µM, mM,
        # M, etc.
        kd = 10 ** (-predicted_affinity)
        if kd < 1e-12:
            kd = f"{kd*1e15:.2f} fM"
        elif kd < 1e-9:
            kd = f"{kd*1e12:.2f} pM"
        elif kd < 1e-6:
            kd = f"{kd*1e9:.2f} nM"
        elif kd < 1e-3:
            kd = f"{kd*1e6:.2f} µM"
        elif kd < 1:
            kd = f"{kd*1e3:.2f} mM"
        else:
            kd = f"{kd:.2f} M"

        summary += f"score:    {round(predicted_affinity, 5)} ({kd})\n"

        if self.args.tsv_out != "":
            summary += f"\nSee {self.args.tsv_out} for predicted weights and contributions."
        else:
            summary += "\nWARNING: No output file specified (--tsv_out). Not saving weights and contributions."

        if self.args.pdb_out:
            summary += f"\n\nSee {self.args.pdb_out} for PDB files with atom_type_gaussian contributions in the beta columns."

        summary += "\n\n" + self.bar + "\n"

        self.summary = summary

        print(summary.strip())

    def generate_terms_weights_contributions(
        self,
        smina_ordered_terms_names: List[str],
        smina_terms_mask: np.ndarray,
        smina_terms_masked: np.ndarray,
        norm_factors_masked: np.ndarray,
        weights_predict: np.ndarray,
        contributions_predict: np.ndarray,
    ):
        """Generate the terms, weights, and contributions for the TSV output.
        
        Args:
            smina_ordered_terms_names (List[str]): The ordered terms names.
            smina_terms_mask (np.ndarray): The mask for the terms.
            smina_terms_masked (np.ndarray): The masked terms.
            norm_factors_masked (np.ndarray): The masked normalization factors.
            weights_predict (np.ndarray): The predicted weights.
            contributions_predict (np.ndarray): The predicted contributions.
        """
        if self.args.tsv_out == "":
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
        """Return the content of the TSV output.

        Returns:
            str: The content of the TSV output.
        """
        return self.summary + self.terms_weights_contributions

