from .consts import to_censible_pair
import numpy as np
import os
from typing import Tuple
import re


class PDBParser:
    """Parses a PDB file and assigns atom types using smina."""

    def __init__(
        self,
        filename: str,
        smina_exec_path: str,
        obabel_exec_path: str,
        do_type: bool = True,
    ):
        """The init function.
        
        Args:
            filename (str): The filename to parse (PDB format
            smina_exec_path (str): The path to the smina executable.
            obabel_exec_path (str): The path to the obabel executable.
            do_type (bool): Whether to assign atom types.
        """
        self.filename = filename
        self.atoms = []
        self.coordinates = []
        self.do_type = do_type
        self.parse_file(smina_exec_path, obabel_exec_path)

    def parse_file(self, smina_exec_path: str, obabel_exec_path: str):
        """Parses the PDB file and assigns atom types using smina.

        Args:
            smina_exec_path (str): The path to the smina executable.
            obabel_exec_path (str): The path to the obabel executable.
        """
        filename_actual = self.filename

        # If self.filename doens't end in ".pdb", then we need to convert it to
        # PDB format first using obabel. Note that the assumption here is that
        # it has already been properly pronated.
        if not self.filename.lower().endswith(".pdb"):
            filename_actual = filename_actual + ".tmp.pdb"
            cmd = f"{obabel_exec_path} {self.filename} -O {filename_actual} > /dev/null"
            os.system(cmd)

        # Save orig_content for reference
        with open(filename_actual, "r") as f:
            self.orig_content = f.read()

        lines = self.orig_content.split("\n")
        atom_idx = 0
        for line in lines:
            if "HOH" in line or "WAT" in line or "TIP" in line:
                # No water molecules
                continue

            if line.startswith(("ATOM", "HETATM")):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atom_data = {
                    "atom_idx": atom_idx,
                    "record": line[0:6].strip(),
                    "atom_num": int(line[6:11].strip()),
                    "atom_name": line[12:16],  # don't strip
                    "alt_loc": line[16].strip(),
                    "res_name": line[17:20].strip(),
                    "chain_id": line[21].strip(),
                    "res_num": int(line[22:26].strip()),
                    "iCode": line[26].strip(),
                    "occupancy": float(line[54:60].strip())
                    if line[54:60].strip()
                    else None,
                    "temp_factor": float(line[60:66].strip())
                    if line[60:66].strip()
                    else None,
                    "element": line[76:78],  # don't strip
                    "charge": line[78:80].strip(),
                    "type": "",
                    "type_pairs": {},
                }
                atom_idx = atom_idx + 1
                self.atoms.append(atom_data)
                self.coordinates.append([x, y, z])

        self.coordinates = np.array(self.coordinates)

        if self.do_type:
            self.assign_atom_types(smina_exec_path)

            # Now keep only those atoms with types. Remove also the corresponding
            # entries in self.coordinates
            self.atoms = [a for a in self.atoms if a["type"] != ""]
            self.coordinates = self.coordinates[[a["atom_idx"] for a in self.atoms]]

        # Clean up if needed
        if not self.filename.lower().endswith(".pdb"):
            os.remove(filename_actual)

    def get_from_orig_pdb(self, chain_replacement: str = None) -> str:
        """Loads from the originalPDB file, replacing the chain, but not doing
        any further processing.
        
        Args:
            chain_replacement (str): The chain replacement character. Keep
                original if None.
            
        Returns:
            str: The PDB file contents.
        """
        lines = self.orig_content.split("\n")
        lines = [l for l in lines if l.startswith("ATOM") or l.startswith("HETATM")]
        if chain_replacement is not None:
            lines = [l[:21] + chain_replacement + l[22:] for l in lines]
        return ("\n".join(lines)).strip()

    def total_gauss(self, type_pair: Tuple) -> float:
        """Returns the total Gaussian value (across all atoms) for the given
        type pair.
        
        Args:
            type_pair (Tuple): A tuple of two atom types.
            
        Returns:
            float: The total Gaussian value for the given type pair.
        """
        type_pair = to_censible_pair(type_pair)
        return sum(
            [
                a["type_pairs"][type_pair] if type_pair in a["type_pairs"] else 0
                for a in self.atoms
            ]
        )

    def get_pdb_text(
        self,
        type_pairs_for_beta: Tuple = None,
        beta_scale: float = 1,
        chain_id: str = None,
    ) -> str:
        """Returns the PDB text.
        
        Args:
            type_pairs_for_beta (Tuple): If not None, then the beta column will
                contain the Gaussian value for the given type pair.
            beta_scale (float): The value to scale the Gaussian value by.
            chain_id (str): The chain. Keep original if None.
            
        Returns:
            str: The PDB text.
        """
        pdb_txt = ""
        for i, atom in enumerate(self.atoms):
            if type_pairs_for_beta is not None:
                # The user has specified a specific pair
                type_pairs_for_beta = to_censible_pair(type_pairs_for_beta)
                if type_pairs_for_beta in atom["type_pairs"]:
                    # This atom has a value for the given type pair
                    beta_val = atom["type_pairs"][type_pairs_for_beta] * beta_scale
                    beta_val = round(beta_val, 2)  # PDB allows only 2 decimal places
                    # if beta_val != 0:
                    # Could be 0 because of rounding. Don't include in PDB
                    # if rounds to 0.
                    pdb_txt += self.make_pdb_line(i, beta_val, chain_id) + "\n"
            else:
                # The user has not specified a specific pair. Use all atoms.
                pdb_txt += self.make_pdb_line(i, 0, chain_id) + "\n"

        return pdb_txt

    def save_pdb(
        self, filename: str, type_pairs_for_beta: Tuple = None, beta_scale: float = 1
    ):
        """Saves the PDB file.

        Args:
            filename (str): The filename to save to.
            type_pairs_for_beta (Tuple): If not None, then the beta column will
                contain the Gaussian value for the given type pair.
            beta_scale (float): The value to scale the Gaussian value by.
        """
        pdb_txt = self.get_pdb_text(type_pairs_for_beta, beta_scale)
        with open(filename, "w") as f:
            f.write(pdb_txt)

    def make_pdb_line(self, atom_idx: int, beta_val: float, chain_id: str = None):
        """Makes a PDB line for the given atom index and beta value.
        
        Args:
            atom_idx (int): The atom index.
            beta_val (float): The beta value.
            chain_id (str): The chain. Keep original if None.
        """
        atom = self.atoms[atom_idx]
        atom_num = atom["atom_num"]
        atom_name = atom["atom_name"]
        res_name = atom["res_name"]
        chain_id = atom["chain_id"] if chain_id is None else chain_id
        res_num = atom["res_num"]
        x, y, z = self.coordinates[atom_idx]
        # atom_type = atom["type"]
        element = atom["element"]
        # radius = TYPES_TO_RADIUS[atom_type]
        return f"ATOM  {atom_num:5d} {atom_name} {res_name:3s} {chain_id:1s}{res_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{beta_val:6.2f}          {element}  "  # {radius:6.3f}"

    def assign_atom_types(self, smina_exec: str):
        """Assigns atom types using smina.
        
        Args:
            smina_exec (str): The path to the smina executable.
        """
        cmd = f"{smina_exec} --atom_terms {self.filename}.types.txt -r {self.filename} -l {self.filename} --score_only > /dev/null"
        os.system(cmd)
        with open(f"{self.filename}.types.txt") as f:
            lines = f.readlines()
        os.remove(f"{self.filename}.types.txt")
        lines = [l for l in lines if "<" in l]
        types = []
        coords = []
        for line in lines:
            # Regular expression pattern to extract the type and coordinates
            pattern = r"(\w+)\s+<(-?[\d\.]+),(-?[\d\.]+),(-?[\d\.]+)>"
            match = re.search(pattern, line)
            if match:
                typ = match.group(1)
                if typ == "PolarHydrogen":
                    # We're ignoring hydrogens. CENsible doens't use them.
                    continue
                types.append(typ)
                coords.append(
                    [
                        float(match.group(2)),
                        float(match.group(3)),
                        float(match.group(4)),
                    ]
                )
            else:
                print("WARNING! No match: " + line.strip())
                print("    This can occur if using older verisons of smina. We recommend smina 2020.12.10.")
        coords = np.array(coords)

        # For each coordinate in coords, I want to find the coordinate in
        # self.coordinates that is closest to it. Then I want to add the
        # corresponding type to the self.atoms entry for that closest match.
        for c, t in zip(coords, types):
            distances = np.linalg.norm(self.coordinates - c, axis=1)
            closest_idx = np.argmin(distances)
            self.atoms[closest_idx]["type"] = t

    def get_coordinates(self) -> np.array:
        """Returns the coordinates.
        
        Returns:
            np.array: The coordinates.
        """
        return self.coordinates

    def get_atoms(self) -> list:
        """Returns the atoms.

        Returns:
            list: The atoms.
        """
        return self.atoms

    def keep_only_near_coords(self, coords: np.array, cutoff: float = 8):
        """Keeps only those atoms that are within cutoff of any of the given
        coordinates.
        
        Args:
            coords (np.array): The coordinates.
            cutoff (float): The cutoff.
        """
        # Remove any entries in self.atoms that are not within cutoff of any
        # coordinate in coords. Also remove associated entries in
        # self.coordinates

        # Create an index list of atoms to keep
        to_keep = []

        for i, atom_coord in enumerate(self.coordinates):
            distances = np.linalg.norm(coords - atom_coord, axis=1)
            if np.any(distances < cutoff):
                to_keep.append(i)

        # Filter atoms and coordinates using the index list
        self.atoms = [self.atoms[i] for i in to_keep]
        self.coordinates = self.coordinates[to_keep]

    def get_all_atom_types(self) -> set:
        """Returns all atom types.

        Returns:
            set: All atom types.
        """
        return {a["type"] for a in self.atoms}

    def add_atomic_gauss(self, atom_idx: int, atom_type_pair: Tuple, val: float):
        """Adds the given Gaussian value to the given atom type pair.

        Args:
            atom_idx (int): The atom index.
            atom_type_pair (Tuple): The atom type pair.
            val (float): The Gaussian value to add.
        """
        # Standardize the atom_type_pair order
        atom_type_pair = to_censible_pair(atom_type_pair)
        if atom_type_pair not in self.atoms[atom_idx]["type_pairs"]:
            self.atoms[atom_idx]["type_pairs"][atom_type_pair] = val
        else:
            self.atoms[atom_idx]["type_pairs"][atom_type_pair] += val
