import glob
import os
import concurrent.futures
from tqdm import tqdm

# Import prody
import prody
from rdkit import Chem
import numpy as np
from concurrent.futures import as_completed

# prody.confProDy(verbosity='warning')

def process_directory(d):
    pdbid = os.path.basename(d)

    lig_name = f"{d}/{pdbid}_ligand.sdf"
    pdb_name = f"{d}/{pdbid}_protein.pdb"

    # Load ligname, which is an sdf file
    lig = Chem.SDMolSupplier(lig_name)[0]

    if lig is None:
        # Try getting it from the mol2 file instead
        lig = Chem.MolFromMol2File(lig_name[:-3] + "mol2")
        if lig is None:
            print(f"Could not load {lig_name}")
            return

    # Get the center of geometry of the ligand
    lig_coords = lig.GetConformer().GetPositions()
    lig_center = np.mean(lig_coords, axis=0)

    # Move the ligand so that its center of geometry is at the origin
    lig_coords -= lig_center

    # Update atom positions in the molecule object
    conf = lig.GetConformer()
    for atom_idx, new_coords in enumerate(lig_coords):
        conf.SetAtomPosition(atom_idx, new_coords)

    # Write the ligand to a new sdf file
    Chem.SDWriter(f"{d}/{pdbid}_ligand_centered.sdf").write(lig)

    # Load the protein, which is a pdb file
    pdb = prody.parsePDB(pdb_name)

    # Move those coordinates so that the ligand center of geometry is at the
    # origin
    pdb_coords = pdb.getCoords()
    pdb_coords -= lig_center
    pdb.setCoords(pdb_coords)

    # Write the protein to a new pdb file
    prody.writePDB(f"{d}/{pdbid}_protein_centered.pdb", pdb)

dirs = glob.glob("1.pdbbind2020/*/????")
# dirs = glob.glob("1.pdbbind2020/*/1lag")


# Set the number of workers to the number of available CPU cores
num_workers = 12 # os.cpu_count()

# Use a ThreadPoolExecutor to parallelize the processing of directories
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    # Create a dictionary to store the Future objects
    future_to_dir = {executor.submit(process_directory, d): d for d in dirs}
    # Initialize the progress bar
    progress_bar = tqdm(total=len(dirs), desc="Processing directories")
    # Iterate over the completed tasks and update the progress bar
    for future in as_completed(future_to_dir):
        d = future_to_dir[future]
        progress_bar.update(1)
    # Close the progress bar
    progress_bar.close()