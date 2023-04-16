#!/usr/bin/env python3

'''take a regular types file and create one with smina terms in it'''

import sys, subprocess, re
from concurrent.futures import ProcessPoolExecutor
from concurrent import futures
from tqdm import tqdm  # Import tqdm

def process_line(line):
    (label, affinity, rec, lig) = line.split()
    r = rec.replace('_protein.gninatypes', '_protein.pdb')
    l = lig.replace('_ligand.gninatypes', '_ligand.sdf')

    r = r.replace('_protein_centered.gninatypes', '_protein_centered.pdb')
    l = l.replace('_ligand_centered.gninatypes', '_ligand_centered.sdf')

    try:
        # smina_exec = "/mnt/data_2/DataC/miniconda3/envs/py39/bin/smina"
        # smina_exec = "/ihome/jdurrant/durrantj/bindingmoad2020_for_gnina/smina/smina.static"
        smina_exec = "/mnt/Data/jdurrant/cenet/prepare_data/smina/smina.static"
        out = subprocess.check_output('%s --custom_scoring allterms.txt --score_only -r ./%s -l ./%s' % (smina_exec, r, l), shell=True)
        out = out.decode()
        m = re.findall('## [^N]\S+ (.*)', out)

        return affinity + " " + m[0] + " " + rec + " " + lig + "\n"
    except Exception as e:
        print(f"Error processing line: {line}. Error: {e}")
        return None

def make_all_types(out_flnm):
    with open(out_flnm) as all_types_file:
        lines = all_types_file.readlines()

    with ProcessPoolExecutor(max_workers=12) as executor:
        # results = executor.map(process_line, lines)

        # Wrap the results iterator with tqdm to show the progress bar
        results = list(tqdm(executor.map(process_line, lines), total=len(lines)))

    # Filter out None values from the results list
    results = [result for result in results if result is not None]

    with open(out_flnm.replace(".types", "_cen.types"), "w") as f:
        for result in results:
            f.write(result)

def main():
    make_all_types("all.types")
    make_all_types("all_centered.types")

if __name__ == '__main__':
    main()
