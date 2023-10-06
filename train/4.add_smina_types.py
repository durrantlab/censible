#!/usr/bin/env python3

"""Take a regular types file and create one with smina terms in it."""

import subprocess, re
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # Import tqdm


def process_line(line):
    (label, affinity, rec, lig) = line.split()
    r = rec.replace("_protein.pdb.nowat.ph7.gninatypes", "_protein.pdb.nowat.ph7.pdb")
    l = lig.replace("_ligand.mol2.ph7.gninatypes", "_ligand.mol2.ph7.mol2")

    try:
        smina_exec = "smina/smina.static"
        out = subprocess.check_output(
            f"{smina_exec} --custom_scoring allterms.txt --score_only -r ./{r} -l ./{l} > /dev/null",
            shell=True,
        )
        out = out.decode()
        m = re.findall("## [^N]\S+ (.*)", out)

        return f"{affinity} {m[0]} {rec} {lig}" + "\n"
    except Exception as e:
        print(f"Error processing line: {line}. Error: {e}")
        return None


def make_all_types(out_flnm):
    with open(out_flnm) as all_types_file:
        lines = all_types_file.readlines()

    with ProcessPoolExecutor(max_workers=12) as executor:
        # Wrap the results iterator with tqdm to show the progress bar
        results = list(tqdm(executor.map(process_line, lines), total=len(lines)))

    # Filter out None values from the results list
    results = [result for result in results if result is not None]

    new_out_filename = out_flnm.replace(".types", "_cen.types")
    with open(new_out_filename, "w") as f:
        for result in results:
            f.write(result)
    print(f"Created file {new_out_filename}")


def main():
    make_all_types("all.types")


if __name__ == "__main__":
    main()
