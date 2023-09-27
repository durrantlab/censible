# Assuming smina and obabel are in the UNIX path

export smina_exec=`which smina`
export obabel_exec=`which obabel`

echo
echo "smina path: " $smina_exec
echo "obabel path: " $obabel_exec

# Remove the converted receptor file if it exists (to regenerate it).
rm -f censible/data/test/1wdn_receptor.pdb.converted.pdb

# Iterate through these files: censible/data/test/1wdn_ligand.*
for f in censible/data/test/1wdn_ligand.*
do
    python predict.py --ligpath $f \
                      --recpath censible/data/test/1wdn_receptor.pdb \
                      --smina_exec_path $smina_exec \
                      --obabel_exec_path $obabel_exec

    # Clean up
    rm -f censible/data/test/1wdn_ligand.*converted.*

    # read -p "Next> "
done

# Also test multiple ligands
python predict.py --ligpath censible/data/test/1wdn_ligand.pdbqt censible/data/test/1wdn_ligand.pdb \
                  --recpath censible/data/test/1wdn_receptor.pdb \
                  --smina_exec_path $smina_exec \
                  --obabel_exec_path $obabel_exec

# Clean up
rm -f censible/data/test/1wdn_ligand.*converted.*

# Test --use_cpu flag to avoid cuda. Also test --out flag.
python predict.py --ligpath censible/data/test/1wdn_ligand.pdb \
                  --recpath censible/data/test/1wdn_receptor.pdb \
                  --smina_exec_path $smina_exec \
                  --obabel_exec_path $obabel_exec \
                  --out test_out.tsv \
                  --use_cpu

# Clean up
rm -f censible/data/test/1wdn_ligand.*converted.*

# Append `--use_cpu` to the above commands to use CPU instead of CUDA GPU
