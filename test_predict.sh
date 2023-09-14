# Assuming smina is in the UNIX path

export smina_exec=`which smina`

echo
echo "smina path: " $smina_exec

#rm censible/data/test/1wdn_receptor.pdb.converted.pdb

python predict.py --ligpath censible/data/test/1wdn_ligand.pdb \
                  --recpath censible/data/test/1wdn_receptor.pdb \
                  --smina_exec_path $smina_exec \
                  --out test_out1.tsv 

read -p "Next> "

#rm censible/data/test/1wdn_receptor.pdb.converted.pdb

python predict.py --ligpath censible/data/test/1wdn_ligand.pdbqt \
                  --recpath censible/data/test/1wdn_receptor.pdb \
                  --smina_exec_path $smina_exec \
                  --out test_out2.tsv

# Append `--use_cpu` to the above command to use CPU instead of CUDA GPU
