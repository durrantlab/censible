# Assuming smina is in the UNIX path

export smina_exec=`which smina`
export obabel_exec=`which obabel`

echo
echo "smina path: " $smina_exec
echo "obabel path: " $obabel_exec

#rm censible/data/test/1wdn_receptor.pdb.converted.pdb

python predict.py --ligpath censible/data/test/1wdn_ligand.pdb \
                  --recpath censible/data/test/1wdn_receptor.pdb \
                  --smina_exec_path $smina_exec \
                  --obabel_exec_path $obabel_exec \
                  --out test_out1.tsv 

read -p "Next> "

python predict.py --ligpath censible/data/test/1wdn_ligand.pdbqt \
                  --recpath censible/data/test/1wdn_receptor.pdb \
                  --smina_exec_path $smina_exec \
                  --obabel_exec_path $obabel_exec \
                  --out test_out2.tsv

# Append `--use_cpu` to the above commands to use CPU instead of CUDA GPU
