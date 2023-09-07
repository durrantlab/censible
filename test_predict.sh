# Assuming smina is in the UNIX path

export smina_exec=`which smina`

echo
echo "smina path: " $smina_exec

python predict.py --ligpath censible/data/test/1wdn_ligand.sdf \
                  --recpath censible/data/test/1wdn_receptor.pdb \
                  --smina_exec_path $smina_exec \
                  --out test_out.tsv 

# Append `--use_cpu` to the above command to use CPU instead of CUDA GPU
