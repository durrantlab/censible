# Assuming smina is in the UNIX path

export smina_exec=`which smina`

python predict.py --ligpath censible/data/test/1wdn_ligand.mol2 \
                  --recpath censible/data/test/1wdn_receptor.pdb \
                  --smina_exec_path $smina_exec \
                  --out test_out.tsv 

# Append `--use_cpu` to the above command to use CPU instead of CUDA GPU
