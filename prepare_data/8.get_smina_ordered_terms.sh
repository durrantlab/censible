/mnt/Data/jdurrant/cenet/prepare_data/smina/smina.static --custom_scoring allterms.txt --score_only -r ./1.pdbbind2020/refined-set/2uz9/2uz9_protein.pdb -l ./1.pdbbind2020/refined-set/2uz9/2uz9_ligand.mol2 | grep "## Name" | sed "s/## Name //g" | tr ' ' '\n' > smina_ordered_terms.txt

echo Created file smina_ordered_terms.txt
