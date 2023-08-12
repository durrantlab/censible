# Save the order of the smina terms

smina/smina.static --custom_scoring allterms.txt --score_only -r ./pdbbind/refined-set/2uz9/2uz9_protein.pdb -l ./pdbbind/refined-set/2uz9/2uz9_ligand.mol2 | grep "## Name" | sed "s/## Name //g" | tr ' ' '\n' > smina_ordered_terms.txt

echo Created file smina_ordered_terms.txt
