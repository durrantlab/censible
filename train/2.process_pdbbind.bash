export OBABEL=/usr/bin/obabel

# Prepare the receptors
find pdbbind -name "*_protein.pdb" | awk '{print "echo " $1 "; grep -v HOH " $1 " > " $1 ".nowat.pdb; $OBABEL -ipdb " $1 ".nowat.pdb -opdb -d > " $1 ".nowat.noh.pdb; $OBABEL -ipdb " $1 ".nowat.noh.pdb -opdb -p 7 > " $1 ".nowat.ph7.pdb; rm " $1 ".nowat.pdb " $1 ".nowat.noh.pdb"}' | parallel

# Prepare the ligands
# find pdbbind/ -name "*_ligand.mol2" | awk '{print "$OBABEL -omol2 " $1 " -omol2 -p 7 > " $1 ".ph7.mol2"}' | parallel
find pdbbind/ -name "*_ligand.mol2" | awk '{print "$OBABEL -imol2 " $1 " -omol2 -p 7 > " $1 ".ph7.mol2"}' | parallel
