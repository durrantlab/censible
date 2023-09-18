# You must install gninatyper and run it on the proteins and ligands. Here's how
# I do it:

module load singularity/3.8.3
module load parallel

find pdbbind/ -name "*protein.pdb.nowat.ph7.pdb" | awk '{print "[ ! -f " $1 ".gninatypes ] && gninatyper " $1 " " $1 ".gninatypes"}' | sed "s/pdb.gninatypes/gninatypes/g" > t
find pdbbind/ -name "*ligand.mol2.ph7.mol2" | awk '{print "[ ! -f " $1 ".gninatypes ] && gninatyper " $1 " " $1 ".gninatypes"}' | sed "s/mol2.gninatypes/gninatypes/g" >> t

# . t | parallel

