# You must install gninatyper and run it on the proteins and ligands. Here's how
# I do it:

module load singularity/3.8.3
module load parallel

find pdbbind/ -name "*protein.pdb" | awk '{print "gninatyper " $1 " " $1 ".gninatypes"}' | sed "s/pdb.gninatypes/gninatypes/g" > t
find pdbbind/ -name "*_ligand.sdf" | awk '{print "gninatyper " $1 " " $1 ".gninatypes"}' | sed "s/sdf.gninatypes/gninatypes/g" >> t

. t | parallel
