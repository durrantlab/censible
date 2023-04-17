module load singularity/3.8.3
module load parallel

# singularity run /ihome/crc/install/gnina/gnina.sif

# find 1.pdbbind2020/ -name "*protein.pdb" | awk '{print "gninatyper " $1 " " $1 ".gninatypes"}' | sed "s/pdb.gninatypes/gninatypes/g" > t
# find 1.pdbbind2020/ -name "*_ligand.sdf" | awk '{print "gninatyper " $1 " " $1 ".gninatypes"}' | sed "s/sdf.gninatypes/gninatypes/g" >> t

find 1.pdbbind2020/ -name "*protein_centered.pdb" | awk '{print "gninatyper " $1 " " $1 ".gninatypes"}' | sed "s/pdb.gninatypes/gninatypes/g" >> t
find 1.pdbbind2020/ -name "*_ligand_centered.sdf" | awk '{print "gninatyper " $1 " " $1 ".gninatypes"}' | sed "s/sdf.gninatypes/gninatypes/g" >> t

# . t | parallel
