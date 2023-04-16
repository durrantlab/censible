module load parallel

# find 1.pdbbind2020/ -name "*protein.pdb" | awk '{print "gninatyper " $1 " " $1 ".gninatypes"}' | sed "s/pdb.gninatypes/gninatypes/g" > t
# find 1.pdbbind2020/ -name "*_ligand.sdf" | awk '{print "gninatyper " $1 " " $1 ".gninatypes"}' | sed "s/sdf.gninatypes/gninatypes/g" >> t

find 1.pdbbind2020/ -name "*protein.pdb" | awk '{print "gninatyper " $1 " " $1 ".gninatypes"}' | sed "s/pdb.gninatypes/gninatypes/g" > t
find 1.pdbbind2020/ -name "*_ligand.sdf" | awk '{print "gninatyper " $1 " " $1 ".gninatypes"}' | sed "s/sdf.gninatypes/gninatypes/g" >> t

. t | parallel
