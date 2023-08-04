# original ligand pdbqt file, gzipped (.pdbqt.gz). Not processed using vina_split
# (this script also does that)
lig=${1}

# The pose ("1", "2", or "3")
posenum=${2}

mkdir -p scratch

#recep="/mnt/Data/jdurrant/cenet_vs/hxk4/3.receptor/receptor.pdb"
recep="/mnt/Data/jdurrant/cenet_vs/hiv_integrase_dude.remember_variants_still_bad.example_good_smina_bad_cenet/3.receptor/receptor.pdb"


# Split the ligand to the cache directory.
#lig="../cenet_vs/4.smina_docking/smina_out/batch_6/CID-9591264-inactives.smina_out.pdbqt.gz"
bsnm=`basename ${lig} .gz`

rm -f ${lig}.cenet.out

cat ${lig} | gunzip > ./scratch/${bsnm}

/home/jdurrant/vina_split --input ./scratch/${bsnm} --ligand ./scratch/${bsnm}.split

# CID-3197330-inactives.smina_out.pdbqt

# Iterate through all imgs/all-fold* directories and run inference on the test set
for model_dir in /mnt/Data/jdurrant/cenet/imgs/all-fold*; do
    # Get all the splits
    python /mnt/Data/jdurrant/cenet/apply_model.py --ligpath scratch/${bsnm}.split${posenum}.pdbqt --recpath ${recep} --model_dir $model_dir --smina_exec_path /mnt/Data/jdurrant/cenet/prepare_data/smina/smina.static --out scratch/${bsnm}.split${posenum}.cenet.inf >> scratch/${bsnm}.split${posenum}.cenet.out

    # for lig in scratch/${bsnm}*.split*.pdbqt; do
    #     echo $model_dir $lig
    #     echo "python apply_model.py --ligpath ${lig} --recpath 5.4EG4_no_extra_ligs.no_wats.not_lig.CAS_to_CYS.pdb --model_dir $model_dir --smina_exec_path /mnt/Data/jdurrant/cenet/prepare_data/smina/smina.static >> ${lig}.cenet.out"
    #     --out_prefix testtt 
    # done
done

rm scratch/${bsnm}*pdbqt
rm scratch/${bsnm}

gzip scratch/${bsnm}.split${posenum}.cenet.inf
gzip scratch/${bsnm}.split${posenum}.cenet.out

