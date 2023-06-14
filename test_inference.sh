lig=${1}
splitnum="3"

# Split the ligand to the cache directory.
#lig="../cenet_vs/4.smina_docking/smina_out/batch_6/CID-9591264-inactives.smina_out.pdbqt.gz"
bsnm=`basename ${lig} .gz`

rm -f ${lig}.cenet.out

cat ${lig} | gunzip > ./scratch/${bsnm}

/home/jdurrant/vina_split --input ./scratch/${bsnm} --ligand ./scratch/${bsnm}.split

# CID-3197330-inactives.smina_out.pdbqt

# Iterate through all imgs/all-fold* directories and run inference on the test set
for model_dir in imgs/all-fold*; do
    # Get all the splits
    python apply_model.py --ligpath scratch/${bsnm}.split${splitnum}.pdbqt --recpath 5.4EG4_no_extra_ligs.no_wats.not_lig.CAS_to_CYS.pdb --model_dir $model_dir --smina_exec_path /mnt/Data/jdurrant/cenet/prepare_data/smina/smina.static >> scratch/${bsnm}.split${splitnum}.cenet.out

    # for lig in scratch/${bsnm}*.split*.pdbqt; do
    #     echo $model_dir $lig
    #     echo "python apply_model.py --ligpath ${lig} --recpath 5.4EG4_no_extra_ligs.no_wats.not_lig.CAS_to_CYS.pdb --model_dir $model_dir --smina_exec_path /mnt/Data/jdurrant/cenet/prepare_data/smina/smina.static >> ${lig}.cenet.out"
    #     --out_prefix testtt 
    # done
done

rm scratch/${bsnm}*pdbqt
rm scratch/${bsnm}
