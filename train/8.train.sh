# Train each fold
python train.py --fold_num 0
python train.py --fold_num 1
python train.py --fold_num 2

# Train on all data
cp all_cen.types allcentrain0_cen.types
cp all_cen.types allcentest0_cen.types
python train.py --fold_num 0 --prefix allcen
rm allcentrain0_cen.types allcentest0_cen.types
