# Download https://raw.githubusercontent.com/gnina/scripts/master/create_caches2.py

curl -O https://raw.githubusercontent.com/gnina/scripts/master/create_caches2.py

python3 create_caches2.py -c 350 --recmolcache rec.molcache2 --ligmolcache lig.molcache2 -d ./ all_cen.types
python3 create_caches2.py -c 350 --recmolcache rec_centered.molcache2 --ligmolcache lig_centered.molcache2 -d ./ all_centered_cen.types
