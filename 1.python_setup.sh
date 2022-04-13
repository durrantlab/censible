conda create --name cen python=3.9
conda activate cen
mkdir -p data
cd data
wget http://bits.csb.pitt.edu/files/cen.tgz
tar xvfz cen.tgz
cd cen
wc -l all_cen.types
head -1 all_cen.types
cat allterms.txt

conda install -y pip cudatoolkit
python3 -m pip install --upgrade pip

cd -
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu111/torch_stable.html
