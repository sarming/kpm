wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

yes

yes
bash
conda config --set auto_activate_base false
conda config --add channels intel
conda update -n base -c defaults conda
y
conda create --name hidalgo intelpython3_core python=3
y
conda activate hidalgo
conda install matplotlib tornado graphviz networkx
pip install profilehooks
