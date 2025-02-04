source activate pytorch # Activate PyTorch environment
DUMP_ROOT="/home/ubuntu"
pip install gdown
pip install onnxruntime==1.20.1 deepsparse==1.8.0 openvino==2024.6.0 tqdm loguru scikit-learn

mkdir $DUMP_ROOT/models
gdown --folder https://drive.google.com/drive/folders/1DvwI_QBd069R7GDCG9tu_j71ZD3uqV0E?usp=drive_link -O $DUMP_ROOT/models
