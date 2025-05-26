source activate base

conda create -n own_ICL python=3.10 -y

conda activate own_ICL

pip install -r requirements.txt

pip install sentencepiece


pip install protobuf

pip install flash-attn

pip install tiktoken

pip install pytest