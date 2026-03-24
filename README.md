使用方式
git clone https://github.com/yyxsxxx/LCAFNet.git

cd LCAFNet

conda create -n LCAFNet python=3.8

conda activate LCAFNet

pip install pandas

pip install lanelet2

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install torch_geometric==2.3.1

conda install pytorch-lightning==2.0.3

下载interaction和argoverse的数据集

# INTERACTION Training
python LCAFNet-INTERACTION/train.py --root /path/to/INTERACTION_root/ --train_batch_size 10 --val_batch_size 2 --devices 4
# INTERACTION Validation
python LCAFNet-INTERACTION/val.py --root /path/to/INTERACTION_root/ --val_batch_size 10 --devices 4 --ckpt_path /path/to/checkpoint.ckpt
# INTERACTION Testing
python LCAFNet-INTERACTION/test.py --root /path/to/INTERACTION_root/ --test_batch_size 10 --devices 1 --ckpt_path /path/to/checkpoint.ckpt

# Argoverse Training
python LCAFNet-Argoverse/train.py --root /path/to/Argoverse_root/ --train_batch_size 2 --val_batch_size 2 --devices 4
# Argoverse Validation
python LCAFNet-Argoverse/val.py --root /path/to/Argoverse_root/ --val_batch_size 2 --devices 4 --ckpt_path /path/to/checkpoint.ckpt
# Argoverse Testing
python LCAFNet-Argoverse/test.py --root /path/to/Argoverse_root/ --test_batch_size 2 --devices 1 --ckpt_path /path/to/checkpoint.ckpt
