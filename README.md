quick start

Dataset preparation
1.download dataset.zip from https://pan.baidu.com/share/init?surl=FMzmEYwuGLNl5EESUGzmvg
2.nerf/Plenoxel: unzip dataset to ./nerf/data/nerf_llff_data
3.3DGS:unzip dataset to ./gaus

Training
Nerf
python run_nerf.py --config configs/subway.txt

Plenoxel
python run_plenoxel.py --config configs/subway-Copy1.txt

3DGS
python train.py -s ./subway --eval
