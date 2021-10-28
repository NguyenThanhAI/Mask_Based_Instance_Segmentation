# Intall

```
conda env create --file mmdet_env.yml
```


or

```
conda create --name pytorch_mmlab python=3.6.8
conda activate pytorch_mmlab
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
pip install mmdet
pip install instaboostfast
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/lvis-dataset/lvis-api.git
pip install albumentations>=0.3.2 --no-binary imgaug,albumentations
```


# Run


```
python maskrcnn.py --image_path= --config_file= --model_path=
```