# ACFormer
Code for Affine-Consistent Transformer for Multi-Class Cell Nuclei Detection (ICCV 2023)

## Requisities
-`python=3.8`

-`pytorch=1.12.1+cu102`


## Installation
Install mmcv using mim
```
pip install -U openmim
mim install mmcv-full==1.6.1
```
Git clone acformer
```
git clone https://github.com/LL3RD/ACFormer.git
```
Install
```
cd ACFormer
cd thirdparty/mmdetection 
python -m pip install -e .
cd ../.. 
python -m pip install -e .
```

## Main Result
### Lizard Dataset
| Method | F1d | F1c | Model Weights |Config Files|
| ---- | -----| ----- |----|----|
| ACFormer | 0.782 | 0.557 | [Checkpoint](https://drive.google.com/file/d/12FyfAQf5VU2poXvqE_FmrB2HL6VDCldj/view?usp=drive_link)|[Config](https://drive.google.com/file/d/14scJog5GjZc-n-Uwn4sIAJcaO2tokxA-/view?usp=drive_link)|

## Evaluation
Modify your dataset path and checkpoint path in tools/inference_lizard.py and run
```
python tools/inference_lizard.py
```

## Acknowledgement
- ACFormer is built based on [SoftTeacher](https://github.com/microsoft/SoftTeacher) and [MMDetection](https://github.com/open-mmlab/mmdetection).
