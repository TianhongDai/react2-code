# REACT2 Segmentation
This code is for the project REACT2 Segmentation.

## Requirements
- torch==1.6.0
```bash
# install torch
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
# install other requirements
pip install -r requirements.txt
```
## Instruction
1. prepare the dataset (generate the corresponding masks):
```bash
python prepare_dataset.py
```
2. train the segmentation model:
```bash
python train.py --arch='unet++' --encoder='resnet50'
```
3. evaluate and draw the boxes on samples:
```bash
python eval.py --arch='unet++' --ckpt-path='...'
```
4. evaluate the metrics on samples:
```bash
python eval_metric.py --arch='unet++' --ckpt-path='...'
```
