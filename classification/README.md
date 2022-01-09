# REACT2 Classification
This code is for the project REACT2 Classification.

## Requirements
- torch==1.6.0
```bash
# install torch
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```
## Instruction
1. train the segmentation model:
```bash
python train.py --cuda
```
2. evaluate the metrics on samples:
```bash
python eval_metric.py --cuda
```
