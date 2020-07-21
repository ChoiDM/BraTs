# Brain Tumor Segmentation

## Getting Started
This repository provides everything necessary to train and evaluate a brain tumor segmentation model.

Requirements:
- Python 3 (code has been tested on Python 3.5.6)
- PyTorch (code tested with 1.1.0)
- CUDA and cuDNN (tested with Cuda 10.0)
- Python pacakges : opencv-python (tested with 4.1), tqdm, SimpleITK, scipy (tested with 1.2.1)

Stucture:
- ```data/```: save directory of datasets
- ```datasets/```: data loading code
- ```network/```: network architecture definitions
- ```options/```: argument parser options
- ```utils/```: image processing code, miscellaneous helper functions, and training/evaluation core code
- ```train.py/```: code for model training
- ```test.py/```: code for model evaluation
- ```preprocess_mask.py/```: code for pre-processing masks (refinement of ce mask and peri-tumoral mask generation)

#### Dataset
Out private dataset which has four types of MRI images (FLAIR, T1GD, T1, T2) and three types of mask (necro, ce, T2).\
The dataset architecture must be as below.
```
data
└───train
│   └───patientDir001
│   │   │   FLAIR_stripped.nii.gz
│   │   │   T1GD_stripped.nii.gz
│   │   │   T1_stripped.nii.gz
│   │   │   T2_stripped.nii.gz
│   │   │   necro_mask.nii.gz
│   │   │   ce_mask.nii.gz
│   │   │   t2_mask.nii.gz
│   │   │
│   └───patientDir002
│   └───patientDir003
│   └───...
│
└───valid
    └───patientDir00a
    └───patientDir00b
    └───...
```

#### Training and Testing
- Before training, call ```python preprocess_mask.py``` to pre-process masks.

- To train a 3D network, call:
```python train.py --batch_size 1 --in_dim 3 --in_depth 128 --in_res 140```

- Before 2D training, call ```python parsing_2D.py``` to parse 2D datasets\
To train a 2D network, call: ```python train.py --batch_size 1 --in_dim 2 --in_res 140```

Once a model has been trained, you can evaluate it with:

```python test.py --in_dim 2 --resume trained_weights.pth```

#### Pretrained Models
Not released yet.