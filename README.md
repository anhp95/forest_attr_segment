# forest_attr_segment

**Update on 2022/10/02. Provide reclassification step to enhance the resolution of result image**

### TODO

- [x] Support different 2D/3D UNET based architecture with Atrous Convolution Blocks for forest attribute (species, age) segmentation
- [x] Support training/validation dataset from Sentinel 1/2 in GIFU prefecture - Japan.

### Introduction

This is the Pytorch (1.9.1) implementation of Deep Learning model in "FOREST-RELATED SDG ISSUES MONITORING FOR DATA-SCARE REGIONS EMPLOYING MACHINE LEARNING AND REMOTE SENSING - A CASE STUDY FOR ENA CITY, JAPAN".

It can use with 2D/3D UNET-based CNN with Atrous Convolution Blocks

### Installation

The source code is test with Anaconda and Python 3.9.7.

1. Clone the repo:

```Shell
    git clone https://github.com/anhp95/forest_attr_segment.git
    cd forest_attr_segment
```

2. Create a conda environment from as follows:

```Shell
    conda env create -f environment.yml
```

### Training

Follow these steps to train the model with our dataset

1. Download the dataset via [Google Drive]()

2. Configure the dataset path in [mypath.py]()

3. Activate your Anaconda environment

4. Input arguments: (see the full set of input arguments via python train.py --help)

   ```Shell
   usage: train_nn.py [-h] [--forest_attr {spec,age}]
                  [--backbone {2d_p2,2d_p1p2,2d_p1p2p3,3d_org,3d_adj,3d_adj_dec_acb,3d_adj_emd_acb,3d_org_emd_acb}]
                  [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--load_model LOAD_MODEL]
                  [--logs_file LOGS_FILE] [--pin_memory] [--no_workers NO_WORKERS]

   PyTorch Tree Species/Age Segmentation Training

   optional arguments:
   -h, --help            show this help message and exit
   --forest_attr {spec,age}
                           which forest attribute is going to be segmented (default: spec)
   --backbone {2d_p2,2d_p1p2,2d_p1p2p3,3d_org,3d_adj,3d_adj_dec_acb,3d_adj_emd_acb,3d_org_emd_acb}
                           backbone of the model (default: 3d_adj_emd_acb)
   --num_epochs NUM_EPOCHS
                           Number of epochs (default: 100)
   --batch_size BATCH_SIZE
                           batch size (default: 16)
   --lr LR               learning rate (default: 1e-5)
   --load_model LOAD_MODEL
                           path to the checkpoint file (default: None)
   --logs_file LOGS_FILE
                           put the path to the logs directory (default: logs/)
   --pin_memory          whether use nesterov (default: False)
   --no_workers NO_WORKERS
                           The number of wokers for dataloader (default: 2)
   ```

### Inference

Follow these step to use the trained model to infer your data

1. Low-resolution inference
2. High-resolution inference

### Acknowledgement
