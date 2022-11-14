## SGDA

This repository includes codes, models, and test results for our paper: "_Towards 3D Universal Pulmonary Nodule 
Detection via Slice Grouped Domain Attention_".
This project is licensed for non-commerical research purpose only.

### Results and Models
Comparison of our SGDA and other multi-domain methods in terms of FROC on dataset LUNA16, tianchi, and russia. Values 
below the names of datasets are FROCs (unit: %). All the methods utilize NoduleNet as backbone: (1) shared models with
the prefix 'uni-', (2) independent models with the word 'single' in the name, (3) multi-domain methods, (4) universal
models with 'SG' in the name (Ours).

|Method              | #Adapters | #Groups | #Params | LUNA16 | tianchi | russia | Avg   | Pre-trained Model   |
|--------------------| --------- | ------- | ------- | ------ | ------- | ------ | ----- | ------------------- |
single NoduleNet     | -         | -       | 16.73Mx3| 77.71  | 68.23   | 37.19  | 61.04 | [model]() & [res]() |
uniNoduleNet         | -         | -       | 39.50M  | 79.88  | 68.60   | 33.35  | 60.61 | [model]() & [res]() |
NoduleNet+BN         | 3         | -       | 39.51M  | 79.94  | 68.12   | 36.52  | 61.52 | [model]() & [res]() |
NoduleNet+series     | 3         | -       | 40.14M  | 78.44  | 70.41   | 33.39  | 60.74 | [model]() & [res]() |
NoduleNet+parallel   | 3         | -       | 40.13M  | 78.57  | 70.14   | 35.61  | 61.44 | [model]() & [res]() |
NoduleNet+separable  | 3         | -       | 34.68M  | 66.31  | 62.26   | 32.96  | 53.84 | [model]() & [res]() |
NoduleNet+SNR        | -         | -       | 39.50M  | 69.52  | 66.57   | 36.76  | 57.61 | [model]() & [res]() |
single NoduleNet+SE  | -         | -       | 16.74Mx3| 77.78  | 68.86   | 38.06  | 61.56 | [model]() & [res]() |
uniSENoduleNet       | -         | -       | 39.51M  | 80.53  | 69.13   | 34.34  | 61.33 | [model]() & [res]() |
NoduleNet+SE         | 3         | -       | 39.54M  | 78.89  | 72.33   | 35.89  | 62.37 | [model]() & [res]() |
DANoduleNet          | 3         | -       | 39.54M  | **82.63** | 73.29| 38.50  | 64.80 | [model]() & [res]() |
single NoduleNet+SGSE| -         | 4       | 16.77Mx3| 78.30  | 70.36   | **39.01** |62.55|[model]() & [res]()|
uniSGSENoduleNet     | -         | 4       | 39.54M  | 81.12  | 71.00   | 38.42  | 63.51 | [model]() & [res]() |
NoduleNet+SGSE       | 3         | 4       | 39.62M  | 80.93  | 70.94   | 38.30  | 63.39 | [model]() & [res]() |
SGDANoduleNet        | 3         | 4       | 39.82M  | 81.91  | **77.13** |37.15 | **65.39**|[model]() & [res]() | 

Comparison of our SGDA and other multi-domain methods in terms of FROC on dataset PN9. The values are pulmonary nodule
detection sensitivities (unit: %) with each column representing the average number of false positives per CT image. All
the methods utilizes SANet as backbone: (1) baseline model with the prefix 'uni-', (2) universal models with 'SG' in 
the name (Ours).

|Method          | #Adapters | #Groups | #Params | 0.125  | 0.25   | 0.5    | 1.0    | 2.0    | 4.0    | 8.0    | Avg    | Pre-trained Model   |
|--------------- | --------- | ------- | ------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |-------------------- |
uniSANet         | -         | -       | 15.28M  | 38.08  | 45.05  | 54.46  | 64.50  | 75.33  | 83.86  | 89.96  | 64.46  | [model](https://drive.google.com/file/d/1oGCsekgLAsZZl8VN3QrqAgZqzMZr_kks/view) & [res]() |
DASANet          | 3         | -       | 15.32M  | 54.86  | 54.86  | 54.86  | 64.94  | 75.43  | 83.53  | 88.18  | 68.09  | [model]() & [res]() |
*SGDASANet w/o CA| 3         | 4       | 15.36M  | 52.06  | 52.06  | **58.63** | **66.33** | **77.05** | **85.13** | **90.12** | 68.77 |[model]() & [res]()|
*SGDASANet w/ CA | 3         | 4       | 15.45M  | **57.63** | **57.63** | 57.63 | 65.73 | 75.09 | 83.56 | 88.25| **69.36** |[model]() & [res]()|


### Requirements

The code is built with the following libraries:

- Python 3.6 or higher
- CUDA 10.0 or higher
- [PyTorch](https://pytorch.org/) 1.2 or higher
- [tqdm](https://github.com/tqdm/tqdm.git)
- [scipy](https://www.scipy.org/)

Besides, you need to install a custom module for bounding box NMS and overlap calculation.
```
cd build/box
python setup.py install
```

### Data

Pulmonary nodule datasets. 'Scans' denotes the number of CT scans. 'Nodules' denotes the number of 
labeled nodules. 'Class' denotes the class number. And 'Raw' means whether the dataset contains 
raw CT scans. 'Image Size' gives the dimensions of the CT image matrix alont the x, 
y, and z axes. 'Spacing' gives the voxel sizes (mm) along the x, y, and z axes.

|Dataset| Year | Scans | Nodules | Class | Raw | File Size | Image Size               | Spacing                           | Source Link          |
|------ | ---- | ----- | ------- | ----- | --- | --------- | ------------------------ | --------------------------------  |--------------------- |
LUNA16  | 2016 | 601   | 1186    | 2     | Yes | 25M-258M  | 512x512x95-512x512x733   | (0.86,0.86,2.50)-(0.64,0.64,0.50) | [link](https://luna16.grand-challenge.org/) & [split]() |
tianchi | 2017 | 800   | 1244    | 2     | Yes | 26M-343M  | 512x512x114-512x512x1034 | (0.66,0.66,2.50)-(0.69,0.69,0.30) | [link](https://tianchi.aliyun.com/competition/entrance/231601/introduction) & [split]() |
russia  | 2018 | 364   | 1850    | 2     | Yes | 80M-491M  | 512x512x313-512x512x1636 | (0.62,0.62,0.80)-(0.78,0.78,0.40) | [link](https://mosmed.ai/en/data-sets/ct_lungcancer_500/) & [split]()|
PN9     | 2021 | 8796  | 40436   | 9     | No  | 5.6M-73M  | 212x212x181-455x455x744  | (1.00,1.00,1.00)-(1.00,1.00,1.00) | [link](https://jiemei.xyz/publications/SANet) & [split](https://jiemei.xyz/publications/SANet)|

Download the datasets and add the information to `configs/*config*.py`.
Please refer to [`specificFiles/LIDC/lung_seg.py`](./specificFiles/LIDC/lung_seg.py) and [`specificFiles/LIDC/preprocess.py`](./specificFiles/LIDC/preprocess.py) for the data preprocessing.

### Testing

Run the following scripts to evaluate the model and obtain the results of FROC analysis.
```
python universal_test_sanet.py --ckpt='./results/model/model.ckpt' --save_dir='./results/'
```

### Training
This implementation supports multi-gpu, `data_parallel` training.

Change training configuration and data configuration in `configs/*config*.py`, especially the path to preprocessed data.

Run the training script:
```
python SGDA_train_sanet_middle_top.py
```

### Citations

If you are using the code/model/data provided here in a publication, please consider citing:

    @article{SGDA22,  
    author={Rui Xu and Zhi Liu and Yong Luo and Han Hu and Li Shen and Bo Du and Kaiming Kuang and and Jiancheng Yang},   
    title={Towards 3D Universal Pulmonary Nodule Detection via Slice Grouped Domain Attention},   
    journal={},    
    year={},  
    volume={},  
    number={},  
    pages={},  
    doi={}} 

### Contact

For any questions, please contact: rui.xu AT whu.edu.cn.

### Acknowledgment

This code is based on the [UOD](https://github.com/frank-xwang/towards-universal-object-detection), [SANet](https://github.com/mj129/SANet) and [NoduleNet](https://github.com/uci-cbcl/NoduleNet).
