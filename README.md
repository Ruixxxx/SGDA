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
single NoduleNet     | -         | -       | 16.73Mx3| 77.71  | 68.23   | 37.19  | 61.04 | [model_luna16](https://pan.baidu.com/s/1JrN7Py0mLlQ81jMWOuCdvg?pwd=46wr) [model_tianchi](https://pan.baidu.com/s/17yztFSpgd2HGjMIYu_8PiQ?pwd=s3wx) [model_russia](https://pan.baidu.com/s/1wnG39_Ueo9LlO6VkHpnO4Q?pwd=fsae) & [res_luna16](https://pan.baidu.com/s/1t65-4iJWOzwAsq7D9vlmdA?pwd=pb3r) [res_tianchi](https://pan.baidu.com/s/1QWYFruSIJmUC4hNS6A1Fhg?pwd=j2yo) [res_russia](https://pan.baidu.com/s/1eDP1Kn9Zo2JAaF3QEy88uQ?pwd=qgn7) |
uniNoduleNet         | -         | -       | 39.50M  | 79.88  | 68.60   | 33.35  | 60.61 | [model](https://pan.baidu.com/s/1ThIx4Q6d6mnf8BIU4u42gA?pwd=cpvu) & [res_luna16](https://pan.baidu.com/s/1LvBwTprJdkHoHya0Fm6TGQ?pwd=5vyb) [res_tianchi](https://pan.baidu.com/s/1dWsMiGUVd8qlNZroIxTEqA?pwd=zig0) [res_russia](https://pan.baidu.com/s/1CffSTugCAjn59z9nzDEyUA?pwd=nmu7) |
NoduleNet+BN         | 3         | -       | 39.51M  | 79.94  | 68.12   | 36.52  | 61.52 | [model](https://pan.baidu.com/s/1gmdBpaeoNL4u5w2TUu3TXA?pwd=kaya) & [res_luna16](https://pan.baidu.com/s/1iuq34mw-2LZbVFlKCZ5FeQ?pwd=pheq) [res_tianchi](https://pan.baidu.com/s/1sIywnTr_WMUPivPjWx_TTQ?pwd=55go) [res_russia](https://pan.baidu.com/s/1vEJq7qqNf1Jzig2CPa5g9A?pwd=t5ug) |
NoduleNet+series     | 3         | -       | 40.14M  | 78.44  | 70.41   | 33.39  | 60.74 | [model](https://pan.baidu.com/s/1rdm5TdjIZjcUzxGIi6ovEw?pwd=xmjo) & [res_luna16](https://pan.baidu.com/s/14CiJBtAW__HU6ytyOuuj_Q?pwd=1e3t) [res_tianchi](https://pan.baidu.com/s/1k3QjALu_vEQL2qleqRUPrA?pwd=r4c5) [res_russia](https://pan.baidu.com/s/1q8kal5JkrZoCnLCjyfa9Xw?pwd=gm6c) |
NoduleNet+parallel   | 3         | -       | 40.13M  | 78.57  | 70.14   | 35.61  | 61.44 | [model](https://pan.baidu.com/s/1zVVqXh5qV9BS7niAE1eFlQ?pwd=4hkx) & [res_luna16](https://pan.baidu.com/s/1nRafQjFBThBvB7ovfw1sLQ?pwd=czsd) [res_tianchi](https://pan.baidu.com/s/1BKQd3x71_S0CCllD4hbEZQ?pwd=ocki) [res_russia](https://pan.baidu.com/s/1SG8dDmdemusXOxDs_tlN5Q?pwd=pv4e) |
NoduleNet+separable  | 3         | -       | 34.68M  | 66.31  | 62.26   | 32.96  | 53.84 | [model](https://pan.baidu.com/s/1QpRaOABNgC1sp3Ne9doXKg?pwd=orca) & [res_luna16](https://pan.baidu.com/s/1doLOIVh4pcIhhe2JXdO0Yw?pwd=bmjb) [res_tianchi](https://pan.baidu.com/s/1gQZjj3u4KTi4ZKR11J0GwA?pwd=7gds) [res_russia](https://pan.baidu.com/s/1e0TJihEeSzlkP6StcIf8JQ?pwd=ky9x) |
NoduleNet+SNR        | -         | -       | 39.50M  | 69.52  | 66.57   | 36.76  | 57.61 | [model](https://pan.baidu.com/s/15R5arMjPA3I-mDjnlh0FKw?pwd=gq2j) & [res_luna16](https://pan.baidu.com/s/1ChhVpL7HYZiGvlb0L3tQIg?pwd=cl0m) [res_tianchi](https://pan.baidu.com/s/1JcaYNW6LdWdjMvj1YydS2g?pwd=p6mq) [res_russia](https://pan.baidu.com/s/1g43-MUHQMISgG3rng4BhCw?pwd=zjrg) |
single NoduleNet+SE  | -         | -       | 16.74Mx3| 77.78  | 68.86   | 38.06  | 61.56 | [model_luna16](https://pan.baidu.com/s/1uIqB5poEMLb6QebjJ2EHGg?pwd=umk5) [model_tianchi](https://pan.baidu.com/s/1M5cfjOZaRqI1Kf43V20ACw?pwd=cg9h) [model_russia](https://pan.baidu.com/s/171tnCkKE518nNWdD8rTEcw?pwd=pu5f) & [res_luna16](https://pan.baidu.com/s/149EOGB5raZk05En3gagDkA?pwd=ng01) [res_tianchi](https://pan.baidu.com/s/1XPSIQALrFXGtpyaWCiS87Q?pwd=kr8t) [res_russia](https://pan.baidu.com/s/1YwGjhuxIk1j-tunfvApAnw?pwd=p75o) |
uniSENoduleNet       | -         | -       | 39.51M  | 80.53  | 69.13   | 34.34  | 61.33 | [model](https://pan.baidu.com/s/1nkCw0V2TZiuCfQJfCL4c4w?pwd=kg3c) & [res_luna16](https://pan.baidu.com/s/1HW86ofYnYDLzIlRFZzfboQ?pwd=3dn0) [res_tianchi](https://pan.baidu.com/s/1aa2cng9lHvNpFnRQxEefsA?pwd=uqd6) [res_russia](https://pan.baidu.com/s/1_SAVl5LaS_apc6RqpIc19w?pwd=jscu) |
NoduleNet+SE         | 3         | -       | 39.54M  | 78.89  | 72.33   | 35.89  | 62.37 | [model](https://pan.baidu.com/s/1aQeRidyCbZK525eNEjlTIQ?pwd=o4lh) & [res_luna16](https://pan.baidu.com/s/16sFuLrJueH9Ov_pbBG6TjQ?pwd=ggpn) [res_tianchi](https://pan.baidu.com/s/1d59P58Rfe68nc84sN4VIAw?pwd=o1rl) [res_russia](https://pan.baidu.com/s/1HUvK6O2lDoyDN7TyOGRMNw?pwd=km5g) |
DANoduleNet          | 3         | -       | 39.54M  | **82.63** | 73.29| 38.50  | 64.80 | [model](https://pan.baidu.com/s/1W2p77Dx5eaoPqNJPK0YZuA?pwd=9vqx) & [res_luna16](https://pan.baidu.com/s/1iH7X0iv_izVIDoVXWH-22g?pwd=09pp) [res_tianchi](https://pan.baidu.com/s/1m4F-sv4I2fKWU71H4G0V3w?pwd=tzkq) [res_russia](https://pan.baidu.com/s/1nGNR4JOcYQl5su8EUUlLUQ?pwd=x6md) |
single NoduleNet+SGSE| -         | 4       | 16.77Mx3| 78.30  | 70.36   | **39.01** |62.55|[model_luna16](https://pan.baidu.com/s/1UpEO1PcRY0TIKmj_siOBYA?pwd=e0oy) [model_tianchi](https://pan.baidu.com/s/1Cb8nMXu3rIw4ptGNtuPKoQ?pwd=xpvw) [model_russia](https://pan.baidu.com/s/1cQ4ltfsS0T1VfA5zwgWugQ?pwd=0hsj) & [res_luna16](https://pan.baidu.com/s/1I9C4f2vp6MOcSh4exIBgDA?pwd=iwn9) [res_tianchi](https://pan.baidu.com/s/1CSDJnZhc5IQ_JqtYzK8TpQ?pwd=nboq) [res_russia](https://pan.baidu.com/s/12qJGPh7e4bblf_ncyyP5Pg?pwd=plzr)|
uniSGSENoduleNet     | -         | 4       | 39.54M  | 81.12  | 71.00   | 38.42  | 63.51 | [model](https://pan.baidu.com/s/1kt_lDdkv42wNA1Q0KdEW1w?pwd=hcvl) & [res_luna16](https://pan.baidu.com/s/1E9Kqns2It0T1HhO3oeMWFw?pwd=1jr9) [res_tianchi](https://pan.baidu.com/s/19uoq76TOyqf8gfvnfoy-AQ?pwd=6oip) [res_russia](https://pan.baidu.com/s/1Uo0TrSgeWfmtds7vso8Z8g?pwd=lmal) |
NoduleNet+SGSE       | 3         | 4       | 39.62M  | 80.93  | 70.94   | 38.30  | 63.39 | [model](https://pan.baidu.com/s/1TwjR4sr7epa-JnhdqXLClA?pwd=sdm3) & [res_luna16](https://pan.baidu.com/s/1reYcOVK2dC2fS2rf-ydOnQ?pwd=k9xk) [res_tianchi](https://pan.baidu.com/s/1iDOz2_WFqoYV9EubdZQfTA?pwd=yhr4) [res_russia](https://pan.baidu.com/s/1BAzOaBoN40qLBV_xFZjrqg?pwd=64b3) |
SGDANoduleNet        | 3         | 4       | 39.82M  | 81.91  | **77.13** |37.15 | **65.39**|[model](https://pan.baidu.com/s/1ItbhOaSRT_mqpM8VJv9-5A?pwd=gbmy) & [res_luna16](https://pan.baidu.com/s/10nLyRFeqrUYI4Sxa412xFw?pwd=dsrl) [res_tianchi](https://pan.baidu.com/s/1woIwU0ap-tbkYRtmI8X-eA?pwd=lj5m) [res_russia](https://pan.baidu.com/s/1T9Jfq2k9CKDftz3L_EGPVA?pwd=ok86) | 

Comparison of our SGDA and other multi-domain methods in terms of FROC on dataset PN9. The values are pulmonary nodule
detection sensitivities (unit: %) with each column representing the average number of false positives per CT image. All
the methods utilizes SANet as backbone: (1) baseline model with the prefix 'uni-', (2) universal models with 'SG' in 
the name (Ours).

|Method          | #Adapters | #Groups | #Params | 0.125  | 0.25   | 0.5    | 1.0    | 2.0    | 4.0    | 8.0    | Avg    | Pre-trained Model   |
|--------------- | --------- | ------- | ------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |-------------------- |
uniSANet         | -         | -       | 15.28M  | 38.08  | 45.05  | 54.46  | 64.50  | 75.33  | 83.86  | 89.96  | 64.46  | [model](https://drive.google.com/file/d/1oGCsekgLAsZZl8VN3QrqAgZqzMZr_kks/view) & [res](https://pan.baidu.com/s/1n07xqxqx09TYv8uwEbwxdQ?pwd=19mb) |
DASANet          | 3         | -       | 15.32M  | 54.86  | 54.86  | 54.86  | 64.94  | 75.43  | 83.53  | 88.18  | 68.09  | [model](https://pan.baidu.com/s/14OHV9hPyYMkTFBEs-Cw2xA?pwd=399f) & [res](https://pan.baidu.com/s/1vFLNrQLFHR0VPuflbxvvIw?pwd=wllw) |
*SGDASANet w/o CA| 3         | 4       | 15.36M  | 52.06  | 52.06  | **58.63** | **66.33** | **77.05** | **85.13** | **90.12** | 68.77 |[model](https://pan.baidu.com/s/1GrtkH4zF4TlpBYwc7LGHxw?pwd=uxfr) & [res](https://pan.baidu.com/s/1AdHkyFxwY2riyPsHSWiGwg?pwd=4w59)|
*SGDASANet w/ CA | 3         | 4       | 15.45M  | **57.63** | **57.63** | 57.63 | 65.73 | 75.09 | 83.56 | 88.25| **69.36** |[model](https://pan.baidu.com/s/1gCyr1gBN0geURMXqTUlQCw?pwd=2tlz) & [res](https://pan.baidu.com/s/1DsHlvNU8e8fFpZVUTrBjcw?pwd=44wi)|


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
LUNA16  | 2016 | 601   | 1186    | 2     | Yes | 25M-258M  | 512x512x95-512x512x733   | (0.86,0.86,2.50)-(0.64,0.64,0.50) | [link](https://luna16.grand-challenge.org/) & [split](https://pan.baidu.com/s/1cseCgWM5ezZzCDE4vUk7rQ?pwd=tvbv) |
tianchi | 2017 | 800   | 1244    | 2     | Yes | 26M-343M  | 512x512x114-512x512x1034 | (0.66,0.66,2.50)-(0.69,0.69,0.30) | [link](https://tianchi.aliyun.com/competition/entrance/231601/introduction) & [split](https://pan.baidu.com/s/1yShLJY49s0FWQzVZfD-4Fw?pwd=8ikj) |
russia  | 2018 | 364   | 1850    | 2     | Yes | 80M-491M  | 512x512x313-512x512x1636 | (0.62,0.62,0.80)-(0.78,0.78,0.40) | [link](https://mosmed.ai/en/data-sets/ct_lungcancer_500/) & [split](https://pan.baidu.com/s/1x3y56n_aMilvccCtywqg6A?pwd=7yzz)|
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
