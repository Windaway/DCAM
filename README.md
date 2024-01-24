# Dual-context aggregation for universal image matting (DCAM)

Official repository for the paper [**Dual-context aggregation for universal image matting**](https://link.springer.com/article/10.1007/s11042-023-17517-w)

## Description

DCAM is a universal matting network.

## Requirements
#### Hardware:

GPU memory >= 12GB for inference on Adobe Composition-1K testing set.

#### Packages:

- torch >= 1.10
- numpy >= 1.16
- opencv-python >= 4.0
- einops >= 0.3.2
- timm >= 0.4.12

## Models
**The model can only be used and distributed for noncommercial purposes.** 

Quantitative results on Adobe Composition-1K.

| Model Name  |   Size   | MSE | SAD | Grad | Conn |
| :------------: |:-----------:| :----:|:---:|:---:|:---:|
| [DCAM](https://pan.baidu.com/s/1dbn_v-qYi8rMN_DrcPUhYA?pwd=klrb) | 181MiB | 3.34 | 22.62 | 7.67 | 18.02 |

Quantitative results on Distinctions-646.
It should be noted that the matting network uses the texture difference between the foreground and the background on the Distinctions-646 dataset as a prior for prediction, which may fail on real images.

| Model Name  |   Size   | MSE | SAD | Grad | Conn |
| :------------: |:-----------:| :----:|:---:|:---:|:---:|
| [DCAM](https://pan.baidu.com/s/1u21PG2njLTEfyHajqjz88A?pwd=gtr1) | 182MiB | 4.86 | 31.27 | 25.50 | 31.72 |


## Evaluation
We provide the script `eval_dcam_adb_tri.py`  for evaluation.


