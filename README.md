# DCAM WIP

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

Quantitative results on Adobe Composition-1K
| Model Name  |   Size   | MSE | SAD | Grad | Conn |
| :------------: |:-----------:| :----:|:---:|:---:|:---:|
| [DCAM]() | 181MiB | 3.34 | 22.62 | 7.67 | 18.02 |

## Evaluation
We provide the script `eval.py`  for evaluation.


