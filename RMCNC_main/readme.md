# 2024-TKDE-RMCNC

PyTorch implementation for ''Robust Multi-view Clustering with Incomplete Information'' (TKDE 2024).

## Requirements

pytorch==1.5.0 

numpy>=1.18.2

scikit-learn>=0.22.2

munkres>=1.1.2

logging>=0.5.1.2

## Datasets

The used datasets could be downloaded from quark (链接：https://pan.quark.cn/s/93ba61e2acd3  提取码：p8Uv).

## Demo

 Train a model with different settings

```bash
## Fully Aligned or Partially Aligned
sh train.sh

## Noisy Correspondence
sh train_NC.sh

## Citation

If you find our work useful in your research, please consider citing:

```latex

@ARTICLE{sun2024RMCNC,
  author={Sun,Yuan, and Qin, Yang, and Li, Yongxiang, and Peng, Dezhong, and Peng, Xi, and Hu, Peng},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Robust Multi-View Clustering with Noisy Correspondence}, 
  year={2024},
  volume={},
  number={},
  pages={}}


@ARTICLE{yang2022SURE,
  author={Yang, Mouxing and Li, Yunfan and Hu, Peng and Bai, Jinfeng and Lv, Jiancheng and Peng, Xi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Robust Multi-View Clustering With Incomplete Information}, 
  year={2023},
  volume={45},
  number={1},
  pages={1055-1069}}
```

