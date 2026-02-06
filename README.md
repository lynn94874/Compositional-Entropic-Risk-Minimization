# pAUC Loss Experiments (LSE)

This repository includes two experiment codes ---- Partial AUC and KL based distributionally robust optimization for paper *[A Geometry-Aware Efficient Algorithm for Compositional Entropic Risk Minimization](https://www.arxiv.org/pdf/2602.02877)*.

## Setup
```
pip install libauc==1.2.0
```

## Run
For PAUC:
```
cd pauc
python main.py --model resnet18 --dataset cifar100 --Lambda 0.1 --loss_fn SCENT --alpha_t 4 --scheduler cosine --batch_size 64 --total_epochs 60 --lr 1e-3 --momentum 0.0 --pretrained [your pretrained model directory] --freeze_backbone
```
For KL-DRO:
```
cd dro
python main.py california_housing
```


## License
MIT License. See LICENSE.
