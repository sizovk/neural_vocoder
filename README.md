# Neural Vocoder

This repository contains code for training of HiFi-GAN model, which have been described in the article https://arxiv.org/abs/2010.05646

## [Training report](https://wandb.ai/k_sizov/neural_vocoder/reports/HiFi-GAN-training--VmlldzozMjAyODYy?accessToken=x8cpyykpqx9vubx1rvha50uouephgtjqil3bfkwxlr1nt11yqppqch06vg6kktdl)

## Reproduce results
### Setup data
```bash
pip install -r requirements.txt
bash data_setup.sh
```

### Train model (V3 configuration)
```bash
python train.py -c configs/default.json 
```