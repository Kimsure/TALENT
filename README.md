# TALENT
「CVPR2026」Official implementation of TALENT: Target-aware Efficient Learning for Referring Image Segmentation.

# Installation

```
git clone https://github.com/Kimsure/TALENT.git
cd TALENT
```

## Environment
```bash
conda create -n TALENT python=3.9.18 -y
conda activate TALENT
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirement.txt
```

## Datasets
The detailed instruction is in [prepare_datasets.md](tools/prepare_datasets.md)

## Pretrained weights
Download the pretrained weights of DiNOv2-B, DiNOv2-L and ViT-B to pretrain
```bash
mkdir pretrain && cd pretrain
## DiNOv2-B as Image encoder
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pt
## CLIP ViT-B as Text encoder
wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

# Quick Start

To evaluate TALENT, run the following script (Remember adjusting your own MODEL/DATASET PATH.)

```
bash run_scripts/test.sh
```

If you want to visualize the results, simply modify the `visualize` to `True` in the config file. 


## Weights

Our model weights have already been open-sourced and can be directly downloaded from [here](https://huggingface.co/Kimsure99/TALENT/tree/main).

# Acknowledgements

The code is based on [CRIS](https://github.com/DerrickWang005/CRIS.pytorch), [ETRIS](https://github.com/kkakkkka/ETRIS) and [DETRIS](https://github.com/jiaqihuang01/DETRIS). We thank the authors for their open-sourced code and encourage users to cite their works when applicable.

# Citation

If you find our work useful, please cite this paper:
```
@article{jin2026talent,
  title={TALENT: Target-aware Efficient Tuning for Referring Image Segmentation},
  author={Jin, Shuo and Yu, Siyue and Zhang, Bingfeng and Yao, Chao and Liu, Meiqin and Xiao, Jimin},
  journal={arXiv preprint arXiv:2604.00609},
  year={2026}
}
```

