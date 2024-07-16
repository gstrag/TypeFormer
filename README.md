# TypeFormer: Transformers for Mobile Keystroke Biometrics

This repository contains the source code of TypeFormer, a novel Transformer-based mobile keystroke verification system proposed in [\[1\]](https://ieeexplore.ieee.org/abstract/document/10042710). The model development and evaluation described is carried out over the [Aalto Mobile Keystroke Database](https://userinterfaces.aalto.fi/typing37k/resources/Mobile_typing_study.pdf). 
The experimental protocol is the same as in [\[2\]](https://ieeexplore.ieee.org/abstract/document/10042710), [\[3\]](https://ieeexplore.ieee.org/document/9539873). 

# Model Development

The model was developed from the following sources:

- https://github.com/getalp/Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices 
- https://github.com/lucidrains/block-recurrent-transformer-pytorch 
- https://github.com/BiDAlab/ExploringTransformers 

# Repo Description

In this repository, we provide:
- The source code for TypeFormer [\[1\]](https://ieeexplore.ieee.org/abstract/document/10042710), and for the preliminary Transformer in [\[2\]](https://ieeexplore.ieee.org/abstract/document/10042710) in `model/Model.py` and in `model/Preliminary.py`.
- The scripts to train (`train.py`) and evaluate (`test.py`) said models on the [Aalto Mobile Keystroke Database](https://userinterfaces.aalto.fi/typing37k/resources/Mobile_typing_study.pdf).
- The script to plot DET curves (`plot_DET.py`).
- The script to plot loss and EER on the training and validation sets over the training epochs  (`read_log.py`).
- The script to plot t-SNE representation (`tSNE.py`).
- Pretrained TypeFormer (`pretrained/TypeFormer_pretrained.pt`) and the preliminary Transformer (`pretrained/preliminary_transformer_pretrained.pt`).
- The [preprocessed data]() on which we perform the experiments should be placed in `data/`. 

# References

[\[1\] *Giuseppe Stragapede, Paula Delgado-Santos, Ruben Tolosana, Ruben Vera-Rodriguez, Richard Guest, and Aythami Morales, “TypeFormer: Transformers for Mobile Keystroke Biometrics”, arXiv:2207.07596, 2023.*](https://ieeexplore.ieee.org/abstract/document/10042710)

[\[2\] *Giuseppe Stragapede, Paula Delgado-Santos, Ruben Tolosana, Ruben Vera-Rodriguez, Richard Guest, and Aythami Morales, “Mobile Keystroke Biometrics Using Transformers”, Proc. of the Int. Conf. on Face and Gesture Recognition (FG), 2023.*](https://ieeexplore.ieee.org/abstract/document/10042710) 

[\[3\] *Alejandro Acien, Aythami Morales, John V. Monaco, Ruben Vera-Rodriguez, and Julian Fierrez, "TypeNet: Deep learning keystroke biometrics." IEEE Transactions on Biometrics, Behavior, and Identity Science (TBIOM), 4.1 (2021): 57-70, 2021.*](https://ieeexplore.ieee.org/document/9539873)

If you use any of the parts of this repo, please cite: 

```

@article{stragapede2022typeformer,
  title={TypeFormer: Transformers for mobile keystroke biometrics},
  author={Stragapede, Giuseppe and Delgado-Santos, Paula and Tolosana, Ruben and Vera-Rodriguez, Ruben and Guest, Richard and Morales, Aythami},
  journal={arXiv preprint arXiv:2212.13075},
  year={2022}
}

@inproceedings{stragapede2023mobile,
  title={Mobile keystroke biometrics using transformers},
  author={Stragapede, Giuseppe and Delgado-Santos, Paula and Tolosana, Ruben and Vera-Rodriguez, Ruben and Guest, Richard and Morales, Aythami},
  booktitle={2023 IEEE 17th International Conference on Automatic Face and Gesture Recognition (FG)},
  pages={1--6},
  year={2023},
  organization={IEEE}
}

```
