# (ICCV 2025) FlowTok: Flowing Seamlessly Across Text and Image Tokens

This repository provides a **PyTorch re-implementation** of FlowTok for the text-to-image generation task.
Compared to the original paper, this implementation extends the generation capability to 512×512 resolution.

>  FlowTok: Flowing Seamlessly Across Text and Image Tokens
>
>  ICCV 2025
>
>  [Ju He](https://tacju.github.io/) | [Qihang Yu](https://yucornetto.github.io/) | [Qihao Liu](https://qihao067.github.io/) | [Liang-Chieh Chen](http://liangchiehchen.com/)
>
>  [[project page](https://tacju.github.io/projects/flowtok.html)] | [[paper](https://arxiv.org/pdf/2503.10772)] | [[arxiv](https://arxiv.org/abs/2503.10772)]

![teaser](https://github.com/TACJu/FlowTok/blob/main/imgs/FlowTok.png)

______

## Setup

- ### Environment

  The code has been tested with PyTorch 2.1.2 and Cuda 12.1.

  An example of installation commands is provided as follows:

  ```
  git clone git@github.com:tacju/FlowTok.git
  cd FlowTok
  
  pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
  pip3 install -U --pre triton
  pip3 install -r requirements.txt
  ```

## Training FlowTok for T2I

We provide a training script for text-to-image (T2I) generation in [`train_flowtok.sh`](https://github.com/tacju/FlowTok/blob/main/train_flowtok.sh). 

## Reproduced checkpoints

We release our re-trained checkpoints for [FlowTok-XL](https://huggingface.co/turkeyju/FlowTok/blob/main/FlowTok-XL.pth), [FlowTok-H](https://huggingface.co/turkeyju/FlowTok/blob/main/FlowTok-H.pth) and [FlowTiTok-512](https://huggingface.co/turkeyju/FlowTok/blob/main/FlowTiTok_512.bin) at HuggingFace.

------

## Terms of use

The project is created for research purposes.

______

## Acknowledgements

This codebase is built upon the following repository:

- [[TA-TiTok](https://github.com/bytedance/1d-tokenizer)]
- [[CrossFlow](https://github.com/qihao067/CrossFlow)]
- [[U-ViT](https://github.com/baofff/U-ViT)]
- [[DiT](https://github.com/facebookresearch/DiT)]
- [[DiMR](https://github.com/qihao067/DiMR)]
- [[DeepFloyd](https://github.com/deep-floyd/IF)]

Much appreciation for their outstanding efforts.

______

## BibTeX

If you use our work in your research, please use the following BibTeX entries.

```BibTeX
@article{he2025flowtok,
  author    = {Ju He and Qihang Yu and Qihao Liu and Liang-Chieh Chen},
  title     = {FlowTok: Flowing Seamlessly Across Text and Image Tokens},
  journal   = {ICCV},
  year      = {2025}
}
```

```BibTeX
@article{liu2025crossflow,
  author    = {Qihao Liu and Xi Yin and Alan Yuille and Andrew Brown and Mannat Singh},
  title     = {Flowing from Words to Pixels: A Noise-Free Framework for Cross-Modality Evolution},
  journal   = {CVPR},
  year      = {2025}
}
```

```BibTeX
@article{kim2025democratizing,
  author    = {Dongwon Kim and Ju He and Qihang Yu and Chenglin Yang and Xiaohui Shen and Suha Kwak and Liang-Chieh Chen},
  title     = {Democratizing Text-to-Image Masked Generative Models with Compact Text-Aware One-Dimensional Tokens},
  journal   = {ICCV},
  year      = {2025}
}
```

```BibTeX
@article{yu2024an,
  author    = {Qihang Yu and Mark Weber and Xueqing Deng and Xiaohui Shen and Daniel Cremers and Liang-Chieh Chen},
  title     = {An Image is Worth 32 Tokens for Reconstruction and Generation},
  journal   = {NeurIPS},
  year      = {2024}
}
```