# HarmoniDiff-RS: Training-Free Diffusion Harmonization for Satellite Image Composition

This is the official implementation of the paper ["**HarmoniDiff-RS: Training-Free Diffusion Harmonization for Satellite Image Composition**"](https://arxiv.org/abs/2604.19392).

HarmoniDiff-RS is a training-free framework designed to achieve high-quality harmonization for remote sensing (RS) images by leveraging the generative power of Diffusion Models.

------

## 🛠️ 1. Installation

We recommend using **Conda** for environment management to ensure dependency compatibility.

### Step 1: Clone the Repository

```
git clone https://github.com/XiaoqiZhuang/HarmoniDiff-RS.git
cd HarmoniDiff-RS
```

### Step 2: Create Environment

```
conda create -n harmonidiff python=3.10 -y
conda activate harmonidiff
```

### Step 3: Install PyTorch

Please install the appropriate version of PyTorch for your CUDA environment. For example, for CUDA 12.1:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Install Dependencies

```
pip install -r requirements.txt
```

------

## 🚀 2. Quick Start (Inference)

Our pipeline integrates the `DiffusionSat` base model and a custom `HarmoniDiff` pipeline.

### Weights Preparation

- **Base Model**: The script will automatically download `BiliSakura/DiffusionSat-Single-512` from Hugging Face.
- **Discriminator**: The discriminator is placed in the `./checkpoints/` directory

### Run Inference

Execute the following command to harmonize a foreground-background pair:

```
python inference.py \
    --model_id "BiliSakura/DiffusionSat-Single-512" \
    --discriminator_path "./checkpoints/discriminator.pt" \
    --dtype "fp16" \
    --visualization_path "./visualization" 
```

------

## 📊 3. RSIC-H Dataset 

The dataset used in our paper, which is an adapted version derived from **fMoW**, is now publicly available on **Zenodo**.

- **Zenodo Repository**: [https://zenodo.org/records/20187102](https://zenodo.org/records/20187102)

> **Note**: This dataset is released under the **Functional Map of the World Challenge Public License**.

------

## 📝 4. Citation

If you find our work useful for your research, please consider citing:



```
@article{zhuang2026harmonidiff,
  title={HarmoniDiff-RS: Training-Free Diffusion Harmonization for Satellite Image Composition},
  author={Zhuang, Xiaoqi and Dos Santos, Jefersson A. and Han, Jungong},
  journal={arXiv preprint arXiv:2604.19392},
  year={2026}
}
```

------

## 🤝 Acknowledgements

This project is built upon the following open-source repositories:

- [diffusers](https://github.com/huggingface/diffusers)
- [DiffusionSat](https://github.com/samar-khanna/DiffusionSat)
- [fMoW](https://github.com/fMoW/dataset)

------

