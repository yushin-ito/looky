# Looky

[![python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
![version](https://img.shields.io/badge/version-3.0.0-red.svg)
![stars](https://img.shields.io/github/stars/yushin-ito/looky?color=yellow)
![commit-activity](https://img.shields.io/github/commit-activity/t/yushin-ito/looky)
![license](https://img.shields.io/badge/license-MIT-green)

<br/>

## 🚀 Usage

1. Clone this repository

```bash
git clone https://github.com/yushin-ito/looky.git
```

<br/>

2. Move to the directory

```bash
cd looky
```

<br/>

3. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

<br/>

3. Install the packages

```bash
pip install -r requirements.txt
```

<br/>
<br/>

## ⚡️ Quick Start

1. Prepare the dataset
```bash
$ bash scripts/prepare_dataset.sh
```

<br/>

2. Train the model
```bash
$ bash scripts/train_garment.sh
$ bash scripts/train_vton.sh
```

<br/>

3. Inference with model
```bash
$ bash scripts/inference.sh
```

<br/>
<br/>

## 📂 Structure

```
looky/
├── src/
│   └── looky/
│       ├── models/
│       │   ├── __init__.py
│       │   ├── embeddings.py
│       │   ├── transformer_garment.py
│       │   └── transformer_vton.py
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── pipeline_agnostic_mask_generation.py
│       │   ├── pipeline_output.py
│       │   └── pipeline_virtual_try_on.py
│       ├── dwpose.py
│       └── frequency_loss.py
├── scripts/
│   ├── inference.py
│   ├── inference.sh
│   ├── prepare_dataset.py
│   ├── prepare_dataset.sh
│   ├── train_garment.py
│   ├── train_garment.sh
│   ├── train_vton.py
│   └── train_vton.sh
├── notebooks/
│   └── example.ipynb
├── data/
│   ├── train/
│   └── test/
├── weights/
│   ├── human_parsing/
│   ├── pose_estimation/
│   └── virtual_try_on/
├── pyproject.toml
├── requirements.txt
├── README.md
└── LICENSE
```

<br/>
<br/>

## 🤝 Contributer

<a href="https://github.com/yushin-ito">
  <img  src="https://avatars.githubusercontent.com/u/75526539?s=48&v=4" width="64px">
</a>

<br/>
<br/>

## 📜 LICENSE

[MIT LICENSE](LICENSE)
