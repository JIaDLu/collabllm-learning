# CollabLLM: From Passive Responders to Active Collaborators  

<div align="left">

[![](https://img.shields.io/badge/Website-CollabLLM-purple?style=plastic&logo=Google%20Chrome)](http://aka.ms/CollabLLM)
[![](https://img.shields.io/badge/Datasets_&_Models-HuggingFace-yellow?style=plastic&logo=Hugging%20Face)](https://huggingface.co/collabllm)
[![](https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=arxiv)](https://arxiv.org/pdf/2502.00640)
[![](https://img.shields.io/badge/PyPI-collabllm-brightgreen?style=plastic&logo=Python)](https://pypi.org/project/collabllm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

📢 Oustanding Paper Award @ ICML 2025

#### News: You can now train CollabLLM on [verl](https://verl.readthedocs.io/en/latest/algo/collabllm.html)! 😄

# Overview
CollabLLM transforms traditional language models from passive responders to active collaborators in multi-turn conversations. This repository provides the complete framework for computing multiturn-aware rewards and training collaborative language models.

---
## Installation

To get started, create a new environment and install `collabllm` via [pip](https://pypi.org/project/collabllm/):

```bash
conda create -n collabllm python=3.10
conda activate collabllm
pip install collabllm
```

### Optional: For distributed training
If you need distributed training:

```bash
pip install deepspeed
conda install mpi4py
```

### Optional: For customized datasets and metrics
You may install additional packages (e.g., `pip install bigcodebench matplotlib`) for task-specific metrics or evaluation.

# Quick Start

- Lightweight usage: Compute Multiturn-aware Rewards (MRs) for any model responses and construct datasets following `notebook_tutorials/`.
- Synthetic data generation: Generating high-quality synthetic conversational data following `scripts/engine/build_dataset.py`. (Include your API keys in `.env` file)
- Train CollabLLM: Conduct SFT/DPO models training to maximize MRs following examples under `scripts/train/*.py`. 


## Add Your Own Task

To apply CollabLLM to a new task:

1. **Add a Dataset:**  
   Place your single-turn dataset in `examples/single_turn_ds/` and register it in `__init__.py`.

2. **(Optional) Add Metrics:**  
   Add new metrics to `examples/metrics/` and register them in `__init__.py`.

You can now run data generation, reward computation, and model training using your customized setup.


# Citation
If you find our work useful in your research, please cite the following:

```bibtex
@inproceedings{collabllm2025,
    title={CollabLLM: From Passive Responders to Active Collaborators},
    author={Shirley Wu and Michel Galley and Baolin Peng and Hao Cheng and 
            Gavin Li and Yao Dou and Weixin Cai and James Zou and 
            Jure Leskovec and Jianfeng Gao},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2025}
}
```
