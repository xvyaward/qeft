# [EMNLP 2024 Findings] QEFT: Quantization for Efficient Fine-Tuning of LLMs 

This is the code for the paper [QEFT: Quantization for Efficient Fine-Tuning of LLMs](https://arxiv.org/abs/2410.08661).

## Table of contents
* [Install](#install)
* [Usage](#usage)

## Install
We highly recommend using a Docker image that supports CUDA. If you prefer Anaconda, you need to set up CUDA for kernel use.

0. A) Using Docker
```
docker run -it --gpus all --ipc=host -v {local_storage}:{docker_container_storage} pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

# install git
apt update && apt install git -y
```

0. B) Using Anaconda instead of Docker
```
conda create -n owq python=3.10 -y
conda activate owq
```

1. Clone the QEFT repository
```
git clone https://github.com/xvyaward/qeft
cd QEFT_PV
```
2. Install all the dependencies
```
pip install -e .
```
3. Install the OWQ CUDA kernel
```
cd qeft/kernel
python setup_cuda.py install
```

## Usage

### 1. Reconstruction and Save Packed Model

### 1-1. Extract global indices for OGR
```bash
CUDA_VISIBLE_DEVICES=0 python -m qeft.extract_outidx meta-llama/Llama-2-7b-hf c4 --wbits 4 --target_rank 128 --seed 42 --no_frob_norm --output_dir global_indices/llama2-7b
```

### 1-2. Reconstruction with OGR
```bash
CUDA_VISIBLE_DEVICES=0 python -m qeft.main meta-llama/Llama-2-7b-hf c4 --wbits 4 --target_rank 128 --groupsize 128 --dtype fp16 --seed 42 --outidx_file global_indices/llama2-7b/w4_r128/outidx.pth --packing --save llama2-7b_w4_g128_r128.pth
```

### 2. Validate Packed Model Operation

### 2-1. Measure PPL Using Packed Model (MatMul). Result is Equal to Reconstruction
```bash
CUDA_VISIBLE_DEVICES=0 python -m qeft.main meta-llama/Llama-2-7b-hf c4 --load llama2-7b_w4_g128_r128.pth
```

### 2-2. Measure PPL Using Packed Model and Testing Acceleration(MatVec).
```bash
CUDA_VISIBLE_DEVICES=0 python -m qeft.main meta-llama/Llama-2-7b-hf c4 --benchmark 128 --load llama2-7b_w4_g128_r128.pth

# With FasterTransformer
CUDA_VISIBLE_DEVICES=0 python -m qeft.main meta-llama/Llama-2-7b-hf c4 --benchmark 128 --load llama2-7b_w4_g128_r128.pth --ft
```

### 3. Benchmark End-to-End Generation
```bash
# FP16
CUDA_VISIBLE_DEVICES=0 python -m qeft.benchmark --model_path meta-llama/Llama-2-7b-hf --method fp --ft

# QEFT 4bit
CUDA_VISIBLE_DEVICES=0 python -m qeft.benchmark --model_path meta-llama/Llama-2-7b-hf --method qeft --load ckpt/llama2-7b_w4_g128_r128.pth --ft
```

## Reference
This code is based on various implementations and research papers related to weight quantization and model optimization.

[OWQ: Outlier-Aware Weight Quantization for Efficient Fine-Tuning and Inference of Large Language Models](https://arxiv.org/abs/2306.02272)

This code is largely based on [OWQ](https://github.com/xvyaward/owq).

[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) / [Code](https://github.com/mit-han-lab/llm-awq)

[Platypus: Quick, Cheap, and Powerful Refinement of LLMs](https://arxiv.org/abs/2308.07317) / [Code](https://github.com/arielnlee/Platypus)

## Cite
If you find our code useful for your research, please consider citing:
```
@article{lee2024qeft,
  title={QEFT: Quantization for Efficient Fine-Tuning of LLMs},
  author={Lee, Changhun and Jin, Jun-gyu and Cho, Younghyun and Park, Eunhyeok},
  journal={arXiv preprint arXiv:2410.08661},
  year={2024}
}
```