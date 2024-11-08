# [EMNLP 2024 Findings] &nbsp; QEFT: Quantization for Efficient Fine-Tuning of LLMs 

This is the code for the paper [QEFT: Quantization for Efficient Fine-Tuning of LLMs](https://arxiv.org/abs/2410.08661).

## Table of contents
* [Install](#install)
* [Usage](#usage)
* [CUDA kernel](#cuda-kernels)

## Files
### Main Files
* **main.py**: OWQ reconstruction and packing model saving. End-to-end benchmark is executed in this file. To use OGR, run **extract_outidx.py** first to obtain the global index and pass it as an argument.
* **extract_outidx.py**: Extracts Global Index for Offline Global Reordering (OGR).
* **recon.py**: Implements GPTQ or OWQ algorithms and handlers.
* **reorder.py**: Implements functions for Offline Global Reordering.
* **finetune.py**: Weak Column Tuning script. Requires the packed model from **main.py**.

### Kernel/Benchmark Related
* **quant.py**: Defines the fake quantizer.
* **qlinear.py**: Defines QuantLinear. The kernel used during forward pass is specified with the set_kernel method.
* **kernel/attention**: FasterTransformer attention kernel, sourced from AWQ.
* **kernel/layernorm**: FasterTransformer layernorm kernel, sourced from AWQ.
* **kernel/quantization_new**: Source files for 4-bit GeMV, GEMM, and dequantize kernels.
* **kernel/quantization_new/gemv/gemv_cuda_qeft.cu**: GeMV kernel for OGR format (W4/16 mixed precision), optimized for k=128; validity for non-multiples is unverified.
* **monkeypatch/**: Monkeypatch files for the LLaMA model implementation with FasterTransformer. Requires installation of the attention kernel. The implementation is tailored for version 4.42.3.
* **benchmark.py**: Script for measuring generation throughput of FP16, Platypus (INT8+LoRA), QLoRA (NF4+LoRA), AWQ, OWQ, and QEFT.

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

### Reconstruction and Save Packed Model
```bash
# Extract global indices for OGR
CUDA_VISIBLE_DEVICES=0 python -m qeft.extract_outidx meta-llama/Llama-2-7b-hf c4 --wbits 4 --target_rank 128 --seed 42 --no_frob_norm --output_dir global_indices/llama2-7b

# Reconstruction with OGR
CUDA_VISIBLE_DEVICES=0 python -m qeft.main meta-llama/Llama-2-7b-hf c4 --wbits 4 --target_rank 128 --groupsize 128 --dtype fp16 --seed 42 --reorder --outidx_file global_indices/llama2-7b/w4_r128/outidx.pth --packing --save llama2-7b_w4_g128_r128.pth
```

#### Reconstruction with GPTQ (no mixed-precision, just 4bit)
```bash
CUDA_VISIBLE_DEVICES=0 python -m qeft.main meta-llama/Llama-2-7b-hf c4 --wbits 4 --groupsize 128 --dtype fp16 --seed 42 --packing --save llama2-7b_w4_g128.pth
```

### Measure PPL Using Packed Model (MatMul). Result is Equal to Reconstruction
```bash
CUDA_VISIBLE_DEVICES=0 python -m qeft.main meta-llama/Llama-2-7b-hf c4 --load llama2-7b_w4_g128_r128.pth
```

### Benchmark End-to-End Generation (MatVec).
```bash
CUDA_VISIBLE_DEVICES=0 python -m qeft.main meta-llama/Llama-2-7b-hf c4 --benchmark 128 --load llama2-7b_w4_g128_r128.pth

# With FasterTransformer
CUDA_VISIBLE_DEVICES=0 python -m qeft.main meta-llama/Llama-2-7b-hf c4 --benchmark 128 --load llama2-7b_w4_g128_r128.pth --ft
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