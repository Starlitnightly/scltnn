# Installation guild

## Pip

The `scltnn` package can be installed via pip using one of the following commands:

Install [PyTorch](https://pytorch.org/get-started/locally/) at first: More about the installation can be found at [PyTorch](https://pytorch.org/get-started/locally/). 

```shell
# ROCM 5.2 (Linux only)
pip3 install torch torchvision torchaudio --extra-index-url
pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/rocm5.2
# CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
# CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# CPU only
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
```

## Main package

The `scltnn` package can be installed via pip using one of the following commands:

```
pip install -U scltnn
```

**Note** To avoid potential dependency conflicts, installing within a pip environment is recommended.

