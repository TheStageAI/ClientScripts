# Client_scripts
Bunch of test scripts for the platform.

# Install
`python -m venv /path/to/new/virtual/environment`

`source /path/to/new/virtual/environment/bin/activate`

`pip install -r requirements.txt`

# CIFAR10
### Single gpu
`CUDA_VISIBLE_DEVICES=0 python nin_cifar.py`

### Multiple gpus
`CUDA_VISIBLE_DEVICES=0,1,2,3 python nin_cifar.py`

# EDSR
### Train on single gpu
`CUDA_VISIBLE_DEVICES=0 python edsr.py`

### Train on multiple gpus
`CUDA_VISIBLE_DEVICES=0,1,2,3 python edsr.py`

### Evaluate
`python edsr.py --evaluate`
