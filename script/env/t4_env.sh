# Pyenv VirtualEnv Environment
PYTHON_PATH="/home/3/uy02093/env/py3918/bin"
export PYTHON_PATH

# Cluster
CLUSTER_NAME="tokyotech_cluster"
export CLUSTER_NAME

# Timm 
# 0.6.12
export TORCH_HOME=/gs/bs/tge-24IJ0078/pretrained/imagenet
# 0.9.2
# https://huggingface.co/docs/transformers/installation?highlight=transformers_cache#caching-models
export TRANSFORMERS_CACHE=$TORCH_HOME
export HUGGINGFACE_HUB_CACHE=$TORCH_HOME
export PYTORCH_TRANSFORMERS_CACHE=$TORCH_HOME
export HF_HOME=$TORCH_HOME

# ARSENAL Environment
ARSENAL_PATH="/gs/bs/tge-24IJ0078/pytorch_dnn_arsenal/"
export ARSENAL_PATH

# Dataset Dir 
DATA_DIR_PATH="/gs/bs/tge-24IJ0078/dataset"
export DATA_DIR_PATH



# # ======== Modules ========

# T3
# source /etc/profile.d/modules.sh
# module load cuda/10.2.89
# module load cudnn/7.6
# module load openmpi/3.1.4-opa10.10
# module load nccl/2.4.2
# module load gcc/8.3.0-cuda

# T4

source /etc/profile.d/modules.sh
module purge
module load cuda/12.1.0
module load cudnn/9.0.0
module load nccl/2.20.5

eval "module list"

# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/var/lib/tcpx/lib64:${LD_LIBRARY_PATH}

# export LD_LIBRARY_PATH=/apps/t4/rhel9/cuda/11.8.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/apps/t4/rhel9/cuda/11.8.0/lib64:$LD_LIBRARY_PATH

# export LD_LIBRARY_PATH=/apps/t4/rhel9/free/cudnn/9.0.0/cuda/12/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/apps/t4/rhel9/free/cudnn/8.9.7/cuda/11/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/apps/t4/rhel9/free/nccl/2.20.5/cuda12.3.2/lib:$LD_LIBRARY_PATH

# export LD_LIBRARY_PATH=/apps/t4/rhel9/cuda/12.1.0/lib64:$LD_LIBRARY_PATH

# CUDA 12.1
export LD_LIBRARY_PATH=/apps/t4/rhel9/cuda/12.1.0/lib64:/apps/t4/rhel9/cuda/12.1.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# cuDNN 9.0.0 (CUDA 12対応)
export LD_LIBRARY_PATH=/apps/t4/rhel9/free/cudnn/9.0.0/cuda/12/lib:$LD_LIBRARY_PATH

# NCCLライブラリ (必要に応じて)
export LD_LIBRARY_PATH=/apps/t4/rhel9/free/nccl/2.20.5/cuda12.1.0/lib:$LD_LIBRARY_PATH

echo $LD_LIBRARY_PATH

eval "nvidia-smi"