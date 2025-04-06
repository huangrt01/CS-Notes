### 安装 - build from source

https://pytorch.org/get-started/locally/ 查看torch和cuda稳定版本

# https://github.com/pytorch/pytorch#from-source

# anaconda
# https://www.anaconda.com/download
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
chmod +x Anaconda3-2024.10-1-Linux-x86_64.sh 
./Anaconda3-2024.10-1-Linux-x86_64.sh

source $HOME/anaconda3/bin/activate 
conda create -n pytorch-debug
conda activate pytorch-debug

# cudnn
cat /usr/include/cudnn_version.h 

https://developer.nvidia.com/cudnn-downloads

wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cudnn
sudo apt-get -y install cudnn-cuda-12 # 11 or 12

# cuda compiler
https://gist.github.com/ax3l/9489132

# pytorch
https://github.com/pytorch/pytorch?tab=readme-ov-file#from-source

git clone --recursive https://github.com/pytorch/pytorch --depth 1
cd pytorch
git submodule sync
git submodule update --init --recursive

conda install cmake ninja
pip install -r requirements.txt

pip install mkl-static mkl-include
# CUDA only: Add LAPACK support for the GPU if needed
conda install -c pytorch magma-cuda121  # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo

# (optional) If using torch.compile with inductor/triton, install the matching version of triton
# Run from the pytorch directory after cloning
# For Intel GPU support, please explicitly `export USE_XPU=1` before running command.
make triton

export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
CUDA_DEVICE_DEBUG=1 DEBUG=1 python setup.py develop



### debug build

CUDA_DEVICE_DEBUG=1
DEBUG=1


`cuda-gdb` and `cuda-memcheck`


### write op

二次开发autograd
* non-tensor在backward时一定返回None



### cuda kernel

https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#cuda-development-tips

https://github.com/pytorch/pytorch/wiki/CUDA-basics


### load_inline

#include <torch/extension.h>
torch::Tensor square_matrix(torch::Tensor matrix);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("square_matrix", torch::wrap_pybind_function(square_matrix), "square_matrix");
}


# setup.py

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='square_matrix',
    ext_modules=[
        CppExtension('square_matrix', ['square_matrix.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })



### gradient hook

https://www.digitalocean.com/community/tutorials/pytorch-hooks-gradient-clipping-debugging