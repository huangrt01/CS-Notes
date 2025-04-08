### 安装 - build from source

https://pytorch.org/get-started/locally/ 查看torch和cuda稳定版本

# https://github.com/pytorch/pytorch#from-source

# anaconda
# https://www.anaconda.com/download
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
chmod +x Anaconda3-2024.10-1-Linux-x86_64.sh 
./Anaconda3-2024.10-1-Linux-x86_64.sh

source $HOME/anaconda3/bin/activate 
# conda create -n pytorch-debug
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



./tools/nightly.py checkout -b my-nightly-branch -p my-env

export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
CUDA_DEVICE_DEBUG=1 DEBUG=1 python setup.py develop



### debug

`cuda-gdb` and `cuda-memcheck`


### 部分构件相关

#### 增量build
python setup.py develop

# 如果需要清理后重新构建（完全重建）
python setup.py clean && python setup.py develop

For subsequent builds (i.e., when build/CMakeCache.txt exists), the build options passed for the first time will persist; 
please run ccmake build/, run cmake-gui build/, or directly edit build/CMakeCache.txt to adapt build options.

#### 只构建test

Working on a test binary?
Run (cd build && ninja bin/test_binary_name) to rebuild only that test binary (without rerunning cmake). 
(Replace ninja with make if you don't have ninja installed)

#### ccache

sudo apt install ccache
# config: cache dir is ~/.ccache, conf file ~/.ccache/ccache.conf
# max size of cache
ccache -M 25Gi  # -M 0 for unlimited
# unlimited number of files
ccache -F 0

To check this is working, do two clean builds of pytorch in a row. The second build should be substantially and noticeably faster than the first build. If this doesn't seem to be the case, check the CMAKE_<LANG>_COMPILER_LAUNCHER rules in build/CMakeCache.txt, where <LANG> is C, CXX and CUDA. Each of these 3 variables should contain ccache, e.g.

//CXX compiler launcher
CMAKE_CXX_COMPILER_LAUNCHER:STRING=/usr/bin/ccache

If not, you can define these variables on the command line before invoking setup.py.

export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
python setup.py develop

#### Rebuild few files with debug information



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