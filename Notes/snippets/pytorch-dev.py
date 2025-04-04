### 安装

https://pytorch.org/get-started/locally/ 查看torch和cuda稳定版本


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