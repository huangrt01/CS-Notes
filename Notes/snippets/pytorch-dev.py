
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