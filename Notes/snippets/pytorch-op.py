### op执行

gpu op是异步的（需要h2d触发或者主动调用torch.cuda.synchronize），cpu op是主进程同步的

### load_inline

pip3 install ninja

import torch
from torch.utils.cpp_extension import load_inline

cpp_source = """
std::string hello_world() {
  return "Hello World!";
}
"""

my_module = load_inline(
    name='my_module',
    cpp_sources=[cpp_source],
    functions=['hello_world'],
    verbose=True,
    build_directory='./tmp'
)

print(my_module.hello_world())


# Define the CUDA kernel and C++ wrapper

cuda_begin = r'''
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
'''

cuda_source = cuda_begin + '''
__global__ void square_matrix_kernel(const float* matrix, float* result, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) { 
        int idx = row * width + col;
        result[idx] = matrix[idx] * matrix[idx];
    }
}

torch::Tensor square_matrix(torch::Tensor matrix) {
    CHECK_INPUT(matrix)
    const auto height = matrix.size(0);
    const auto width = matrix.size(1);

    auto result = torch::empty_like(matrix);

    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks(cdiv(width, threads_per_block.x), cdiv(height, threads_per_block.y));

    square_matrix_kernel<<<number_of_blocks, threads_per_block>>>(
        matrix.data_ptr<float>(), result.data_ptr<float>(), width, height);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
    }
'''

cpp_source = "torch::Tensor square_matrix(torch::Tensor matrix);"

def load_cuda(name, cuda_source, cpp_source, funcs, opt=False, verbose=False):
  return load_inline(name=name,
                     cuda_sources=[cuda_src],
                     cpp_sources=[cpp_src],
                     functions=funcs,
                     extra_cuda_cflags=["-O2"] if opt else [],
                     verbose=verbose,
                     # extra_cuda_cflags=['--expt-relaxed-constexpr']
                     build_directory='./load_inline_cuda')
                     


# Load the CUDA kernel as a PyTorch extension

square_matrix_extension = load_cuda(cuda_src, cpp_source, ['square_matrix'], verbose=True)


a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device='cuda')
print(square_matrix_extension.square_matrix(a))


### 写Op

TODO： https://pytorch.org/tutorials/advanced/cpp_extension.html

https://pytorch.org/tutorials/advanced/python_custom_ops.html

* 关于非contiguous

如果我们只是用[]/(), 索引，他们都是操作符重载，内部考虑了shape, stride, order, offset等，不会出错。在很多情况下可以节省大量内存
但是我们拿指针出来操作数据的所有情况，都要保证是contiguous的， 否则可能出错。


### argsort

sort_indices = torch.argsort(shard_ids, stable=True)

### concat split
split的输出是tuple，经常需要转为list


### jagged/padding


def generate_row_splits_from_row_lengths(
    row_lengths: torch.Tensor) -> torch.Tensor:
  row_splits = torch.cat([
      torch.tensor([0], dtype=torch.int32, device=row_lengths.device),
      torch.cumsum(row_lengths, dim=0, dtype=torch.int32)
  ])
  return row_splits


q = torch.ops.fbgemm.jagged_to_padded_dense(values=q_varlen,
                                          offsets=[row_splits],
                                          max_lengths=[max_length],
                                          padding_value=0)

### rearrange

from einops import rearrange

q = rearrange(q, 'b t (h d) -> b t h d', h=nheads)

### Example


class All2AllSingle(torch.autograd.Function):
  @staticmethod
  def forward(ctx: Any, tensor: torch.Tensor,
              output_split_sizes: list[int],
              input_split_sizes: list[int]) -> Any:
    ctx.output_split_sizes = output_split_sizes
    ctx.input_split_sizes = input_split_sizes

    output_shape = (sum(output_split_sizes), tensor.size(1))
    output_tensor = torch.empty(
      output_shape, dtype=tensor.dtype, device=tensor.device)
    dist.all_to_all_single(output_tensor, tensor,
                           output_split_sizes=output_split_sizes,
                           input_split_sizes=input_split_sizes)
    return output_tensor

  @staticmethod
  def backward(ctx: Any, *grad_outputs: Any) -> Any:
    assert len(grad_outputs) == 1
    tensor_grad = grad_outputs[0]
    output_split_sizes = ctx.output_split_sizes
    input_split_sizes = ctx.input_split_sizes
    output_shape = (sum(input_split_sizes), tensor_grad.size(1))
    output_tensor_grad = torch.empty(
      output_shape, dtype=tensor_grad.dtype, device=tensor_grad.device)

    assert tensor_grad.is_contiguous(), f"tensor_grad not contiguous, {tensor_grad.shape}, {tensor_grad}"
    dist.all_to_all_single(output_tensor_grad, tensor_grad,
                           output_split_sizes=input_split_sizes,
                           input_split_sizes=output_split_sizes)
    return output_tensor_grad, None, None