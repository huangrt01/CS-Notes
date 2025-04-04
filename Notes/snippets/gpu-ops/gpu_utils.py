import os
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load_inline

# os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

import torch
import triton
import triton.language as tl


def show_img(x, figsize=(4,3), **kwargs):
    "Display HW or CHW format image `x`"
    plt.figure(figsize=figsize)
    plt.axis('off')
    if len(x.shape)==3: x = x.permute(1,2,0)  # CHW -> HWC
    plt.imshow(x.cpu(), **kwargs)

cuda_begin = r'''
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CUDA_ERR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
'''

def load_cuda(cuda_source, cpp_source, funcs, opt=False, verbose=False, name=None):
  if name is None: name = funcs[0]
  opt_flags = "-O3 -Xptxas -O3 -Xcompiler -O3" if opt else "-O0 -Xptxas -O0 -Xcompiler -O0"
  return load_inline(
      name=name,
      cuda_sources=[cuda_source],
      cpp_sources=[cpp_source],
      functions=funcs,
      extra_cuda_cflags=[opt_flags] + ['--ptxas-options=-v'], # '--expt-relaxed-constexpr'
      verbose=verbose,
      # extra_cuda_cflags=['--expt-relaxed-constexpr']
      build_directory='./load_inline_cuda')


def check_tensors_gpu_ready(*tensors):
  for t in tensors:
    assert t.is_contiguous, "A tensor is not contiguous"
    if not os.environ.get('TRITON_INTERPRET') == '1':
      assert t.is_cuda, "A tensor is not on cuda"


def test_pid_conds(conds, pid_0=[0], pid_1=[0], pid_2=[0]):
  '''Test if condition on pids are fulfilled
    E.g.:
        '=0'  checks that pid_0 == 0
        ',>1' checks that pid_1 > 1
        '>1,=0' checks that pid_0 > 1 and pid_1 == 0
    '''
  pids = pid_0[0], pid_1[0], pid_2[0]
  conds = conds.replace(' ', '').split(',')
  for i, (cond, pid) in enumerate(zip(conds, pids)):
    if cond == '':
      continue
    op, threshold = cond[0], int(cond[1:])
    if op not in ['<', '>', '>=', '<=', '=', '!=']:
      raise ValueError(
          f"Rules may only use these ops: '<','>','>=','<=','=', '!='. Invalid rule: '{condition}'."
      )
    op = '==' if op == '=' else op
    if not eval(f'{pid} {op} {threshold}'):
      return False
  return True


assert test_pid_conds('')
assert test_pid_conds('>0', [1], [1])
assert not test_pid_conds('>0', [0], [1])
assert test_pid_conds('=0,=1', [0], [1], [0])


def breakpoint_if(conds, pid_0=[0], pid_1=[0], pid_2=[0]):
  '''Stop kernel, if any condition of pids is fulfilled'''
  if test_pid_conds(conds, pid_0, pid_1, pid_2):
    set_trace()


def print_if(txt, conds, pid_0=[0], pid_1=[0], pid_2=[0]):
  '''Print txt, if any condition of pids is fulfilled'''
  if test_pid_conds(conds, pid_0, pid_1, pid_2):
    print(txt)


def cdiv(a, b):
  return (a + b - 1) // b


assert cdiv(10, 2) == 5
assert cdiv(10, 3) == 4


@triton.jit
def get_1d_offset(size, n_prev_chunks):
  return n_prev_chunks * size + tl.arange(0, size)


@triton.jit
def get_2d_offset(offs_0, offs_1, stride_0, stride_1=1):
  return tl.expand_dims(offs_0, 1) * stride_0 + tl.expand_dims(offs_1,
                                                               0) * stride_1


@triton.jit
def get_1d_mask(offs, max):
  return offs < max


@triton.jit
def get_2d_mask(offs_0, offs_1, max_0, max_1):
  return (tl.expand_dims(offs_0, 1) < max_0) & (tl.expand_dims(offs_1, 0)
                                                < max_1)
