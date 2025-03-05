import os

# os.environ["TRITON_INTERPRET"] = "1"

import numpy as np

np.seterr(all='warn')

import triton
import triton.language as tl
import torch

from triton_util import cdiv, breakpoint_if, print_if, check_tensors_gpu_ready, get_1d_offset, get_2d_offset, \
  get_1d_mask, get_2d_mask


# # This is a normal python function, which launches the triton kernels
def copy(x, bs, kernel_fn):
  z = torch.zeros_like(x).cuda()
  check_tensors_gpu_ready(x, z)
  n = x.numel()
  n_blocks = cdiv(n, bs)
  grid = (n_blocks,)  # how many blocks do we have? can be 1d/2d/3d-tuple or function returning 1d/2d/3d-tuple

  # launch grid!
  # - kernel_fn is the triton kernel, which we write below
  # - grid is the grid we constructed above
  # - x,z,n,bs are paramters that are passed into each kernel function
  kernel_fn[grid](x, z, n, bs)

  return z


# # This is the triton kernel:

# The triton.jit decorator takes a python function and turns it into a triton kernel, which is run on the GPU.
# Inside this function only a subset of all python ops are allowed.
# E.g., when NOT simulating, we can't print or use breakpoints, as these don't exist on the GPU.
@triton.jit
# When we pass torch tensors, they are automatically converted into a pointer to their first value
# E.g., above we passed x, but here we receive x_ptr
def _copy(x_ptr, z_ptr, n, bs: tl.constexpr):
  pid = tl.program_id(0)
  offs = pid * bs + tl.arange(0, bs)
  mask = offs < n
  x = tl.load(x_ptr + offs, mask)
  tl.store(z_ptr + offs, x, mask)
  # print_if(f'pid = {pid} | offs = {offs}, mask = {mask}, x = {x}', '')


x = torch.tensor([1, 2, 3, 4, 5, 6]).cuda()
y = torch.tensor([0, 1, 0, 1, 0, 1]).cuda()
z = copy(x, bs=2, kernel_fn=_copy)

import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from pathlib import Path
import torchvision as tv
import torchvision.transforms.functional as tvf
from torchvision import io


def show_img(x, figsize=(4, 3), **kwargs):
  plt.figure(figsize=figsize)
  plt.axis('off')
  if len(x.shape) == 3: x = x.permute(1, 2, 0)  # CHW -> HWC
  plt.imshow(x.cpu(), **kwargs)


@triton.jit
def rgb2grey_k(x_ptr, out_ptr, h, w, bs0: tl.constexpr, bs1: tl.constexpr):
  pid_0 = tl.program_id(0)
  pid_1 = tl.program_id(1)

  offs_0 = pid_0 * bs0 + tl.arange(0, bs0)  # 1d vector
  offs_1 = pid_1 * bs1 + tl.arange(0, bs1)  # 1d vector

  # Weirdness: None-slicing currently doesn't work when simulating on cpu. Use tl.expand_dim instead.
  # offs = w * tl.expand_dims(offs_0, 1) + tl.expand_dims(offs_1, 0)
  offs = w * offs_0[:, None] + offs_1[None, :]  # 2d matrix! - we multiply first offset by width, see image above

  mask_0 = offs_0 < h  # 1d vector
  mask_1 = offs_1 < w  # 1d vector

  # mask = tl.expand_dims(mask_0, 1) & tl.expand_dims(mask_1, 0)
  mask = mask_0[:, None] & mask_1[None,
                           :]  # 2d matrix! - data musn't go out of bounds along either axis, therefore `logical and` of the individual masks

  r = tl.load(x_ptr + 0 * h * w + offs, mask=mask)
  g = tl.load(x_ptr + 1 * h * w + offs, mask=mask)
  b = tl.load(x_ptr + 2 * h * w + offs, mask=mask)

  # Weirdness: multiplying float with uint vectors fails when simulating on cpu
  out = 0.2989 * r + 0.5870 * g + 0.1140 * b  # don't worry why it's these 3 numbers we're multiplying with

  tl.store(out_ptr + offs, out, mask=mask)


def rgb2grey(x, bs):
  c, h, w = x.shape
  out = torch.empty((h, w), dtype=x.dtype, device=x.device)

  # grid can be a function returning a 1d/2d/3d-tuple
  # (having a grid function is not more useful than a grid tuple in this case, but will be below when benchmarking & auto-tuning)
  grid = lambda meta: (cdiv(h, meta['bs0']), cdiv(w, meta['bs1']))

  rgb2grey_k[grid](x, out, h, w, bs0=bs[0], bs1=bs[1])  # all kwargs are passed into grid function
  return out.view(h, w)


# url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/1600px-Cute_dog.jpg?20140729055059'
# path_img = Path('puppy.jpg')
# if not path_img.exists(): urlretrieve(url, path_img)
# img = io.read_image('puppy.jpg')
# print(img.shape)
# img[:2, :3, :4]
#
# img = tvf.resize(img, 150, antialias=True)
# ch, h, w = img.shape
# ch, h, w, h * w
#
# show_img(img)
# grey_img = rgb2grey(img.to('cuda'), bs=(32, 32)).to('cpu')
# show_img(grey_img, cmap='gray')

from functools import partial


def matmul(a, b, matmul_k_fn, bs=16, group_sz=None):
  assert a.shape[1] == b.shape[0], "matrix dims not compatible for matmul"
  check_tensors_gpu_ready(a, b)
  (m, k), (_, n) = a.shape, b.shape
  c = torch.empty((m, n), device=a.device, dtype=torch.float16)
  grid = lambda meta: (triton.cdiv(m, meta['bm']), triton.cdiv(n, meta['bn']))
  group_sz = {} if group_sz is None else {
    "group_sz": group_sz}  # not used in naive_matmul, but will be in grouped_matmul further below
  matmul_k_fn[grid](
    a, b, c,
    m, n, k,
    a.stride(0), a.stride(1),
    b.stride(0), b.stride(1),
    c.stride(0), c.stride(1),
    bm=bs, bn=bs, bk=bs,  # Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
    **group_sz
  )
  return c


@triton.jit
def naive_matmul_k(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr
):
  pid_m, pid_n = tl.program_id(0), tl.program_id(1)
  # chunks along m/n/k dimensions
  rm = get_1d_offset(size=bm, n_prev_chunks=pid_m)
  rn = get_1d_offset(size=bn, n_prev_chunks=pid_n)
  rk = get_1d_offset(size=bk, n_prev_chunks=0)
  # relevant offsets of a, b
  offs_a = a_ptr + get_2d_offset(rm, rk, stride_am, stride_ak)
  offs_b = b_ptr + get_2d_offset(rk, rn, stride_bk, stride_bn)
  # initialize and iteratively update accumulator
  acc = tl.zeros((bm, bn), dtype=tl.float32)
  for _ in range(0, k, bk):
    # todo umer: don't we need mask when loading a & b?
    a = tl.load(offs_a)
    b = tl.load(offs_b)
    acc += tl.dot(a, b,
                  allow_tf32=False)  # matmul in block ; Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
    # increase offets, so next iteration loads next chunks
    offs_a += bk * stride_ak
    offs_b += bk * stride_bk
  c = c_ptr + get_2d_offset(rm, rn, stride_cm, stride_cn)
  mask = get_2d_mask(rm, rn, m, n)
  tl.store(c, acc, mask=mask)


naive_matmul = partial(matmul, matmul_k_fn=naive_matmul_k)

a = torch.ones((3, 4), dtype=torch.float32, device='cuda')
b = torch.ones((4, 5), dtype=torch.float32, device='cuda')

naive_matmul(a, b)

torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
triton_output = naive_matmul(a, b)
torch_output = torch.matmul(a, b)
if torch.allclose(triton_output, torch_output, atol=5e-2, rtol=0):
  print("✅ Triton and Torch match")
else:
  print("❌ Triton and Torch differ")


@triton.jit
def swizzle_k(x_ptr, z_ptr, group_sz: tl.constexpr):
  pid_m, pid_n = tl.program_id(0), tl.program_id(1)
  num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)

  pid_m_, pid_n_ = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n,
                                group_sz)  # Weirdness: tl.swizzle2d doesn't work when simulating on CPU

  offs_m = get_1d_offset(1, n_prev_chunks=pid_m)
  offs_n = get_1d_offset(1, n_prev_chunks=pid_n)

  offs = get_2d_offset(offs_m, offs_n, stride_0=num_pid_n)
  mask = get_2d_mask(offs_m, offs_n, max_0=num_pid_m, max_1=num_pid_n)

  offs_sw_m = get_1d_offset(1, n_prev_chunks=pid_m_)
  offs_sw_n = get_1d_offset(1, n_prev_chunks=pid_n_)

  offs_sw = get_2d_offset(offs_sw_m, offs_sw_n, stride_0=num_pid_n)
  mask_sw = get_2d_mask(offs_sw_m, offs_sw_n, max_0=num_pid_m, max_1=num_pid_n)

  x = tl.load(x_ptr + offs, mask=mask)
  tl.store(z_ptr + offs_sw, x, mask=mask_sw)


blocks_m, blocks_n = 5, 4
x = torch.arange(blocks_m * blocks_n, device='cuda').view(blocks_m, blocks_n)
print(x)
z = -torch.ones_like(x)
swizzle_k[(blocks_m, blocks_n)](x, z, group_sz=3)
print(z)


@triton.autotune(
  # Choices of configs to auto-tune over
  configs=[
    triton.Config({'bm': 128, 'bn': 256, 'bk': 64, 'group_sz': 8}, num_stages=3, num_warps=8),
    triton.Config({'bm': 64, 'bn': 256, 'bk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
    triton.Config({'bm': 128, 'bn': 128, 'bk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
    triton.Config({'bm': 128, 'bn': 64, 'bk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
    triton.Config({'bm': 64, 'bn': 128, 'bk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
    triton.Config({'bm': 128, 'bn': 32, 'bk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
    triton.Config({'bm': 64, 'bn': 32, 'bk': 32, 'group_sz': 8}, num_stages=5, num_warps=2),
    triton.Config({'bm': 32, 'bn': 64, 'bk': 32, 'group_sz': 8}, num_stages=5, num_warps=2),
  ],
  # Definition of problem size. If it changes, a new auto-tune is run for the new problem size.
  key=['m', 'n', 'k'],
)
@triton.jit
def grouped_autotuned_matmul_k(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr, group_sz: tl.constexpr
):
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  num_pid_m = tl.num_programs(0)
  num_pid_n = tl.num_programs(1)
  # determine location of block in grouped ordering
  pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n,
                              group_sz)  # Weirdness: tl.swizzle2d doesn't work when simulating on CPU
  # chunks along m/n/k dimensions
  rm = get_1d_offset(size=bm, n_prev_chunks=pid_m)
  rn = get_1d_offset(size=bn, n_prev_chunks=pid_n)
  rk = get_1d_offset(size=bk, n_prev_chunks=0)
  # relevant offsets of a, b
  offs_a = a_ptr + get_2d_offset(rm, rk, stride_am, stride_ak)
  offs_b = b_ptr + get_2d_offset(rk, rn, stride_bk, stride_bn)
  # initialize and iteratively update accumulator
  acc = tl.zeros((bm, bn), dtype=tl.float32)
  for _ in range(0, k, bk):
    # todo umer: don't we need mask when loading a & b?
    a = tl.load(offs_a)
    b = tl.load(offs_b)
    acc += tl.dot(a, b,
                  allow_tf32=False)  # block level matrix multiplication ; Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
    # increase offets, so next iteration loads next chunks
    offs_a += bk * stride_ak
    offs_b += bk * stride_bk
  c = c_ptr + get_2d_offset(rm, rn, stride_cm, stride_cn)
  mask = get_2d_mask(rm, rn, m, n)
  tl.store(c, acc, mask=mask)


@triton.jit
def grouped_matmul_k(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr, group_sz: tl.constexpr
):
  pid_m, pid_n = tl.program_id(0), tl.program_id(1)
  num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)
  # determine location of block in grouped ordering - swizzle!
  pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n,
                              group_sz)  # Weirdness: tl.swizzle2d doesn't work when simulating on CPU
  # chunks along m/n/k dimensions
  rm = get_1d_offset(size=bm, n_prev_chunks=pid_m)
  rn = get_1d_offset(size=bn, n_prev_chunks=pid_n)
  rk = get_1d_offset(size=bk, n_prev_chunks=0)
  # relevant offsets of a, b
  offs_a = a_ptr + get_2d_offset(rm, rk, stride_am, stride_ak)
  offs_b = b_ptr + get_2d_offset(rk, rn, stride_bk, stride_bn)
  # initialize and iteratively update accumulator
  acc = tl.zeros((bm, bn), dtype=tl.float32)
  for _ in range(0, k, bk):
    # todo umer: don't we need mask when loading a & b?
    a = tl.load(offs_a)
    b = tl.load(offs_b)
    acc += tl.dot(a, b,
                  allow_tf32=False)  # block level matrix multiplication ; Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
    # increase offets, so next iteration loads next chunks
    offs_a += bk * stride_ak
    offs_b += bk * stride_bk
  c = c_ptr + get_2d_offset(rm, rn, stride_cm, stride_cn)
  mask = get_2d_mask(rm, rn, m, n)
  tl.store(c, acc, mask=mask)


grouped_matmul = partial(matmul, matmul_k_fn=grouped_matmul_k)

grouped_matmul(a, b, group_sz=4)

triton_output = grouped_matmul(a, b, group_sz=32)
torch_output = torch.matmul(a, b)
if torch.allclose(triton_output, torch_output, atol=5e-2, rtol=0):
  print("✅ Triton and Torch match")
else:
  print("❌ Triton and Torch differ")


@triton.testing.perf_report(
  triton.testing.Benchmark(
    x_names=['block_size'], x_vals=[2 ** i for i in range(4, 7, 1)], x_log=True,
    # > 7 makes shared memory requirement exceeds limit 232448
    line_arg='provider', line_vals=['naive', 'grouped', 'torch'], line_names=['Naive', 'Grouped', 'Torch'],
    styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
    ylabel='GB/s', plot_name='matmul-performance-block-size', args={}
  ))
def benchmark_block_size(block_size, provider):
  sz = 512
  a = torch.rand((sz, sz), device='cuda', dtype=torch.float32)
  b = torch.rand((sz, sz), device='cuda', dtype=torch.float32)
  quantiles = [0.5, 0.2, 0.8]
  if provider == 'naive':   ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_matmul(a, b, bs=block_size),
                                                                         quantiles=quantiles)
  if provider == 'grouped': ms, min_ms, max_ms = triton.testing.do_bench(
    lambda: grouped_matmul(a, b, bs=block_size, group_sz=8), quantiles=quantiles)
  if provider == 'torch':   ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b),
                                                                         quantiles=quantiles)
  gbps = lambda ms: 12 * sz / ms * 1e-6
  return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark_block_size.run(print_data=True, show_plots=True, save_path='.')


@triton.autotune(
  # Choices of configs to auto-tune over
  configs=[
    triton.Config({'bm': 128, 'bn': 256, 'bk': 64, 'group_sz': 8}, num_stages=3, num_warps=8),
    triton.Config({'bm': 64, 'bn': 256, 'bk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
    triton.Config({'bm': 128, 'bn': 128, 'bk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
    triton.Config({'bm': 128, 'bn': 64, 'bk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
    triton.Config({'bm': 64, 'bn': 128, 'bk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
    triton.Config({'bm': 128, 'bn': 32, 'bk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
    triton.Config({'bm': 64, 'bn': 32, 'bk': 32, 'group_sz': 8}, num_stages=5, num_warps=2),
    triton.Config({'bm': 32, 'bn': 64, 'bk': 32, 'group_sz': 8}, num_stages=5, num_warps=2),
    triton.Config({'bm': 32, 'bn': 32, 'bk': 32, 'group_sz': 8}, num_stages=5, num_warps=2),
    triton.Config({'bm': 32, 'bn': 32, 'bk': 16, 'group_sz': 8}, num_stages=5, num_warps=2),
    triton.Config({'bm': 16, 'bn': 16, 'bk': 16, 'group_sz': 8}, num_stages=5, num_warps=2),
  ],
  # Definition of problem size. If it changes, a new auto-tune is run for the new problem size.
  key=['m', 'n', 'k'],
)
@triton.jit
def grouped_autotuned_matmul_k(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr, group_sz: tl.constexpr
):
  grouped_matmul_k(a_ptr, b_ptr, c_ptr, m, n, k, stride_am, stride_ak, stride_bk, stride_bn,
                   stride_cm, stride_cn,
                   bm, bn, bk, group_sz)


def grouped_autotuned_matmul(a, b):
  matmul_k_fn = grouped_autotuned_matmul_k

  assert a.shape[1] == b.shape[0], "matrix dims not compatible for matmul"
  check_tensors_gpu_ready(a, b)
  (m, k), (_, n) = a.shape, b.shape
  c = torch.empty((m, n), device=a.device, dtype=torch.float16)
  grid = lambda meta: (triton.cdiv(m, meta['bm']), triton.cdiv(n, meta['bn']))
  matmul_k_fn[grid](
    a, b, c,
    m, n, k,
    a.stride(0), a.stride(1),
    b.stride(0), b.stride(1),
    c.stride(0), c.stride(1),
    # bm=bs, bn=bs, bk=bs, <- will be autotuned
    # **group_sz <- will be autotuned
  )
  return c


@triton.testing.perf_report(
  triton.testing.Benchmark(
    x_names=['square_matrix_size'],  # Argument names to use as an x-axis for the plot.
    x_vals=[2 ** i for i in range(5, 12, 1)],  # Different possible values for `x_name`.
    x_log=True,  # x axis is logarithmic.
    line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
    line_vals=['naive', 'grouped', 'grouped-autotuned', 'torch'],  # Possible values for `line_arg`.
    line_names=['Naive', 'Grouped', 'Grouped & Auto-Tuned', 'Torch'],  # Label name for the lines.
    styles=[('blue', '-'), ('green', '-'), ('green', '--'), ('orange', '-')],  # Line styles.
    ylabel='GB/s',  # Label name for the y-axis.
    plot_name='matmul-performance-matrix-size',  # Name for the plot. Used also as a file name for saving the plot.
    args={},  # Values for function arguments not in `x_names` and `y_name`.
  ))
def benchmark_matrix_size(square_matrix_size, provider):
  sz = square_matrix_size
  a = torch.rand((sz, sz), device='cuda', dtype=torch.float32)
  b = torch.rand((sz, sz), device='cuda', dtype=torch.float32)
  quantiles = [0.5, 0.2, 0.8]
  if provider == 'naive':   ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_matmul(a, b),
                                                                         quantiles=quantiles)
  if provider == 'grouped': ms, min_ms, max_ms = triton.testing.do_bench(lambda: grouped_matmul(a, b, group_sz=8),
                                                                         quantiles=quantiles)
  if provider == 'grouped-autotuned': ms, min_ms, max_ms = triton.testing.do_bench(
    lambda: grouped_autotuned_matmul(a, b), quantiles=quantiles)

  if provider == 'torch':   ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b),
                                                                         quantiles=quantiles)
  gbps = lambda ms: 12 * sz / ms * 1e-6
  return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark_matrix_size.run(print_data=True, show_plots=True, save_path='.')
