import os

os.environ["TRITON_INTERPRET"] = "1"

import triton
import triton.language as tl
import torch

from triton_util import cdiv, breakpoint_if, print_if, check_tensors_gpu_ready, get_1d_offset, get_2d_offset, \
  get_1d_mask, get_2d_mask


# # This is a normal python function, which launches the triton kernels
def copy(x, bs, kernel_fn):
  z = torch.zeros_like(x)
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
  print_if(f'pid = {pid} | offs = {offs}, mask = {mask}, x = {x}', '')


x = torch.tensor([1, 2, 3, 4, 5, 6])
y = torch.tensor([0, 1, 0, 1, 0, 1])
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
