# matmul-performance-matrix-size:
# square_matrix_size      Naive    Grouped  Grouped & Auto-Tuned       Torch  Numpy-Broadcast  Cuda-Naive  Cuda-Shared
# 0                32.0   2.053476   2.075676              1.959184    1.641026         0.016209    1.200000      1.443609
# 1                64.0   7.603961   7.245283              7.349282    5.840304         0.031616    3.746341      4.938907
# 2               128.0  25.077551  24.674698             26.369099   18.231454         0.062338   10.538593     15.095823
# 3               256.0  60.383290  60.681482             72.495577   61.904283         0.123212    9.850101     38.641509
# 4               512.0  59.290710  59.254975            140.034190  127.007750         0.246561    6.216461     41.920684
# 5              1024.0  35.155656  35.146227            113.090598  130.074762         0.451891    3.345024     26.210031
# 6              2048.0  18.051198  18.061873             68.486634   71.406181         0.823596    1.689025     13.825796

# Analysis:
# - shared memory: 128k/2k -> 64
# - threads: 1536/256 -> 6


import numpy as np
import triton
import triton.language as tl
import torch

np.set_printoptions(precision=2, linewidth=140)
torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
np.seterr(all="warn")

from triton_utils import (cdiv, breakpoint_if, print_if,
                          check_tensors_gpu_ready, get_1d_offset, get_2d_offset,
                          get_1d_mask, get_2d_mask, load_cuda)
from functools import partial


def matmul(a, b, matmul_k_fn, bs=16, group_sz=None):
  assert a.shape[1] == b.shape[0], "matrix dims not compatible for matmul"
  check_tensors_gpu_ready(a, b)
  (m, k), (_, n) = a.shape, b.shape
  c = torch.empty((m, n), device=a.device, dtype=a.dtype)
  grid = lambda meta: (triton.cdiv(m, meta["bm"]), triton.cdiv(n, meta["bn"]))
  group_sz = ({} if group_sz is None else {
      "group_sz": group_sz
  })  # not used in naive_matmul, but will be in grouped_matmul further below
  matmul_k_fn[grid](
      a,
      b,
      c,
      m,
      n,
      k,
      a.stride(0),
      a.stride(1),
      b.stride(0),
      b.stride(1),
      c.stride(0),
      c.stride(1),
      bm=bs,
      bn=bs,
      bk=
      bs,  # Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
      **group_sz,
  )
  return c


@triton.jit
def naive_matmul_k(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    bm: tl.constexpr,
    bn: tl.constexpr,
    bk: tl.constexpr,
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
    acc += tl.dot(
        a, b, allow_tf32=False
    )  # matmul in block ; Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
    # increase offets, so next iteration loads next chunks
    offs_a += bk * stride_ak
    offs_b += bk * stride_bk
  c = c_ptr + get_2d_offset(rm, rn, stride_cm, stride_cn)
  mask = get_2d_mask(rm, rn, m, n)
  tl.store(c, acc, mask=mask)


naive_matmul = partial(matmul, matmul_k_fn=naive_matmul_k)

torch.manual_seed(0)
a = torch.randn((512, 512), device="cuda", dtype=torch.float32)
b = torch.randn((512, 512), device="cuda", dtype=torch.float32)
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

  pid_m_, pid_n_ = tl.swizzle2d(
      pid_m, pid_n, num_pid_m, num_pid_n,
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
x = torch.arange(blocks_m * blocks_n, device="cuda").view(blocks_m, blocks_n)
print(x)
z = -torch.ones_like(x)
swizzle_k[(blocks_m, blocks_n)](x, z, group_sz=3)
print(z)


@triton.autotune(
    # Choices of configs to auto-tune over
    configs=[
        triton.Config({
            "bm": 128,
            "bn": 256,
            "bk": 64,
            "group_sz": 8
        },
                      num_stages=3,
                      num_warps=8),
        triton.Config({
            "bm": 64,
            "bn": 256,
            "bk": 32,
            "group_sz": 8
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "bm": 128,
            "bn": 128,
            "bk": 32,
            "group_sz": 8
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "bm": 128,
            "bn": 64,
            "bk": 32,
            "group_sz": 8
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "bm": 64,
            "bn": 128,
            "bk": 32,
            "group_sz": 8
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "bm": 128,
            "bn": 32,
            "bk": 32,
            "group_sz": 8
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "bm": 64,
            "bn": 32,
            "bk": 32,
            "group_sz": 8
        },
                      num_stages=5,
                      num_warps=2),
        triton.Config({
            "bm": 32,
            "bn": 64,
            "bk": 32,
            "group_sz": 8
        },
                      num_stages=5,
                      num_warps=2),
    ],
    # Definition of problem size. If it changes, a new auto-tune is run for the new problem size.
    key=["m", "n", "k"],
)
@triton.jit
def grouped_autotuned_matmul_k(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    bm: tl.constexpr,
    bn: tl.constexpr,
    bk: tl.constexpr,
    group_sz: tl.constexpr,
):
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  num_pid_m = tl.num_programs(0)
  num_pid_n = tl.num_programs(1)
  # determine location of block in grouped ordering
  pid_m, pid_n = tl.swizzle2d(
      pid_m, pid_n, num_pid_m, num_pid_n,
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
    acc += tl.dot(
        a, b, allow_tf32=False
    )  # block level matrix multiplication ; Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
    # increase offets, so next iteration loads next chunks
    offs_a += bk * stride_ak
    offs_b += bk * stride_bk
  c = c_ptr + get_2d_offset(rm, rn, stride_cm, stride_cn)
  mask = get_2d_mask(rm, rn, m, n)
  tl.store(c, acc, mask=mask)


@triton.jit
def grouped_matmul_k(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    bm: tl.constexpr,
    bn: tl.constexpr,
    bk: tl.constexpr,
    group_sz: tl.constexpr,
):
  pid_m, pid_n = tl.program_id(0), tl.program_id(1)
  num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)
  # determine location of block in grouped ordering - swizzle!
  pid_m, pid_n = tl.swizzle2d(
      pid_m, pid_n, num_pid_m, num_pid_n,
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
    acc += tl.dot(
        a, b, allow_tf32=False
    )  # block level matrix multiplication ; Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
    # increase offets, so next iteration loads next chunks
    offs_a += bk * stride_ak
    offs_b += bk * stride_bk
  c = c_ptr + get_2d_offset(rm, rn, stride_cm, stride_cn)
  mask = get_2d_mask(rm, rn, m, n)
  tl.store(c, acc, mask=mask)


grouped_matmul = partial(matmul, matmul_k_fn=grouped_matmul_k)

triton_output = grouped_matmul(a, b, group_sz=32)
torch_output = torch.matmul(a, b)
if torch.allclose(triton_output, torch_output, atol=5e-2, rtol=0):
  print("✅ Triton and Torch match")
else:
  print("❌ Triton and Torch differ")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["block_size"],
        x_vals=[2**i for i in range(4, 7, 1)],
        x_log=True,
        # > 7 makes shared memory requirement exceeds limit 232448
        line_arg="provider",
        line_vals=["naive", "grouped", "torch"],
        line_names=["Naive", "Grouped", "Torch"],
        styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
        ylabel="GB/s",
        plot_name="matmul-performance-block-size",
        args={},
    ))
def benchmark_block_size(block_size, provider):
  sz = 512
  a = torch.rand((sz, sz), device="cuda", dtype=torch.float32)
  b = torch.rand((sz, sz), device="cuda", dtype=torch.float32)
  quantiles = [0.5, 0.2, 0.8]
  if provider == "naive":
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: naive_matmul(a, b, bs=block_size), quantiles=quantiles)
  if provider == "grouped":
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: grouped_matmul(a, b, bs=block_size, group_sz=8),
        quantiles=quantiles)
  if provider == "torch":
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b),
                                                 quantiles=quantiles)
  gbps = lambda ms: 12 * sz / ms * 1e-6
  return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark_block_size.run(print_data=True, show_plots=True, save_path=".")


@triton.autotune(
    # Choices of configs to auto-tune over
    configs=[
        triton.Config({
            "bm": 128,
            "bn": 256,
            "bk": 64,
            "group_sz": 8
        },
                      num_stages=3,
                      num_warps=8),
        triton.Config({
            "bm": 64,
            "bn": 256,
            "bk": 32,
            "group_sz": 8
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "bm": 128,
            "bn": 128,
            "bk": 32,
            "group_sz": 8
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "bm": 128,
            "bn": 64,
            "bk": 32,
            "group_sz": 8
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "bm": 64,
            "bn": 128,
            "bk": 32,
            "group_sz": 8
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "bm": 128,
            "bn": 32,
            "bk": 32,
            "group_sz": 8
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "bm": 64,
            "bn": 32,
            "bk": 32,
            "group_sz": 8
        },
                      num_stages=5,
                      num_warps=2),
        triton.Config({
            "bm": 32,
            "bn": 64,
            "bk": 32,
            "group_sz": 8
        },
                      num_stages=5,
                      num_warps=2),
        triton.Config({
            "bm": 32,
            "bn": 32,
            "bk": 32,
            "group_sz": 8
        },
                      num_stages=5,
                      num_warps=2),
        triton.Config({
            "bm": 32,
            "bn": 32,
            "bk": 16,
            "group_sz": 8
        },
                      num_stages=5,
                      num_warps=2),
        triton.Config({
            "bm": 16,
            "bn": 16,
            "bk": 16,
            "group_sz": 8
        },
                      num_stages=5,
                      num_warps=2),
    ],
    # Definition of problem size. If it changes, a new auto-tune is run for the new problem size.
    key=["m", "n", "k"],
)
@triton.jit
def grouped_autotuned_matmul_k(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    bm: tl.constexpr,
    bn: tl.constexpr,
    bk: tl.constexpr,
    group_sz: tl.constexpr,
):
  grouped_matmul_k(
      a_ptr,
      b_ptr,
      c_ptr,
      m,
      n,
      k,
      stride_am,
      stride_ak,
      stride_bk,
      stride_bn,
      stride_cm,
      stride_cn,
      bm,
      bn,
      bk,
      group_sz,
  )


def grouped_autotuned_matmul(a, b):
  matmul_k_fn = grouped_autotuned_matmul_k

  assert a.shape[1] == b.shape[0], "matrix dims not compatible for matmul"
  check_tensors_gpu_ready(a, b)
  (m, k), (_, n) = a.shape, b.shape
  c = torch.empty((m, n), device=a.device, dtype=torch.float16)
  grid = lambda meta: (triton.cdiv(m, meta["bm"]), triton.cdiv(n, meta["bn"]))
  matmul_k_fn[grid](
      a,
      b,
      c,
      m,
      n,
      k,
      a.stride(0),
      a.stride(1),
      b.stride(0),
      b.stride(1),
      c.stride(0),
      c.stride(1),
      # bm=bs, bn=bs, bk=bs, <- will be autotuned
      # **group_sz <- will be autotuned
  )
  return c


def numpy_matmul(a, b):
  (ar, ac), (br, bc) = a.shape, b.shape
  c = torch.zeros(ar, bc, device=a.device, dtype=a.dtype)
  for i in range(ar):
    c[i] = (a[i, :, None] * b).sum(dim=0)
  return c


cuda_begin = r'''
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
'''

cuda_src = cuda_begin + r'''

// k对应的维度做tiling（phases）

constexpr int TILE_SIZE = 16; // TILE_SIZE设为32会导致性能严重劣化

__global__ void sharedMatMult( float *a, float *b, float *c, int M, int K, int N, int BLOCK_SIZE){
    C10_LAUNCH_BOUNDS_0;
    __shared__ float aTile[TILE_SIZE][TILE_SIZE];  
    __shared__ float bTile[TILE_SIZE][TILE_SIZE];
    assert(BLOCK_SIZE <= TILE_SIZE);
  
    // note how threadIdx.x is the fastest moving bit --> coalesced memory access, improve 7x at most
    int ir = threadIdx.y;
    int ic = threadIdx.x; 

    int row = blockIdx.y * blockDim.y + ir;
    int col = blockIdx.x * blockDim.x + ic;
    float sum = 0.0f;

    if (row < M && col < N) {
        for(int phase = 0; phase < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; phase++){
            int a_col = ic + phase * BLOCK_SIZE;
            int b_row = ir + phase * BLOCK_SIZE;
            aTile[ir][ic] = ((a_col < K) ? a[row * K + a_col] : 0.0f);
            bTile[ir][ic] = ((b_row < K) ? b[b_row * N + col] : 0.0f);
            __syncthreads();

            for(int i = 0; i < BLOCK_SIZE; i++){
                sum += aTile[ir][i] * bTile[i][ic];
            }
            __syncthreads();
        }
        c[row * N + col] = sum;
    }
}


__global__ void matrixMulGPU( float * a, float * b, float * c, int M, int K, int N)
{
  float val = 0;

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N)
  {
    for ( int k = 0; k < K; ++k )
      val += a[row * K + k] * b[k * N + col];
    c[row * N + col] = val;
  }
}


void matrixMulCPU( float * a, float * b, float * c, int M, int K, int N)
{
  for( int row = 0; row < M; ++row )
    for( int col = 0; col < N; ++col )
    {
      float val = 0;
      for ( int k = 0; k < K; ++k )
        val += a[row * K + k] * b[k * N + col];
      c[row * N + col] = val;
    }
}


torch::Tensor naive_matmul(torch::Tensor m, torch::Tensor n) {
    CHECK_INPUT(m); CHECK_INPUT(n);
    int h = m.size(0);
    int w = n.size(1);
    int k = m.size(1);
    TORCH_CHECK(k==n.size(0), "Size mismatch!");
    auto output = torch::zeros({h, w}, m.options());

    dim3 tpb(16, 16);
    dim3 blocks(cdiv(h, tpb.y), cdiv(w, tpb.x));
    matrixMulGPU<<<blocks, tpb>>>(
        m.data_ptr<float>(), n.data_ptr<float>(), output.data_ptr<float>(), h, k, w);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor group_matmul(torch::Tensor m, torch::Tensor n) {
    CHECK_INPUT(m); CHECK_INPUT(n);
    int h = m.size(0);
    int w = n.size(1);
    int k = m.size(1);
    TORCH_CHECK(k==n.size(0), "Size mismatch!");
    auto output = torch::zeros({h, w}, m.options());

    dim3 tpb(TILE_SIZE, TILE_SIZE);
    dim3 blocks(cdiv(h, tpb.y), cdiv(w, tpb.x));
    sharedMatMult<<<blocks, tpb>>>(
        m.data_ptr<float>(), n.data_ptr<float>(), output.data_ptr<float>(), h, k, w, tpb.x);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
'''

cpp_src = '''
torch::Tensor naive_matmul(torch::Tensor m, torch::Tensor n);
torch::Tensor group_matmul(torch::Tensor m, torch::Tensor n);
'''

mm_module = load_cuda("mm_load_inline",
                      cuda_src,
                      cpp_src, ['naive_matmul', 'group_matmul'],
                      verbose=True)  # 'group_matmul'

cuda_output_1 = mm_module.naive_matmul(a, b)
if torch.allclose(cuda_output_1, torch_output, atol=5e-2, rtol=0):
  print("✅ Cuda_Naive_Matmul and Torch match")
else:
  print("❌ Cuda_Naive_Matmul and Torch differ")

cuda_output_2 = mm_module.group_matmul(a, b)
if torch.allclose(cuda_output_2, torch_output, atol=5e-2, rtol=0):
  print("✅ Cuda_Group_Matmul and Torch match")
else:
  print("❌ Cuda_Group_Matmul and Torch differ")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["square_matrix_size"
                ],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(5, 12, 1)
               ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg=
        "provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=[
            "naive", "grouped", "grouped-autotuned", "torch", "numpy-broadcast",
            "cuda-naive", "cuda-shared"
        ],  # Possible values for `line_arg`.
        line_names=[
            "Naive", "Grouped", "Grouped & Auto-Tuned", "Torch",
            "Numpy-Broadcast", "Cuda-Naive", "Cuda-Shared"
        ],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-"), ("green", "--"), ("orange", "-"),
                ("red", "-"), ("purple", "-"),
                ("purple", "--")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name=
        "matmul-performance-matrix-size",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark_matrix_size(square_matrix_size, provider):
  sz = square_matrix_size
  a = torch.rand((sz, sz), device="cuda", dtype=torch.float32)
  b = torch.rand((sz, sz), device="cuda", dtype=torch.float32)
  quantiles = [0.5, 0.2, 0.8]
  if provider == "naive":
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_matmul(a, b),
                                                 quantiles=quantiles)
  elif provider == "grouped":
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: grouped_matmul(a, b, group_sz=8), quantiles=quantiles)
  elif provider == "grouped-autotuned":
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: grouped_autotuned_matmul(a, b), quantiles=quantiles)
  elif provider == "torch":
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b),
                                                 quantiles=quantiles)
  elif provider == "numpy-broadcast":
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: numpy_matmul(a, b),
                                                 quantiles=quantiles)
  elif provider == "cuda-naive":
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: mm_module.naive_matmul(a, b), quantiles=quantiles)
  elif provider == "cuda-shared":
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: mm_module.group_matmul(a, b), quantiles=quantiles)

  def gbps(ms):
    return 12 * sz * sz / ms * 1e-6

  return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark_matrix_size.run(print_data=True, show_plots=True, save_path=".")
