# TODO:
# reduce6
# total_reduce: reduce5
# total_reduce: thread_coarsning

# reduce-performance(gbps):
# size  torch.sum  torch.compile(sum)  ReduceSum0  ReduceSum1  ReduceSum2  ReduceSum3  ReduceSum4  ReduceSum5
# 0    64.0   0.028470            0.046784    0.041667    0.043716    0.043716    0.045714    0.045977    0.045455
# 1   128.0   0.056140            0.088889    0.082902    0.087432    0.089385    0.088398    0.089888    0.092486
# 2   256.0   0.106312            0.175824    0.160804    0.174863    0.176796    0.174863    0.181818    0.183908
# 3   512.0   0.187134            0.336842    0.300469    0.338624    0.342246    0.351648    0.357542    0.355556
# 4  1024.0   0.368876            0.589862    0.528926    0.643216    0.649746    0.670157    0.691892    0.699454


import torch
import triton
import triton.language as tl
import triton.testing

from gpu_kernel_utils import load_cuda, cuda_begin, cdiv

REDUCE_SIZE = 1024

cuda_src = cuda_begin + r'''
// reduce0: Interleaved Addressing
// Problem: highly divergent warps are very inefficient, and % operator is very slow

__global__ void reduce0(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	// do reduction in shared mem
	for(unsigned int s=1; s < blockDim.x; s *= 2) {
		if (tid % (2*s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// reduce1: 优化 % operator
// Problem: Shared Memory Bank Conflicts

__global__ void reduce1(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if ((tid & (2*s - 1)) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// reduce2: sequential addressing
// Problem: idle threads

__global__ void reduce2(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	// do reduction in shared mem
	for(unsigned int s=blockDim.x/2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// reduce3: first add during global load
// blockDim.x减半

__global__ void reduce3(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
	__syncthreads();
	// do reduction in shared mem
	for(unsigned int s=blockDim.x/2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Instruction Bottleneck
// At 17 GB/s, we’re far from bandwidth bound
// And we know reduction has low arithmetic intensity

// Therefore a likely bottleneck is instruction overhead
// Ancillary instructions that are not loads, stores, or
// arithmetic for the core computation
// In other words: address arithmetic and loop overhead

// Strategy: unroll loops

// reduce4: unroll last warp

__device__ void warpReduce(volatile int* sdata, int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

__global__ void reduce4(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
	__syncthreads();
	// do reduction in shared mem
	for(unsigned int s=blockDim.x/2; s > 32; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid < 32) warpReduce(sdata, tid);
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// reduce5: Completely unrolled with Templates
// 编译器确定if/else

template <unsigned int blockSize>
__device__ void warpReduce(volatile int* sdata, unsigned int tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce5(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
	__syncthreads();
	if (blockSize >= 512) {
		if (tid < 256) {
			sdata[tid] += sdata[tid + 256];
		}
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) {
			sdata[tid] += sdata[tid + 128];
		}
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64) {
			sdata[tid] += sdata[tid + 64];
		}
		__syncthreads();
	}
	if (tid < 32) warpReduce<blockSize>(sdata, tid);
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// O(N)
// O(N/P + log N)

// Cost of a parallel algorithm is processors × time complexity
// Allocate threads instead of processors: O(N) threads
// Time complexity is O(log N), so cost is O(N log N) : not cost efficient!

// Brent’s theorem suggests O(N/log N) threads
// Cost = O((N/log N) * log N) = O(N) : cost efficient!

// Sometimes called algorithm cascading
// Can lead to significant speedups in practice
// each thread should sum O(log n) elements

// In my experience, beneficial to push it even further
// Possibly better latency hiding with more work per thread
// More threads per block reduces levels in tree of recursive kernel invocations
// High kernel launch overhead in last levels with few blocks

// On G80, best perf with 64-256 blocks of 128 threads
// 1024-4096 elements per thread

// reduce 6: Multiple Adds Per Thread

template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata, unsigned int n) {
	extern __shared__ int sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;
	while (i < n) {
		sdata[tid] += g_idata[i] + g_idata[i+blockSize];
		i += gridSize; // gridSize loop stride to maintain coalescing!
	}
	__syncthreads();
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) warpReduce(sdata, tid);
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


void ReduceSum(torch::Tensor input, torch::Tensor output, int version) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);

    const auto size = input.numel();
    int threads_per_block = (size >= 1024) ? 1024 : size;
    int blocks = cdiv(size, threads_per_block);
    const int smem_size = threads_per_block * sizeof(int);
    if (version == 0) {
        reduce0<<<blocks, threads_per_block, smem_size>>>(
            input.data_ptr<int>(),
            output.data_ptr<int>()
        );
    } else if (version == 1) {
        reduce1<<<blocks, threads_per_block, smem_size>>>(
            input.data_ptr<int>(),
            output.data_ptr<int>()
        );
    } else if (version == 2) {
        reduce2<<<blocks, threads_per_block, smem_size>>>(
            input.data_ptr<int>(),
            output.data_ptr<int>()
        );
    } else if (version == 3) {
        threads_per_block /= 2;
        reduce3<<<blocks, threads_per_block, smem_size>>>(
            input.data_ptr<int>(),
            output.data_ptr<int>()
        );
    } else if (version == 4) {
        threads_per_block /= 2;
        reduce4<<<blocks, threads_per_block, smem_size>>>(
            input.data_ptr<int>(),
            output.data_ptr<int>()
        );
    } else if (version == 5) {
        threads_per_block /= 2;
        auto d_idata = input.data_ptr<int>();
        auto d_odata = output.data_ptr<int>();
        switch (threads_per_block) {
            case 512:
                reduce5<512><<< blocks, threads_per_block, smem_size >>>(d_idata, d_odata); break;
            case 256:
                reduce5<256><<< blocks, threads_per_block, smem_size >>>(d_idata, d_odata); break;
            case 128:
                reduce5<128><<< blocks, threads_per_block, smem_size >>>(d_idata, d_odata); break;
            case 64:
                reduce5< 64><<< blocks, threads_per_block, smem_size >>>(d_idata, d_odata); break;
            case 32:
                reduce5< 32><<< blocks, threads_per_block, smem_size >>>(d_idata, d_odata); break;
            case 16:
                reduce5< 16><<< blocks, threads_per_block, smem_size >>>(d_idata, d_odata); break;
            case 8:
                reduce5< 8><<< blocks, threads_per_block, smem_size >>>(d_idata, d_odata); break;
            case 4:
                reduce5< 4><<< blocks, threads_per_block, smem_size >>>(d_idata, d_odata); break;
            case 2:
                reduce5< 2><<< blocks, threads_per_block, smem_size >>>(d_idata, d_odata); break;
            case 1:
                reduce5< 1><<< blocks, threads_per_block, smem_size >>>(d_idata, d_odata); break;
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


'''

cpp_src = r'''
void ReduceSum(torch::Tensor input, torch::Tensor output, int version);
'''

reduce_module = load_cuda(
  cuda_src,
  cpp_src,
  ['ReduceSum'],
  opt=True,
  verbose=True,
  name='reduce',
)


def cuda_reduce_sum(A, version):
  output = torch.empty(cdiv(A.numel(), REDUCE_SIZE),
                       dtype=A.dtype,
                       device='cuda')
  reduce_module.ReduceSum(A, output, version)
  return output


def test_reduce():
  torch.manual_seed(0)
  test_sizes = [128, 256, 512, 1024, 2048, 4096]
  for total_size in test_sizes:
    x = torch.randint(0, 100, (total_size,), dtype=torch.int32, device='cuda')

    expected_output = x.reshape(-1, REDUCE_SIZE).sum(
      dim=1) if total_size > 1024 else x.sum().unsqueeze(0)
    assert expected_output.dtype == torch.int64, expected_output.dtype  # sum of ints is int

    output0 = cuda_reduce_sum(x, 0)
    assert output0.dtype == torch.int32, f"ReduceSum0 output dtype mismatch: expected torch.int32, got {output0.dtype}"
    assert output0.shape == expected_output.shape, f"ReduceSum0 output shape mismatch: expected {expected_output.shape}, got {output0.shape}"
    assert torch.equal(
      output0.long(), expected_output
    ), f"ReduceSum0 failed.\nOutput:\n{output0}\nExpected:\n{expected_output}"
    print("ReduceSum0 test passed.")

    output1 = cuda_reduce_sum(x, 1)
    assert output1.dtype == torch.int32, f"ReduceSum1 output dtype mismatch: expected torch.int32, got {output1.dtype}"
    assert output1.shape == expected_output.shape, f"ReduceSum1 output shape mismatch: expected {expected_output.shape}, got {output1.shape}"
    assert torch.equal(
      output1.long(), expected_output
    ), f"ReduceSum1 failed.\nOutput:\n{output1}\nExpected:\n{expected_output}"
    print("ReduceSum1 test passed.")

    output2 = cuda_reduce_sum(x, 2)
    assert output2.dtype == torch.int32, f"ReduceSum2 output dtype mismatch: expected torch.int32, got {output2.dtype}"
    assert output2.shape == expected_output.shape, f"ReduceSum2 output shape mismatch: expected {expected_output.shape}, got {output2.shape}"
    assert torch.equal(
      output2.long(), expected_output
    ), f"ReduceSum2 failed.\nOutput:\n{output2}\nExpected:\n{expected_output}"
    print("ReduceSum2 test passed.")

    output3 = cuda_reduce_sum(x, 3)
    assert output3.dtype == torch.int32, f"ReduceSum3 output dtype mismatch: expected torch.int32, got {output3.dtype}"
    assert output3.shape == expected_output.shape, f"ReduceSum3 output shape mismatch: expected {expected_output.shape}, got {output3.shape}"
    assert torch.equal(
      output3.long(), expected_output
    ), f"ReduceSum3 failed.\nOutput:\n{output3}\nExpected:\n{expected_output}"
    print("ReduceSum3 test passed.")

    output4 = cuda_reduce_sum(x, 4)
    assert output4.dtype == torch.int32, f"ReduceSum4 output dtype mismatch: expected torch.int32, got {output4.dtype}"
    assert output4.shape == expected_output.shape, f"ReduceSum4 output shape mismatch: expected {expected_output.shape}, got {output4.shape}"
    assert torch.equal(
      output4.long(), expected_output
    ), f"ReduceSum4 failed.\nOutput:\n{output4}\nExpected:\n{expected_output}"
    print("ReduceSum4 test passed.")

    output5 = cuda_reduce_sum(x, 5)
    assert output5.dtype == torch.int32, f"ReduceSum5 output dtype mismatch: expected torch.int32, got {output5.dtype}"
    assert output5.shape == expected_output.shape, f"ReduceSum5 output shape mismatch: expected {expected_output.shape}, got {output5.shape}"
    assert torch.equal(
      output5.long(), expected_output
    ), f"ReduceSum5 failed.\nOutput:\n{output5}\nExpected:\n{expected_output}"
    print("ReduceSum5 test passed.")

    print(f"All reduce tests passed successfully! total_size: {total_size}")


@triton.testing.perf_report(
  triton.testing.Benchmark(
    x_names=['size'],
    x_vals=[2**i for i in range(6, 11, 1)],
    line_arg='operation',
    line_vals=[
      'torch', 'torch_compile', 'cuda0', 'cuda1', 'cuda2', 'cuda3', 'cuda4', 'cuda5'
    ],
    line_names=[
      'torch.sum', 'torch.compile(sum)', 'ReduceSum0', 'ReduceSum1', 'ReduceSum2',
      'ReduceSum3', 'ReduceSum4', 'ReduceSum5'
    ],
    styles=[('blue', '-'), ('blue', '--'), ('green', '-'), ('red', '-'), ('purple', '-'),
            ('orange', '-'), ('cyan', '-'), ('magenta', '-')],
    ylabel='us',
    plot_name='reduce-performance(gbps)',
    args={},
  ))
def benchmark(size, operation):
  x = torch.randint(0, 100, (size,), device='cuda', dtype=torch.int32)
  if operation == 'torch':
    if size <= 1024:
      fn = lambda: x.sum()
    else:
      reshaped_x = x.reshape(-1, REDUCE_SIZE)
      fn = lambda: reshaped_x.sum(dim=1)
  elif operation == 'torch_compile':
    compiled_sum = torch.compile(torch.sum)
    if size <= 1024:
      fn = lambda: compiled_sum(x)
    else:
      reshaped_x = x.reshape(-1, REDUCE_SIZE)
      fn = lambda: compiled_sum(reshaped_x, dim=1)
  elif operation == 'cuda0':
    fn = lambda: cuda_reduce_sum(x, 0)
  elif operation == 'cuda1':
    fn = lambda: cuda_reduce_sum(x, 1)
  elif operation == 'cuda2':
    fn = lambda: cuda_reduce_sum(x, 2)
  elif operation == 'cuda3':
    fn = lambda: cuda_reduce_sum(x, 3)
  elif operation == 'cuda4':
    fn = lambda: cuda_reduce_sum(x, 4)
  elif operation == 'cuda5':
    fn = lambda: cuda_reduce_sum(x, 5)
  else:
    raise ValueError(f"Invalid operation: {operation}")
  ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
  gbps = lambda ms: (x.numel() * x.element_size()) / ms * 1e-6
  return gbps(ms), gbps(max_ms), gbps(min_ms)
  # return ms * 1e3, max_ms * 1e3, min_ms * 1e3


if __name__ == '__main__':
  test_reduce()
  benchmark.run(print_data=True, save_path='.')
