// reduce0: Interleaved Addressing
// Problem: highly divergent warps are very inefficient, and % operator is very slow
// Performance: Time (2^22 ints): 8.05ms, Bandwidth: 2.08GB/s

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
// Performance: Time (2^22 ints): 3.46ms, Bandwidth: 4.85GB/s

// reduce2: sequential addressing
// Problem: idle threads
// Performance: Time (2^22 ints): 1.72ms, Bandwidth: 9.74GB/s

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
// Performance: Time (2^22 ints): 0.97ms, Bandwidth: 17.38GB/s

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
// Performance: Time (2^22 ints): 0.54ms, Bandwidth: 31.29GB/s

__device__ void warpReduce(volatile int* sdata, int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

__global__ void reduce3(int *g_idata, int *g_odata) {
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
// Performance: Time (2^22 ints): 0.38ms, Bandwidth: 41.00GB/s

switch (threads) {
	case 512:
		reduce5<512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
	case 256:
		reduce5<256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
	case 128:
		reduce5<128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
	case 64:
		reduce5< 64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
	case 32:
		reduce5< 32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
	case 16:
		reduce5< 16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
	case 8:
		reduce5< 8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
	case 4:
		reduce5< 4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
	case 2:
		reduce5< 2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
	case 1:
		reduce5< 1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
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


Template <unsigned int blockSize>
__device__ void warpReduce(volatile int* sdata, int tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
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
// Performance: Time (2^22 ints): 0.27ms, Bandwidth: 62.67GB/s

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

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
