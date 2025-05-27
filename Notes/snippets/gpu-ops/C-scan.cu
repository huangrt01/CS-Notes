

__global__ void add_kernel(float* output, float* partialSums, unsigned int N) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (blockIdx.x > 0) {
        output[i] += partialSums[blockIdx.x - 1];
    }
}

// Kogge-Stone Parallel Scan
// 问题：两次sync开销大
#define BLOCK_DIM 1024
__global__ void scan_kernel_1(float* input, float* output, float* partialSums, unsigned int N) {
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ float buffer_s[BLOCK_DIM];
	buffer_s[threadIdx.x] = input[i];
	__syncthreads();
	for (unsigned int stride = 1; stride <= BLOCK_DIM/2; stride *= 2) {
	    float v;
	    if (threadIdx.x >= stride) {
	        v = buffer_s[threadIdx.x - stride];
	    }
	    __syncthreads();  // Wait for everyone to read
	    if (threadIdx.x >= stride) {
	        buffer_s[threadIdx.x] += v;
	    }
	    __syncthreads();
	}
	if (threadIdx.x == BLOCK_DIM - 1) {
	    partialSums[blockIdx.x] = buffer_s[threadIdx.x];
	}
	output[i] = buffer_s[threadIdx.x];
}


// double buffering
__global__ void scan_kernel_2(float* input, float* output, float* partialSums, unsigned int N) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ float buffer1_s[BLOCK_DIM];
    __shared__ float buffer2_s[BLOCK_DIM];
    float* inBuffer_s = buffer1_s;
    float* outBuffer_s = buffer2_s;
    inBuffer_s[threadIdx.x] = input[i];
    __syncthreads();

    for (unsigned int stride = 1; stride <= BLOCK_DIM/2; stride *= 2) {
        if (threadIdx.x >= stride) {
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx.x - stride];
        } else {
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
        }
        __syncthreads();
        float* temp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = temp;
    }

    if (threadIdx.x == BLOCK_DIM - 1) {
        partialSums[blockIdx.x] = inBuffer_s[threadIdx.x];
    }

    output[i] = inBuffer_s[threadIdx.x];
}