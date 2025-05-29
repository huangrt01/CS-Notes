

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


// Brent-Kung Parallel Scan
__global__ void scan_kernel_3(float* input, float* output, float* partialSums, unsigned int N) {
    unsigned int segment = 2*blockIdx.x*blockDim.x;

    __shared__ float buffer_s[2*BLOCK_DIM];
    buffer_s[threadIdx.x] = input[segment + threadIdx.x];
    buffer_s[threadIdx.x + BLOCK_DIM] = input[segment + threadIdx.x + BLOCK_DIM];
    __syncthreads();

    // First tree
    for (unsigned int stride = 1; stride <= BLOCK_DIM; stride *= 2) {
        unsigned int i = (threadIdx.x + 1)*2*stride - 1;
        if (i < 2*BLOCK_DIM) {
            buffer_s[i] += buffer_s[i - stride];
        }
        __syncthreads();
    }

    // Second tree
    for (unsigned int stride = BLOCK_DIM/2; stride >= 1; stride /= 2) {
        unsigned int i = (threadIdx.x + 1)*2*stride - 1;
        if (i + stride < 2*BLOCK_DIM) {
            buffer_s[i + stride] += buffer_s[i];
        }
        __syncthreads();
    }

    // Store partial sum
    if (threadIdx.x == 0) {
        partialSums[blockIdx.x] = buffer_s[2*BLOCK_DIM - 1];
    }

    // Store output
    output[segment + threadIdx.x] = buffer_s[threadIdx.x];
    output[segment + threadIdx.x + BLOCK_DIM] = buffer_s[threadIdx.x + BLOCK_DIM];
}


// thread coarsening

#define COARSE_FACTOR 8

__global__ void scan_kernel_4(float* input, float* output, float* partialSums, unsigned int N) {
    unsigned int segment = blockIdx.x*blockDim.x*COARSE_FACTOR;

    // Load elements from global memory to shared memory
    __shared__ float buffer_s[BLOCK_DIM*COARSE_FACTOR];
    for(unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        buffer_s[c*BLOCK_DIM + threadIdx.x] = input[segment + c*BLOCK_DIM + threadIdx.x];
    }
    __syncthreads();

    // Thread scan
    unsigned int threadSegment = threadIdx.x*COARSE_FACTOR;
    for(unsigned int c = 1; c < COARSE_FACTOR; ++c) {
        buffer_s[threadSegment + c] += buffer_s[threadSegment + c - 1];
    }
    __syncthreads();

    // Allocate and initialize double buffers for partial sums
    __shared__ float buffer1_s[BLOCK_DIM];
    __shared__ float buffer2_s[BLOCK_DIM];
    float* inBuffer_s = buffer1_s;
    float* outBuffer_s = buffer2_s;
    inBuffer_s[threadIdx.x] = buffer_s[threadSegment + COARSE_FACTOR - 1];
    __syncthreads();

    // Parallel scan of partial sums
    for(unsigned int stride = 1; stride <= BLOCK_DIM/2; stride *= 2) {
        if(threadIdx.x >= stride) {
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx.x - stride];
        } else {
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
        }
        __syncthreads();
        float* tmp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = tmp;
    }

    // Add previous thread's partial sum
    if(threadIdx.x > 0) {
        float prevPartialSum = inBuffer_s[threadIdx.x - 1];
        for(unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            buffer_s[threadSegment + c] += prevPartialSum;
        }
    }
    __syncthreads();

    // Save block's partial sum
    if(threadIdx.x == BLOCK_DIM - 1) {
        partialSums[blockIdx.x] = inBuffer_s[threadIdx.x];
    }

    // write output
    for(unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        output[segment + c*BLOCK_DIM + threadIdx.x] = buffer_s[c*BLOCK_DIM + threadIdx.x];
    }
}

__global__ void add_kernel(float* output, float* partialSums, unsigned int N) {
    unsigned int segment = COARSE_FACTOR*blockIdx.x*blockDim.x;
    if(blockIdx.x > 0) {
        for(unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            output[segment + c*BLOCK_DIM + threadIdx.x] += partialSums[blockIdx.x - 1];
        }
    }
}