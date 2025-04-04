// 类型
unsigned char* 对应 uint8类型


// 内核函数定义
__global__ void kernelFunction(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] *= 2;
}

// 执行配置

- 2^31 max blocks for dim 0, 2^16 max for dims 1 & 2
- 1024 max threads per block (use a multiple of 32)

int numThreadsPerBlock = 256;
int numBlocks = (dataSize + numThreadsPerBlock - 1) / numThr eadsPerBlock;
kernelFunction<<<numBlocks, numThreadsPerBlock>>>(d_data);

__global__ void globalKernel() {
    // 内核函数代码
}

__device__ void deviceFunction() {
    // 设备端函数代码
}

__host__ void hostFunction() {
    // 主机端函数代码
}

__global__ void kernel() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // 使用tid进行计算
}

// warp操作

__shfl_sync 是 CUDA 中的一个内建函数，用于在同一个 warp（CUDA 中线程的基本调度单位，通常包含 32 个线程）内的线程之间同步地交换数据。
它允许线程获取同一 warp 内其他线程的指定变量值，而无需通过全局内存或共享内存，从而实现高效的线程间通信

for (int i = 0; i < 32; i++) {  
    int other_x = __shfl_sync(0xfffffff, x, i);  
    int other_y = __shfl_sync(0xfffffff, y, i);  
    int other_z = __shfl_sync(0xfffffff, z, i);  
}

__sync_warp