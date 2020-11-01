[toc]

[Cuda-Downloads](https://developer.nvidia.com/cuda-downloads)

[Cuda-C-Best-Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#who-should-read-this-guide)



3 ways to accelerate applications:

1.Libraries    

2.OpenACC    	(add directive, for applications like HPC)

3.Programming Languages



CPU: latency-optimized low latency processor

GPU: throughput-optimized high throughput processor

![CPU-GPU](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/nvidia/CPU-GPU.png)

#### 1.Accelerating Applications with CUDA C/C++

[课程网页](https://courses.nvidia.com/courses/course-v1:DLI+C-AC-01+V1/courseware/85f2a3ac16a0476685257996b84001ad/9ef2f68fb10d40c5b54b783392938d04/?activate_block_id=block-v1%3ADLI%2BC-AC-01%2BV1%2Btype%40sequential%2Bblock%409ef2f68fb10d40c5b54b783392938d04)

##### Writing Application Code for the GPU

 CUDA accelerates applications drastically with little effort, has an ecosystem of highly optimized libraries for [DNN](https://developer.nvidia.com/cudnn), [BLAS](https://developer.nvidia.com/cublas), [graph analytics](https://developer.nvidia.com/nvgraph), [FFT](https://developer.nvidia.com/cufft), and more, and also ships with powerful [command line](http://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview) and [visual profilers](http://docs.nvidia.com/cuda/profiler-users-guide/index.html#visual).

CUDA supports many, if not most, of the [world's most performant applications](https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/catalog/?product_category_id=58,59,60,293,98,172,223,227,228,265,487,488,114,389,220,258,461&search=) in, [Computational Fluid Dynamics](https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/catalog/?product_category_id=10,12,16,17,19,51,53,71,87,121,124,156,157,195,202,203,204,312,339,340,395,407,448,485,517,528,529,541,245,216,104,462,513,250,492,420,429,490,10,12,16,17,19,51,53,71,87,121,124,156,157,195,202,203,204,312,339,340,395,407,448,485,517,528,529,541,245,216,104,462,513,250,492,420,429,490,10,12,16,17,19,51,53,71,87,121,124,156,157,195,202,203,204,312,339,340,395,407,448,485,517,528,529,541,245,216,104,462,513,250,492,420,429,490&search=), [Molecular Dynamics](https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/catalog/?product_category_id=8,57,92,123,211,213,237,272,274,282,283,307,325,337,344,345,351,362,365,380,396,398,400,435,507,508,519,8,57,92,123,211,213,237,272,274,282,283,307,325,337,344,345,351,362,365,380,396,398,400,435,507,508,519,8,57,92,123,211,213,237,272,274,282,283,307,325,337,344,345,351,362,365,380,396,398,400,435,507,508,519,8,57,92,123,211,213,237,272,274,282,283,307,325,337,344,345,351,362,365,380,396,398,400,435,507,508,519&search=), [Quantum Chemistry](https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/catalog/?product_category_id=8,57,92,123,211,213,237,272,274,282,283,307,325,337,344,345,351,362,365,380,396,398,400,435,507,508,519,8,57,92,123,211,213,237,272,274,282,283,307,325,337,344,345,351,362,365,380,396,398,400,435,507,508,519&search=), [Physics](https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/catalog/?product_category_id=6,24,116,118,119,135,229,231,372,373,392,393,489,493,494,495,496,497,498,67,170,216,281,6,24,116,118,119,135,229,231,372,373,392,393,489,493,494,495,496,497,498,67,170,216,281,6,24,116,118,119,135,229,231,372,373,392,393,489,493,494,495,496,497,498,67,170,216,281,6,24,116,118,119,135,229,231,372,373,392,393,489,493,494,495,496,497,498,67,170,216,281,6,24,116,118,119,135,229,231,372,373,392,393,489,493,494,495,496,497,498,67,170,216,281&search=) and HPC.

```c++
nvidia-smi
cudaMallocManaged()
cudaDeviceSynchronize()
  
nvcc -arch=sm_70 -o hello-gpu 01-hello/01-hello-gpu.cu -run
```

code executed on the CPU is referred to as **host** code, and code running on the GPU is referred to as **device** code

```c++
void CPUFunction()
{
  printf("This function is defined to run on the CPU.\n");
}

__global__ void GPUFunction()
{
  printf("This function is defined to run on the GPU.\n");
}
// __global__表示CPU/GPU均可执行，必须返回void

int main()
{
  CPUFunction();
	// launch a kernel, provide an execution configuration
  GPUFunction<<<1, 1>>>();
  cudaDeviceSynchronize();
  // CPU等待GPU
}
```

At a high level, execution configuration allows programmers to specify the **thread hierarchy** for a kernel launch, which defines the number of thread groupings (called **blocks**), as well as how many **threads** to execute in each block.

[**NVIDIA CUDA Compiler**](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html), [documentation](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)

[`arch` flag](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation), [virtual architecture features](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list) 



##### CUDA Thread Hierarchy

三层抽象：grid, block, thread
* kernel execution configuration ~ grid

CUDA-Provided Thread Hierarchy Variables，可以在`__global__`函数里直接用，作为标识来实现并行
* 并行是无序的
* `threadIdx.x + blockIdx.x * blockDim.x`

```c++
gridDim.x: num of blocks
blockDim.x: num of threads in a block
blockIdx.x: block index
threadIdx.x: thread index
```

如果不同的线程在不同的Warp里，他们的执行顺序会有所不同（但如果执行的操作非常简单，那么它们执行的先后时间也相差非常小），因为硬件（SM）是以Warp为单位调度线程运行的，每个Warp有32个线程

##### Allocating Memory to be accessed on the GPU and the CPU
```c++
// CPU-only

int N = 2<<20;
size_t size = N * sizeof(int);

int *a;
a = (int *)malloc(size);

// Use `a` in CPU-only program.

free(a);
// Accelerated

int N = 2<<20;
size_t size = N * sizeof(int);

int *a;
// Note the address of `a` is passed as first argument.
cudaMallocManaged(&a, size);

// Use `a` on the CPU and/or on any GPU in the accelerated system.

cudaFree(a);
```

 这个地址在统一的内存空间里，GPU和CPU都可以使用，但物理上数据可以不在它被访问的设备里，这时会产生page fault（缺页错误），对这个错误的处理就是把数据拷贝到需要访问它的设备或主机内存里，这个操作是透明的（自动执行）。

https://on-demand.gputechconf.com/gtc/2018/presentation/s8430-everything-you-need-to-know-about-unified-memory.pdf
https://developer.nvidia.com/blog/unified-memory-cuda-beginners/


##### Grid Size Work 

Amount Mismatch
`size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;`

* threads_per_bloack最大为1024

Grid-Stride Loops

```cpp
__global__ void kernel(int *a, int N)
{
  int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;

  for (int i = indexWithinTheGrid; i < N; i += gridStride)
  {
    // do work on a[i];
  }
}
```

##### Error Handling

```c++
cudaError_t err;
err = cudaMallocManaged(&a, N)                    // Assume the existence of `a` and `N`.

if (err != cudaSuccess)                           // `cudaSuccess` is provided by CUDA.
{
  printf("Error: %s\n", cudaGetErrorString(err)); // `cudaGetErrorString` is provided by CUDA.
}

someKernel<<<1, -1>>>();  // -1 is not a valid number of threads.

cudaError_t err;
err = cudaGetLastError(); // `cudaGetLastError` will return the error from above.
if (err != cudaSuccess)
{
  printf("Error: %s\n", cudaGetErrorString(err));
}
```

```c++
#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

int main()
{

/*
 * The macro can be wrapped around any function returning
 * a value of type `cudaError_t`.
 */

  checkCuda(cudaDeviceSynchronize())
}
```



利用dim3完成2d、3d任务

```c++
dim3 threads_per_block(16, 16, 1);
dim3 number_of_blocks(16, 16, 1);
someKernel<<<number_of_blocks, threads_per_block>>>()
```

##### Environments and Projects

https://github.com/NVIDIA/nvidia-docker

GPU加速[Mandelbrot Set Simulator](https://github.com/sol-prog/Mandelbrot_set)

Peruse [*GPU-Accelerated Libraries for Computing*](https://developer.nvidia.com/gpu-accelerated-libraries) to learn where you can use highly optimized CUDA libraries for tasks like [basic linear algebra solvers](https://developer.nvidia.com/cublas) (BLAS), [graph analytics](https://developer.nvidia.com/nvgraph), [fast fourier transforms](https://developer.nvidia.com/cufft) (FFT), [random number generation](https://developer.nvidia.com/curand) (RNG), and [image and signal processing](https://developer.nvidia.com/npp), to name a few.

#### 2.Managing Accelerated Application Memory with CUDA Unified Memory and nsys

Assess, Parallelize, Optimize, Deploy(APOD) design cycle

##### Iterative Optimizations with the NVIDIA Command Line Profiler

Profile configuration details, Report file(s) generation details, CUDA API Statistics, CUDA Kernel Statistics, CUDA Memory Operation Statistics (time and size), OS Runtime API Statistics

```shell
nvcc -o single-thread-vector-add 01-vector-add/01-vector-add.cu -run
nsys profile --stats=true -o output-report ./single-thread-vector-add
```

##### Streaming Multiprocessors and Querying the Device

GPU内部很多functional units: SMs(Streaming Multiprocessors)，一个SM可以schedule多个block，但同一时间只能执行一个

GPU and Programming Model

* A set of CUDA cores
  * Tensor core相比CUDA core，实现了MMA operations，支持2:4 sparsity，支持in8和int4，更高效
  
  * CUDA Core和Thread在抽象层次上对应; SM <-> Thread Block; Device <-> Grid
* Warp is successive 32 threads in a block
  * 一个SM有64个warp，一个warp scheduler 16个warp
  * 如果不能整除，余数占据one more warp
  * 一个warp中的线程执行同一指令（Volta架构之后，可以执行不同指令，但不同时）
  * Instructions will be issued to execution units by warp.
  * Latency is caused by not able to issue next instruction to execution unit
  * warp's context switching is free -> 通过大量warp来hide memory latency
* Registers / Shared Memory / L1 Cache
* SMs share Global Memory 
* PCIe / NVLINk 与CPU Chipset交互

![SM](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/nvidia/SM.png)

* 白框：subpartition

![memory-hierarchy](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/nvidia/memory-hierarchy.png)

* SM片上单元比L2快3倍，比Global Memory快几十倍 

```c++
cudaMallocManaged()     不注意的话开销大
cudaMalloc()       分配显存
cudaMemcpyHostToDevice
```

* 编译器决定kernel内定义的变量是分配在寄存器上（**没有超过上限的标量**）还是per-thread local memory上
* 寄存器之间的值不一定是私有的，可以shuffle

![shared-memory](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/nvidia/shared_memory.png)

![nvlink](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/nvidia/nvlink.png)

* 左边的不是全连接结构，NV引入了NVSwitch单元，交换机芯片，支持最多16个GPU的直联



block size的选择，最小取64，通常取128、256

* SM的倍数
* 32的倍数， [in depth coverage of SMs and warps](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation)

```c++
#include <stdio.h>

int main()
{
  int deviceId;
  cudaGetDevice(&deviceId);                  
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId); 

  int computeCapabilityMajor = props.major;
  int computeCapabilityMinor = props.minor;
  int multiProcessorCount = props.multiProcessorCount;
  int warpSize = props.warpSize;

  printf("Device ID: %d\nNumber of SMs: %d\nCompute Capability Major: %d\nCompute Capability Minor: %d\nWarp Size: %d\n", deviceId, multiProcessorCount, computeCapabilityMajor, computeCapabilityMinor, warpSize);
}
```



![workflow](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/nvidia/workflow.png)

**dynamic parallelism in cuda**: kernel内执行kernel，但launch kernel开销较大，有几微秒



##### CPU-GPU Interaction Optimization

* Host<->device data transfer has much lower bandwidth than global memory access.

  * 16 GB/s (PCIe x16 Gen3) vs 250 GB/s & 10.6 T inst/s (GP100) 

* Minimize transfer

  * Intermediate data can be allocated, operated, de-allocated directly on GPU

  * Sometimes it’s even better to re-compute on GPU

* Group transfer
  * One large transfer much better than many small ones
  * Overlap memory transfer with computation








#### 3.Optimization Workflow

unmanaged memory allocation and migration; pinning, or page-locking host memory; and non-default concurrent CUDA streams.

![optimization-workflow](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/nvidia/optimization-workflow.png)

优化工具：

* NVVP & nvprof (legacy)
* Nsight System & Nsight Compute



prefetch: 减少HtoD耗时（因为larger chunks），大幅减少kernel耗时（不再page fault）

init-kernel: 不再有HtoD，OS耗时也没了

https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/



##### CUDA Streams
* Kernels within any single stream must execute in order
* Kernels in different, non-default streams can interact concurrently
* The default stream is special: it blocks all kernels in all other streams 
  * 这一规则有副作用，因此推荐用non-default streams

```c++
cudaStream_t stream;       // CUDA streams are of type `cudaStream_t`.
cudaStreamCreate(&stream); // Note that a pointer must be passed to `cudaCreateStream`.

someKernel<<<number_of_blocks, threads_per_block, 0, stream>>>(); // `stream` is passed as 4th EC argument.
// 3th argument: the number of bytes in shared memory

cudaStreamDestroy(stream); // Note that a value, not a pointer, is passed to `cudaDestroyStream`.


for (int i = 0; i < 5; ++i){
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  printNumber<<<1, 1, 0, stream>>>(i);
  cudaStreamDestroy(stream);
}
cudaDeviceSynchronize();
```



Exericise: Accelerate and Optimize an N-Body Simulator

nbody-raw.cu -> nbody-optimized.cu



##### Memory Optimization

* Improve memory access pattern to reduce wasted transactions，提高bus utilization
  * per warp (coalesced, 32/64/128B)
    * 线程访问可以不连续，但内存需要连续
  * in discrete chunks (transported in segments: L2 cache line, 32B)
    * 长度和index均需对齐
* Reduce redundant access: shared memory
  * Inter-block communication
  * User-managed cache to reduce redundant global memory accesses
  * Avoid non-coalesced access: shared memory没有cache line的概念，e.g. matrix-transposition.cu

![stencil](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/nvidia/stencil.png)

Shared memory应用于矩阵乘法：global read次数除以BATCH_SIZE

![warp](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/nvidia/warp-sharing.png)

Manual Device Memory Allocation and Copying

```c++
int *host_a, *device_a;        // Define host-specific and device-specific arrays.
cudaMalloc(&device_a, size);   // `device_a` is immediately available on the GPU.
cudaMallocHost(&host_a, size); // `host_a` is immediately available on CPU, and is page-locked, or pinned.

initializeOnHost(host_a, N);   // No CPU page faulting since memory is already allocated on the host.

// `cudaMemcpy` takes the destination, source, size, and a CUDA-provided variable for the direction of the copy.
cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);

kernel<<<blocks, threads, 0, someStream>>>(device_a, N);

// `cudaMemcpy` can also copy data from device to host.
cudaMemcpy(host_a, device_a, size, cudaMemcpyDeviceToHost);

verifyOnHost(host_a, N);

cudaFree(device_a);
cudaFreeHost(host_a);          // Free pinned memory like this.
```

cudaHostAlloc: Pinned(Non-pageable) Memory, very expensive




##### Latency Optimization

**Warp State**

* Active: warps inside the pool which has non-exiting threads
* Eligible: active warps that are not stalled
* Issued: a single eligible warp that the warp scheduler choose to issue one or more instructions on this cycle

**Latency**

* bound: for many cycles, lack of eligible warps to issue instructions

* hiding: switching warp
* technique: increase active warps

**Occupancy & Active Warps**

* Occupancy: ratio of active warps per SM to the maximum number of allowed warps
  * Hardware limit: 64 in Volta GV100 Per SM(16 per sub-partition), but **32** in Turing
* We need the occupancy to be high enough to hide latency
* Theoretical occupancy is limited by resource usage (shared memory/registers/blocks per SM)

**Achieved occupancy can be significantly lower than theoretical occupancy when:** 

*  Unbalanced workload within blocks
* Unbalanced workload across blocks
* Too few blocks launched

**Occupancy Optimization**

* Know the occupancy: NVIDIA Visual profiler / Nsight Compute 
* Adjust resource usage to increase theoretical occupancy
  * Change block size
  * Limit register usage

    * Compiler option –maxregcount=n: per file

    * `__launch_bounds__`: per kernel 
  * Limit shared memory usage.

* Launch enough load-balanced blocks to increase achieved occupancy

```c++
__global__ void
__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
MyKernel(...){
  ...
}
```




Using Streams to Overlap Data Transfers and Code Execution

* cudaMemcpyAsync: `manual-malloc.cu`, `memcpy-async.cu`



##### Instruction Optimization

* Use float if precision allow
  * Adding “f” to floating literals (e.g. 1.0f) because the default is double 

* Fast math functions
  * Two types of runtime math library functions
    * func(): slower but higher accuracy (5 ulp or less)
    * __func(): fast but lower accuracy (SFU做，see prog. guide for full details) 
    *  -use_fast_math: forces every func() to __func ()

* High-throughput function:
  * DP4A and DP2A for int8 and int16 dot productions 
  * Warp matrix function for tensor core operations

![control-flow](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/nvidia/control-flow.png)

##### CUDA cooperative group 协作线程组

<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/nvidia/cooperative-groups.png" alt="cooperative-groups.png" style="zoom:100%;" />

```c++
namespace cooperative_groups{
class thread_group{
public:
  __device__ unsigned int size() const;
  __device__ unsigned int thread_rank() const;
  __device__ void sync() const;
};
  
thread_block myblock = this_thread_block();
// intra-block groups
thread_group tile32 = tiled_partition(myblock, 32);
thread_group tile4 = tiled_partition(tile32, 4);
thread_block_tile<8> tile8 = tiled_partition<8>(this_thread_block());


// Warp collectives

template <unsigned int Size>
class thread_block_tile : public thread_group{
public:
  __device__ unsigned int size() const;
  __device__ unsigned int thread_rank() const;
  __device__ void sync() const;
  
  // Shuffle collectives
  __device__ int shfl(int var, int srcRank) const;
  __device__ int shfl_down(int var, unsigned int delta) const;
  __device__ int shfl_up(int var, unsigned int delta) const;
  __device__ int shfl_xor(int var, unsigned int laneMask);
  
  // Vote collectives
  __device__ int any(int predicate) const;
  __device__ int all(int predicate) const;
  __device__ unsigned int ballot(int predicate) const;
  
  // Match collectives
  __device__ unsigned int match_any(int val);
  __device__ unsigned int match_all(int val, int &pred);
}; 
}
```



![shuffle](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/nvidia/shuffle.png)



#### 4.Lecture: Introduction to Nsight Profiling Tools

![nsight-product](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/nvidia/nsight-product.png)


```shell
nsys profile -t cuda,osrt,nvtx -o baseline -w true python main.py
```

Support:

* OS Thread state and CPU utilization, pthread, file I/O, etc.
* User annotations API (NVTX)
* Compute
  * CUDA API: Kernel launch and execution correlation
  * Libraries and directive: cuBLAS, cuDNN, OpenACC
* Graphics
  * Vulkan, OpenGL, DX11, DX12, DXR, V-sync



Key features

* section is a group of metrics

![warp-scheduler](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/nvidia/warp-scheduler.png)



#### 5.Lecture: NVIDIA GPU通用推理加速及部署SDK

线上推理加速的思路
* 模型本身的加速，TensorRT、DL complier
  * [Layer & Tensor Fusion](https://docs.nvidia.com/deeplearning/tensorrt/best-practices/): 横向/纵向的融合，减少copy显存; layer merge (concatforward)
  * Weights & Activation Precision Calibration
    * Symmetric quantization: 超参threshold，超过会截断，提高转换精度
    * 用KL-divergence来衡量threshold

  * Kernel Auto-Tuning: 找当前硬件下最优的卷积算法、kernels、tensor layouts
  * Dynamic Tensor Memory: 给层加引用计数 

```c++
  IBuilderConfig * config = builder->createBuilderConfig(); 
  config->setFlag(BuilderFlag::kFP16); //INT8 and FP16 can be both set
```

* 部署服务，CPU or GPU

* Model Parser 解析TensorFlow/Caffe模型
  * [ONNX Parser](https://github.com/onnx)
* TensorRT Network Definition API
  * 自定义算子需要自己写

```c++
IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
INetworkDefinition* network = builder->createNetwork();
ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{1, INPUT_H, INPUT_W});
IScaleLayer* scale_1 = network->addScale(*data, ScaleMode::kUNIFORM, shift, scale, power);
IConvolutionLayer* conv2 = network->addConvolution(*scale_1->getOutput(0), 50, DimsHW{5, 5}, weightMap["conv2filter"], weightMap["conv2bias"]);
conv2->setStride(DimsHW{1, 1});
ISoftMaxLayer* prob = network->addSoftMax(*conv2->getOutput(0));
prob->getOutput(0)->setName(OUTPUT_BLOB_NAME); // set output 
network->markOutput(*prob->getOutput(0)); // mark output

//序列化反序列化
IHostMemory* trtModelStream = engine->serialize(); //store model to disk
//<...>
IRuntime* runtime = createInferRuntime(gLogger.getTRTLogger());
ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
IExecutionContext* context = engine->createExecutionContext();
```



TF-TRT (TensorFlow integration with TensorRT) parses the frozen TF graph or saved model, and **converts each supported subgraph to a TRT optimized node** (TRTEngineOp), allowing TF to execute the remaining graph.

```c++
# Set Precision
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
precision_mode=trt.TrtPrecisionMode.INT8)
# Convert to TF-TRT Graph
converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir, conversion_params=conversion_params)
# INT8 Calibration
converter.convert(calibration_input_fn=my_calibration_fn)
# Run Inference 
converter.save(output_saved_model_dir)
```



Triton Inference Server

* Client/server在本地：Inputs/outputs needed to be passed to/from Triton are stored in system/CUDA shared memory. Reduces HTTP/gRPC overhead

![utilize-gpu](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/nvidia/utilize-gpu.png)

```
dynamic_batching {
	preferred_batch_size:[4,8],
	max_queue_delay_microseconds: 100,
}
```

Case Study: NVIDA BERT Solution: [FasterTransformer2.0](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer) and [TensorRT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT/trt)



#### 6.Lecture: NVIDIA ASR & TTS SOLUTIONS

#####  ASR WFST decoding on GPU

ASR Pipeline
* 多级的转换：speech -> phoneme -> character -> word -> sentence
  * 即使是深度学习兴起，工业界少有用e2e
  * 多级带来海量choices，需要构建一个decoder解决识别任务(a search problem)
* ASR system overview

![ASR-system](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/nvidia/ASR-system.png)

*Q: How do we combine HMM, Lexicon & LM together?*

*A: WFST (Weighted Finite State Transducer)*



WFST是一种图的表示方式，能通用地表示上述三种模型，然后这三张图可以合并。

* HMM的输出是phoneme的输入，phoneme的输出是language model的输入

* WFST Decoding: 图的最短路径问题，Token Pathing，Traverse the graph by copying token

Kaldi CUDA decoding pipeline

* WFST Decoding逻辑判断和对象copy较多，之前很长时间之后CPU实现
* GPU DECODE CHALLENGES
  * Dynamic workload
    * Amount of parallelism varies greatly throughout decode process
    * Can have few or many candidates moving from frame to frame
  * Limited parallelism
    * Even with many candidates, the amount of parallelism is still far smaller to saturate a GPU
  * Complex data structure
    * Need a GPU-friendly data layout to obtain high performance on GPU
* CUDA DECODER
  * Operate FST on GPU
    * CudaFst takes ~1/3 of its original size
  * Accelerate decoding by parallelization
    * Batch processing: batch不同语句的chunks，支持context switch
    * Token Passing in parallel
  * Process in streaming manner
* ASR GPU PIPELINE: e2e acceleration, feature extraction + Acoustic Model + Language Model
  * 结合Triton Inference Server


![asr-pipeline](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/nvidia/asr-pipeline.png)



Reference:

- Blogs: https://developer.nvidia.com/blog/gpu-accelerated-speech-to-text-with-kaldi-a-tutorial-on-getting-started/
- Kaldi integration with Triton: https://github.com/NVIDIA/DeepLearningExamples/tree/master/Kaldi/SpeechRecognition
- Kaldi GPU decoder
  - NGC: nvcr.io/nvidia/kaldi:20.08-py3
  - Kaldi github: github.com/kaldi-asr/src/cudadecoder



##### Text To Speech(TTS) Synthesis

Modern TTS Solution

* Synthesizer: TACOTRON 2模型，合成发音特征
* Vocoder：声码器 WAVENET、WAVEGLOW
  * 思路：利用可逆网络生成声音, affine coupling layer很关键

![waveglow](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/nvidia/waveglow.png)



APEX ～ Mixed Precision Training

BERT

* Challenge: polyphone disambiguation, prodisic structure prediction

* BERT Optimization: 对self-attention layer做kernel fusion