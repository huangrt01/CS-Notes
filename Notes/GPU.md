[toc]

[Cuda-Downloads](https://developer.nvidia.com/cuda-downloads)

[Cuda-C-Best-Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#who-should-read-this-guide)

https://docs.nvidia.com/cuda/cuda-c-programming-guide/



### Intro

> * GPU Mode
>   * Youtube: https://www.youtube.com/channel/UCJgIbYl6C5no72a0NUAPcTA
>   * lecture: https://github.com/gpu-mode/lectures
>   * Blog: https://christianjmills.com/blog.html#listing-listing-page=1
>
> * 书：Programming Massively Parallel Processors (PMPP) 3rd edition
>   * 视频：https://www.youtube.com/@pmpp-book/videos?view=0&sort=dd&shelf_id=2
> * [Nvidia’s CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
> * [How GPU Computing Works (YouTube)](https://www.youtube.com/watch?v=3l10o0DYJXg)
> * [GPU Programming: When, Why and How?](https://enccs.github.io/gpu-programming/)

* Intro
  * 科普小视频，绘画形象说明GPU和CPU区别：https://www.bilibili.com/video/BV1ry4y1y7KZ

* Amdahl's Law
  * achievable speedup is limited by the parallelizable portion **p** of programs
    * speedup<1/(1-**p**)
    * e.g., if **p** is 90%, speedup < 10×
  * Fortunately, for many real applications, **p** > 99% especially for large datasets, and speedups >100× are attainable



### GPU

#### Intro

* launch many threads, 每个output element一个thread是合理的
* Op开发流程
  * If it's not fast enough, `torch.compile` it.
  * If it's not fast enough, check if you can rewrite your code to make it more suitable for `torch.compile`.
  * If it's not fast enough, check which parts are slow and write custom Triton kernel(s) for those.
  * If it's not fast enough, check which parts are slow and write custom CUDA kernel(s) for those.

#### GPU vs CPU

* GPU 101
  * https://blog.codingconfessions.com/p/gpu-computing

* GPU 的设计目标，GPU v.s CPU：
  - 并行&串行
    - GPU 侧重并行计算
    - CPU侧重串行计算（[instruction pipelining](https://en.wikipedia.org/wiki/Instruction_pipelining), [out of order execution](https://en.wikipedia.org/wiki/Out-of-order_execution), [speculative execution](https://en.wikipedia.org/wiki/Speculative_execution) and multilevel caches）
  - 吞吐&延迟
    - GPU: throughput-optimized high throughput processor
      - designed for massive levels of parallelism and high throughput, at the cost of medium to high instruction latency
    - CPU: latency-optimized low latency processor
      - 前端消耗大
  - 吞吐数据 (these numbers are from 2021)：
    - The Nvidia Ampere A100: 9.5 TFLOPS for 32-bit precision
    - Intel 24-core processor: 0.66 TFLOPS for 32-bit precision 
  - Scaling Law
    - GPU: Huang's law
    - CPU: Moore's law
      - higher clock rate trend for CPU slowed in 2003: energy consumption & heat dissipation
      - ![image-20250226023725815](./GPU/image-20250226023725815.png)


![CPU-GPU](./GPU/CPU-GPU.png)

![image-20221103003942622](./GPU/CPU-GPU-2.png)

* GPU and CPU
  * sequential parts on CPU, numerical intensive parts on GPU



#### GPU Architecture

##### Intro

* SM之上的高层封装
  * GPC、TPC（纹理处理clusters）
  * V100: 6 GPC * 14 SM = 84 SM
  * A100: 8 GPC * 16 SM = 128 SM

![image-20250402120227882](./GPU/image-20250402120227882.png)

![image-20250316033709934](./GPU/image-20250316033709934.png)

##### GPU Compute Architecture

* GPU内部很多functional units:
  * SMs(Streaming Multiprocessors)，一个SM可以schedule多个block，但同一时间只能执行一个
  * ![image-20250404010805785](./GPU/image-20250404010805785.png)
  
* SP (Streaming Processor) <-> CUDA Core<->Thread

  * 资源：
    * registers & local memory

    * cuda core

    * tensor core

  * Tensor core相比CUDA core，实现了MMA operations，支持2:4 sparsity，支持in8和int4，更高效

* SM <-> Thread Block pool

  * 资源：
    * N*SP
    * warp scheduler
    * control unit resources
    * register
    * shared memory(scratchpad)
  * A set of CUDA cores
  * thread之间可同步，可通过shared memory通信

* Device <-> Grid

  * 资源：Global memory

  * subpartition

* SM可以看做GPU的心脏（对比CPU核心），register和shared memory是SM的稀缺资源。CUDA将这些资源分配给所有驻留在SM中的threads。因此，这些有限的资源就使每个SM中active warps有非常严格的限制，也就限制了并行能力。

  * 每个SM包含的SP数量依据GPU架构而不同，Fermi架构GF100是32个，GF10X架构是48个，Kepler架构是192个，Maxwell架构是128个，Turing架构是64个。相同架构的GPU包含的SM数量则根据GPU的中高低端来定。

![0f4c3f5e-1d1c-4556-8c7e-2725cc82d2df_971x593](./GPU/0f4c3f5e-1d1c-4556-8c7e-2725cc82d2df_971x593.webp)

##### GPU Memory Architecture

![image-20250226190716305](./GPU/image-20250226190716305.png)

<img src="./GPU/image-20250502161610622.png" alt="image-20250502161610622" style="zoom:50%;" />

* Registers
  * 65536 per SM (A100/H100)
    * 256KB Register File for GA10x、A100 (65536 * 4B)
    * 512KB Register File for H100
  * allocated to cores dynamically depending on the requirement of the threads
  * private to the threads
* Constant Caches
  * cache constant data used by the code executing on the SM
* **Shared Memory (SRAM)**
  * a small amount of fast and low latency on-chip programmable SRAM memory
  * 192KB of on-chip SRAM per each of 108 SMs (A100)
    * 192*108=20MB
    * 128*82=10,496  for GA10x
  * Usage: 优化threads共享的访存、as a synchronization mechanism between threads executing within a block
* L1 Cache
  * each SM
  * cache frequently accessed data from L2 cache
* L2 Cache
  * shared by all SMs
  * 作为Global Memory的Cache
* **Global Memory (HBM)**
  * SMs share a high capacity and high bandwidth DRAM
  * 80 GB high bandwidth memory (HBM) with bandwidth of 3000 GB/s (H100)

![image-20250224172901222](./GPU/image-20250224172901222.png)

![SM](./GPU/SM.png)

![memory-hierarchy](./GPU/memory-hierarchy.png)

![memory-hierarchy-1](./GPU/memory-hierarchy-1.png)

* More registers than L1 cache
  * half gemm用32位寄存器做累加，input/output用16位寄存器


![gpu-memory-latency](./GPU/gpu-memory-latency.png)

* SM片上单元比L2快3倍，比Global Memory快几十倍 

```c++
cudaMallocManaged()     不注意的话开销大
cudaMalloc()       分配显存
cudaMemcpyHostToDevice
```

* 编译器决定kernel内定义的变量是分配在寄存器上（**没有超过上限的标量**）还是per-thread local memory上
* 寄存器之间的值不一定是私有的，可以shuffle

![shared-memory](./GPU/shared_memory.png)

###### CUDA视角

![image-20250404200229175](./GPU/image-20250404200229175.png)

###### Unified Memory

* Pascal之后有硬件支持
* 解决cpu&gpu空间均需访问某一tensor的问题
  * 本质上是通过地址映射，让GPU可以访问某块CPU Memory

* 两种实现：
  * 基于page fault
    * 这个地址在统一的内存空间里，GPU和CPU都可以使用，但物理上数据可以不在它被访问的设备里，这时会产生page fault（缺页错误），对这个错误的处理就是把数据拷贝到需要访问它的设备或主机内存里，这个操作是透明的（自动执行）。
    * https://on-demand.gputechconf.com/gtc/2018/presentation/s8430-everything-you-need-to-know-about-unified-memory.pdf
    * https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
  * 用load/store，UVA（Unified Virtual Addressing）或者zero copy access
* e.g. bitsandbytes paged optimizer
  * https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/docs/source/explanations/optimizers.mdx
  * https://github.com/bitsandbytes-foundation/bitsandbytes/issues/962
  * Compared to CPU offloading, a paged optimizer has zero overhead if all the memory fits onto the device and only some overhead if some of memory needs to be evicted. For offloading, you usually offload fixed parts of the model and need to off and onload all this memory with each iteration through the model (sometimes twice for both forward and backward pass).

* 精讲UnifiedMemory的博客 https://blog.csdn.net/weixin_41172895/article/details/115403922
  * 使用了虚拟内存(cudaMallocManaged)，降低代码复杂度，系统会将大于gpu内存的空间自动放到cpu上，内存申请**cudaMemPrefetchAsync**
  * **cudaMemAdvise**
    * cudaMemAdviseSetReadMostly 及 cudaMemAdviseUnSetReadMostly
      这两个是cudaMemAdvise的Flag，用来为某个device设定/解除内存空间ReadMostly的特性，device所指的设备可以有一个只读副本而不发生数据迁移，当两端没有写入时，两个副本的数据是一致的。
    * cudaMemAdvise(cudaMemAdviseSetAccessedBy), gpu上有的直接使用，cpu上就直接pci访问，什么时候搬运到gpu可以自行指定



##### GPU Execution Model

> GPU-Mode Lecture 4 https://www.youtube.com/watch?v=lTmYrKwjSOU

* Grid
  * kernel launch grid of threads
  * All threads execute the same code: Single program multiple-data (SPMD)
  * Threads are hierarchically organized into **grid blocks** & **thread blocks**
  * threads in same block
    * can access **the same shared mem**
    * up to 1024 threads can be in a thread block
      * Hopper架构：每个维度上的最大线程数分别是 1024（x 维度）、1024（y 维度）和 64（z 维度），且乘积不能超过1024
    * threads can be scheduled in any order

![image-20250226194634249](./GPU/image-20250226194634249.png)

**A warp is the basic schedule unit in kernel execution**

* block按32 cdiv，由多个warps执行
  * 一个warp是successive 32 threads in a block
  * Threads如果不能被32整除，余数占据one more warp
* SIMT，一个时钟周期内，一个warp被调度到一个SM上，内部32个线程执行相同的指令

  * execution on a set of cores called a **processing block**.

> AMD wavefronts: 64 threads (可 配置为更低）

* Instructions are SIMD synchronous within a warp

  * 一个warp中的线程执行同一指令

    * e.g. 【code/reduce.cu】`reduce3()`
      * 各种优化技巧，包括unrolling、algorithm cascading
      * each thread should sum O(log n) elements
    * [Independent Thread Scheduling](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture): Volta架构之后，可以执行不同指令，但不同时

  * **Latency hiding: having multiple warps on the SM allows warps to compute while others wait**

    (i.e. the memory transfer “+” compute becomes a max(…, …))

    → roofline model

  * 对于control flow，可能是周期T一半的线程执行if语句，周期T+1另一半的线程执行else语句

* Instructions will be issued to execution units by warp.

  * warp scheduler: Decode and schedule the next instructions
  * Latency is caused by not able to issue next instruction to execution unit

* **Warp State**

  * Active: warps inside the pool which has non-exiting threads
  * Eligible: active warps that are not stalled
  * Issued: a single eligible warp that the warp scheduler choose to issue one or more instructions on this cycle

* threading blocks、warp、processing units、SM的关系
  * threading blocks映射到SM
  * warp由一个processing unit执行
  * 一个threading block的thread数量通常是32的倍数（对应N个warp）
  * thread block从x维开始按顺序分配给多个warp
    * ![image-20250404020117023](./GPU/image-20250404020117023.png)
  
* 限制：
  * **Nvidia H100**:
    * each SM can handle 32 blocks, 64 warps (i.e., 2048 threads), and 1024 threads per block.
    * max 1536 threads assignable to one SM

* 一个SM有一个thread block pool，一个thread block有多个warp，一个warp scheduler 16个warp
  * [How to choose how many threads/blocks to have?](https://forums.developer.nvidia.com/t/how-to-choose-how-many-threads-blocks-to-have/55529)
  * The only thing that really matters for occupancy and if performance depends on occupancy is warps. You want to have as close to 64 active warps as possible, all other factors being equal.
  * very small block sizes (e.g. 32 threads per block) may limit performance due to occupancy.
  * Very large block sizes for example **1024 threads per block, may also limit performance**
    * if there are resource limits (e.g. registers per thread usage, or shared memory usage) which prevent 2 threadblocks (in this example of 1024 threads per block) from being resident on a SM
    * **不能整除1536**
  * 推荐值：one thread block, **128~512 threads**
    * 最小取64，通常取128、256
  
      * SM的倍数
  
      * 32的倍数， [in depth coverage of SMs and warps](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation)
  
* 一些单元：
  * SFU: special function unit
  * Load/Store memory

###### warp divergence

* if、loop可能导致divergence

* ![image-20250404021824460](./GPU/image-20250404021824460.png)
  * divergence时，不能做sync操作
* ![image-20250404022026152](./GPU/image-20250404022026152.png)

* ![image-20250404022209555](./GPU/image-20250404022209555.png)

###### Pipeline

* ASYNC in SMEM and ILP in RMEM
  * ![image-20250515020212872](./GPU/image-20250515020212872.png)

##### GPU Network

> 基于 [ICI(tpu)](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)/[RoCE](https://en.wikipedia.org/wiki/InfiniBand)/IB 实现高速网络互联
>
> nvlink 2025现状介绍 https://zhuanlan.zhihu.com/p/28229263860
>
> IB介绍：https://network.nvidia.com/pdf/whitepapers/Intro_to_IB_for_End_Users.pdf

* NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB
  (50 GB/s).
* 内存 - pin memory - 显存 - GPU
  * 通常由CPU负责调度
  * pin memory和内存的传输：由GPU上的DMA负责调度
  * 内存通常是显存的2倍以上比较合理
* 硬盘 - 显存：
  * GPU Direct Storage
* 网络 - 显存： RDMA
* PCIe / NVLINk 与CPU Chipset交互
  * nvlink的带宽 > GPU-CPU offload带宽


![nvlink](./GPU/nvlink.png)

#### Execution of a Kernel on the GPU

* H2D
  * 有可能直接从host memory读：[EMOGI: Efficient Memory-access for Out-of-memory Graph-traversal in GPUs](https://arxiv.org/pdf/2006.06890.pdf)
* Scheduling thread blocks on SMs
  * ![image-20250224192038955](./GPU/image-20250224192038955.png)
  * waitlisted blocks
  * Compute Capability 3.x (Kepler)
    - **4 warp schedulers per SM**  ([Why only one of the warps is executed by a SM in cuda?](https://stackoverflow.com/questions/13463440/why-only-one-of-the-warps-is-executed-by-a-sm-in-cuda))
    -  those SHARE one instruction (but Volta+ does have per-thread program counter)
    - Dispatch 1 or 2 instructions per warp scheduler
* Single Instruction Multiple Threads (SIMT) and Warps
  * 参考 「GPU Execution Model」
* Warp Scheduling and Latency Tolerance
  * **Zero-overhead Scheduling**
    * As each thread in each warp has its own set of registers, there is no overhead for the SM to switch from executing one warp to another. 
    * context switching in CPU is expensive because the CPU needs to save the registers into main memory, and restore the state of the other process
    * --> 通过大量warp来hide memory latency
* Copying of Result Data From Device to Host Memory

#### 计算：TensorCore

* Intro

  * Volta, Turing, Ampere 开始的GPU架构
  * **Nvidia Tensor cores are dedicated to performing general matrix multiplication (GEMM) and half-precision matrix multiplication and accumulation (HMMA) operations.** In short, GEMM performs matrix operations in the format of A*B + C, and HMMA converts the operation into the half-precision format.
  * https://resources.nvidia.com/en-us-tensor-core
  * 相比CUDA core，实现了MMA operations，支持2:4 sparsity，支持in8和int4，更高效
    * https://www.wevolver.com/article/tensor-cores-vs-cuda-cores

* paper

  * [NVIDIA Tensor Core Programmability, Performance & Precision](https://arxiv.org/pdf/1803.04014)

  * [Analyzing GPU Tensor Core Potential for Fast Reductions](https://arxiv.org/pdf/1903.03640)

  * Demystifying Tensor Cores to Optimize Half-Precision Matrix Multiply

* Guide:

  * GTC 教程：https://developer.nvidia.com/gtc/2020/video/s21745-vid

  * https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/

  * Q: Does Matmul involve reduction sum? Why can it be done in FP16?

    * A: In tensor core FP16 MAC (Multiply-Accumulate) unit, the accumulation is always done in full precision, which avoids the problem of arithmetic underflow.
    * Reference: https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/

  * To employ Tensor Cores in [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html), the dimensions of a GEMM ([M, K] x [K, N] -> [M, N]) must be multiples of 8.

    * convolution没有限制
    * https://github.com/NVIDIA/apex/issues/221#issuecomment-478084841


  ![image-20250502214129368](./GPU/image-20250502214129368.png)

##### [Speaking Tensor Cores —— GPU Mode Lecture 23](https://www.youtube.com/watch?v=hQ9GPnV0-50)

> Vijay Thakkar & Pradeep Ramani (Representing the CUTLASS Team @NVIDIA)

* Intro
  * Hardware block that accelerates MatMul
  * FMAs accelerate vector dot products - O(N) operations
  * Tensor cores accelerate matrix multiplies - O(N3) operations
  * Increases flop / byte ratio – more temporal and spatial reuse

![image-20250515013418204](./GPU/image-20250515013418204.png)

![image-20250515013430058](./GPU/image-20250515013430058.png)

![image-20250515014503276](./GPU/image-20250515014503276.png)

* 时空重用

![image-20250515014922279](./GPU/image-20250515014922279.png)

* 问题：Ad-hoc partitioning doesn’t scale
  * Problem 1: Complicated Partitioning Patterns
    * Prevent us from writing canonical loops for all MMAs
  * Problem 2: Programmer Managed Asynchrony
    * GPUs require deeply async, managed, producer/consumer software pipelines
    * Feeding the tensor cores constantly is hard – requires managing asynchrony and deep software pipelines
    * With newer architectures like Hopper, even the MMA instruction is asynchronous
    * Concurrency programming - Writing kernels isn’t just about getting the layouts right anymore

![image-20250515015509463](./GPU/image-20250515015509463.png)



#### 通信：NVLink等

* NVLink
  * 高速、低延迟的通用串行总线接口技术，GPU卡间通信，带宽很高
  * H100:
    * 50GB/link
    * 18links
* 多机
  * Ethernet
  * InfiniBand
  * OmniPath
  * RoCE（RDMA over Converged Ethernet）

#### 计算：cuDNN

* Intro

  * cuDNN全称NVIDIA CUDA® Deep Neural Network library， 是一个用于深度神经网络的GPU加速库。

  * cuDNN包含了为神经网络中常见的计算任务提供高度优化的实现。包括前向卷积、反向卷积、注意力机制、矩阵乘法（matmul）、池化（pooling）和归一化（normalization）等。

  * cuDNN的最常见用途是在深度学习框架（如TensorFlow或PyTorch）的开发中。深度学习框架开发者在编写框架时，通常会调用cuDNN，从而几乎不直接与CUDA进行交互。而对于我们使用PyTorch做AI应用的终端用户来说，更没有机会使用cuDNN的。

* cuDNN和CUDA Toolkit的关系

  * CUDA Toolkit不包含cuDNN。CUDA Toolkit是一个更底层的工具包，其中的库是针对的是更基础的操作，比如线性代数中各种矩阵和向量的运算，还有用于文件I/O，支持在GPU上进行高性能文件操作等。而cuDNN是专门为深度学习的各种运算所设计的库，它需要使用CUDA Toolkit中的一些库。

#### 显卡驱动

* 英伟达的显卡驱动程序通常会随CUDA Toolkit一起安装。但是，这个驱动程序是为了开发目的而安装的。这意味着它主要用于开发和调试CUDA应用程序，以帮助开发人员在其工作站上进行开发和测试。这个驱动程序不建议在生产环境中与英伟达的GPU一起使用。在生产环境中，通常需要专门的、经过验证的驱动程序以确保系统的稳定性和性能。

#### 共享卡 —— 如何实现算力和显存隔离

* 隔离方式：时间片 vs 空间
* 隔离级别：不隔离 vs 强隔离 vs 弹性
* 几种隔离技术对比：
  * vGPU(Grid)(Nvidia)：虚拟化；容器化支持不好，license
  * vCuda(腾讯)：cuda hook；性能损耗严重
  * cGPU(Alibaba)：ioctl；损耗小，硬隔离，侵入内核（机器容易坏）
  * MPS(Nvidia)：thread；显存隔离，故障隔离不好
    * 进程级别，进程数量会受限于显存
  * MIG(~A100)：sm/global memory；硬件层面隔离



* 3 ways to accelerate applications:
  * Libraries    
  * OpenACC (add directive, for applications like HPC)
  * Programming Languages



#### DCGM

https://developer.nvidia.com/dcgm

https://on-demand.gputechconf.com/gtc/2018/presentation/s8505-gpu-monitoring-and-management-with-nvidia-data-center-gpu-manager-dcgm-v2.pdf

* Note:
  * 监控 PCIE-RX ---> H2D 带宽
  * Health Check

##### nvidia-smi

```shell
nvidia-smi
nvidia-smi --query-gpu=name --format=csv,noheader
```

#### OAM

* OAM是一种异构计算设备的接口协议，对标nvlink.   nvlink是封闭的协议，OAM是若干大厂共建的一个开放协议。
  * 目的：避免每一种卡都有自己的卡间通信标准，尽量做到 （除了NV的卡之外）任何厂家的计算卡都兼容
  * 技术层面， OAM 和 NVSWITCH （对应A100 这代） 基本是对等的 
    - OAM：是 full connect 模式， 各卡独享约 ~80GB/s 带宽 
      - full connect 的结构更为简单，有一些散热和能耗的优势
    - NV switch: 整个switch提供 600GB/s 带宽 
    - 单机八卡时，OAM 和 NV switch 差不多；卡数少时 nvsiwtch 效率高

### 机型 & 硬件 & 精度 & 吞吐 & 故障

#### 精度支持

* Int4
  * Blackwell GPUs will [no longer support int4 tensor cores](https://www.nvidia.com/en-us/data-center/tensor-cores/).

* fp8
  * FP8 GEMM Accumulation Precision in Tensor Cores：
    * After aligning 32 mantissa products by right-shifting based on the maximum
      exponent, the Tensor Core only uses the highest 14 bits of each mantissa product for addition, and truncates bits exceeding this range. The accumulation of addition results into registers also
      employs **14-bit mantissa precision**.


#### 浮点计算精度

* 浮点计算
  * 硬件机制：结合律可能不适用，大量累加的顺序，会有精度差异
    * python: `1.0 + (-1.0 + 1e-17 )`
    
    * 比如reduction操作
    
      * ```python
        # We'll use several small numbers that, when added together first, could show a difference
        numbers = [1e-20] * 10 + [1e20, -1e20]  # 10 small numbers followed by a large positive and negative number
        
        # Sum the list from left to right
        sum_left_to_right_adjusted = sum(numbers)
        
        # Sum the list from right to left
        sum_right_to_left_adjusted = sum(reversed(numbers))
        
        # 0.0 9.999999999999997e-20
        print(sum_left_to_right_adjusted, sum_right_to_left_adjusted)
        ```
    
  * cuDNN：
    * deterministic=True：尽量消除算子底层实现的随机性
    * benchmark=False：仅使用同一种卷积算法
    
  * 算子实现：随机采样

#### 吞吐

* ![image-20250331121025135](./GPU/image-20250331121025135.png)

* INT8加速比：
  * https://github.com/pytorch/ao/pull/748
  * a100加速比2，
  * ![image-20250331133109379](./GPU/image-20250331133109379.png)

#### 新硬件架构

##### Hopper

* Thread Block Group的概念
* WGMMA 的异步能力 https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/



##### Blackwell

https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/



#### Engegy Model

![image-20250331151735124](./GPU/image-20250331151735124.png)

* 《1.1 computing’s energy problem (and what we can do about it)》
* 《PokeBNN: A binary pursuit of lightweight accuracy.》





#### **机型基础

* Nvidia GPU的算力([Compute Capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability)), 只是一个版本号, 用来表示核心架构. 一般用`X.X`的方式表示, 第一位是主版本号, 第二位是次版本号, 如下:

| 架构                                                         | 算力 | 上市时间 | 产品                                                         | NVLink                                                       | NVSwitch                     | PCIe              |
| ------------------------------------------------------------ | ---- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------- | ----------------- |
| [Blackwell](https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-architecture-technical-brief) |      |          |                                                              |                                                              |                              |                   |
| [Ada](https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-V2.02.pdf) |      |          | L40、L4                                                      |                                                              |                              |                   |
| Hopper architecture (霍普)                                   | 9    |          | H100, 训练卡                                                 | 3td-NVLink, SXM2/SXM4900GB/s最多18个                         | 3td-NVSwitch: 900GB/s最多8个 |                   |
| [Ampere architecture ](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) | 8    | 2020     | A100, 训练卡, 80G HBM2e 显存 19.5 TFLOPS Cuda Cores: 6912 Tensor Cores: 432, 108 SMs; | 3td-NVLink, SXM2/SXM3 600GB/s最多12个                        | 2nd-NVSwitch: 600GB/s最多8个 | PCIe Gen4 64 GB/s |
| Turing architecture (图灵)                                   | 7.5  | 2018     | T4, 推理卡, 16GB GDDR6 显存 8.1 TFLOPS Cuda Cores: 2560 Tensor Cores: 320 |                                                              |                              | PCIe Gen332 GB/s  |
| Volta architecture (伏特)                                    | 7    | 2017     | V100, 训练卡, 24G HBM2显存14~16.4 TFLOPSCuda Cores: 5120Tensor Cores: 640 | 2nd-NVLink, SXM2300GB/s最多6个                               | 1st-NVSwitch: 300GB/s最多8个 | PCIe Gen332 GB/s  |
| Pascal architecture (帕斯卡)                                 | 6    | 2016     | P100, 训练卡, 16G HBM2显存 9.3 ~ 10.6 TFLOPSCuda Cores: 3840P40, 训练卡, 24G GDDR5 显存P4, 推理卡, 8G GDDR5 显存 | 1st-NVLink, SXMP100: 732 GB/s160 GB/sP40: 346 GB/sP4: 192 GB/s |                              | PCIe Gen332 GB/s  |
| Maxwell architecture (麦克斯韦)                              | 5    | 2014     | M40, M60                                                     |                                                              |                              |                   |
| Kepler architecture (开普勒)                                 | 3    | 2010     | K10, K20, K40, K80                                           |                                                              |                              |                   |
| Fermi architecture (费米)                                    | 2    | 2010     |                                                              |                                                              |                              |                   |
| Tesla architecture (特斯拉)                                  | 1    | ~        |                                                              |                                                              |                              |                   |

* H20
  * 132 SMs
  * 96GB
* H100 GPU
  * 132 SMs with 64 cores per SM, totalling a whopping 8448 cores.
  * each SM can handle 32 blocks, 64 warps (i.e., 2048 threads), and 1024 threads per block.
* A100
  * GPU
    * GPU Memory：80 GB
    * GPU Memory Bandwidth：2039 GB/s
    * https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
    * 192KB of on-chip SRAM per each of 108 SMs
    * Float32 Tensor Core：156 TFlops
    * Float16 Tensor Core：314 TFlops
    * Float32 CUDA Core：19.5 TFlops
  * CPU：2 socket
    * 内存TB级别
  * Interconnect：
    * NVLink：600GB/s （50GB/s，12 links）
    * PCIe Gen4: 64GB/s

  * 《Dissecting the Ampere GPU architecture via microbenchmarking》
  * 《Nvidia A100 tensor core GPU architecture》
* V100
  * https://datacrunch.io/blog/nvidia-v100-gpu-specs
  * roofline: 125/0.9 =139FLOPS/Byte
* GA10x：RTX 3090, has 82 SMs.
* Each SM in GA10x GPUs contain 128 CUDA Cores, 4 third-generation Tensor Cores, 2 FP64 Cores
  * a 256 KB Register File, and 128 KB of L1/Shared Memory
  * 4*32 FP32 units (one per thread), half  of which know INT32
  * L1 cache and shared memory share hardware (128KB) directly on the SM shmem can be 0/8/16/32/64/100KB
    * L1 Cache the remainder (>=28KB)
  
* ![image-20250404011658421](./GPU/image-20250404011658421.png)



Nvidia GPU 产品根据使用场景不同分为不同的序列:

- GeForce: 用于家庭和个人电脑，包括游戏和娱乐等;
  - **前缀**: 显卡档次与代号. GT: 频率提升版本; GS: GT的缩减版，级别在GT之后; GTX: 一般可以理解为GT eXtreme，代表了极端、极致的意思; GTS: GTX的缩减版，级别在GTX之后. RTX-> GTX > GTS > GT > GS
  - **数字**: 例如1060, 10代表的是第几代, 6代表显卡性能档次的定位
  - **后缀**: SE的意思是阉割版, TI表示增强版, M表示移动端, LE表示
- Quadro: 用于工业渲染、艺术设计，工作站等场合
- Tesla: 用于科学计算，深度学习加速等场景, 对于15年以后的产品, 一般或以省去Tesla, 或用NVIDIA代替, 如P100, T4, V100, A100等.

GPU的Compute Capability与CUDA版本不是同一回事, 后者是开发套件的版本. 

![h100](./GPU/h100.png)



* [A10 v.s. A10G](https://www.baseten.co/blog/nvidia-a10-vs-a10g-for-ml-model-inference/)
  * The A10 is an Ampere-series datacenter GPU well-suited to many model inference tasks, such as running seven billion parameter LLMs. However, AWS users run those same workloads on the A10G, a variant of the graphics card created specifically for AWS. The A10 and A10G have somewhat different specs — most notably around tensor compute — but are interchangeable for most model inference tasks because they share the same GPU memory and bandwidth, and most model inference is memory bound.
  * the A10 prioritizes tensor compute, while the A10G has a higher CUDA core performance
  * 根据ops_to_byte分析是compute bound还是memory bound
    * arithmetic_intensity (Llama 2 7B, Single-Headed Attention Operation)
          ~= total compute / total memory movement
          = 4d(N^2) + 3N^2 ops / 8N^2 + 8Nd bytes
          = 62 ops/byte
  * [A guide to LLM inference and performance](https://www.baseten.co/blog/llm-transformer-inference-guide/) TODO

##### 显存一览

<img src="./GPU/image-20250507025921271.png" alt="image-20250507025921271" style="zoom:50%;" />

* H20: 96G

##### 显存带宽

* NVIDIA H100 (SXM5 with HBM3) 带宽估算:

  - 显存类型 : HBM3
  - 显存位宽 (Memory Interface Width) : 5120-bit
  - HBM3 等效数据速率 (Effective Data Rate per pin) : 约 5.2 GT/s (GigaTransfers per second) 或更高，具体取决于 SKU，以达到约 3.35 TB/s 的总带宽。
  - 带宽计算 :
    带宽 (GB/s) = (显存位宽 (bits) / 8 bits/Byte) * 等效数据速率 (GT/s)

    代入 H100 的数值 (以 5.2 GT/s 为例进行说明)： (5120 bit / 8 bits/Byte) * 5.2 GT/s = 640 Bytes/Transfer_cycle * 5.2 * 10^9 Transfers/sec = 3328 * 10^9 Bytes/sec = 3328 GB/s = 3.328 TB/s

* G80 GPU （2006）

  * 384-bit memory interface（数据总线位数）, 900 MHz DDR
  * 384 * 1800 / 8 = 86.4 GB/s
    * 由于DDR的时钟脉冲上升沿和下降沿都传输数据，因此倍增系数为2

#### 硬件降频

* 聊一聊英伟达GPU的降频问题 https://zhuanlan.zhihu.com/p/13866293937

#### 故障率 0.5% 以上

* https://www.pugetsystems.com/labs/articles/puget-systems-most-reliable-hardware-of-2024

### CUDA

> Nvidia Lecture 1: Accelerating Applications with CUDA C/C++
>
> [课程网页](https://courses.nvidia.com/courses/course-v1:DLI+C-AC-01+V1/courseware/85f2a3ac16a0476685257996b84001ad/9ef2f68fb10d40c5b54b783392938d04/?activate_block_id=block-v1%3ADLI%2BC-AC-01%2BV1%2Btype%40sequential%2Bblock%409ef2f68fb10d40c5b54b783392938d04)

#### Intro

* CUDA：Compute Unified Device Architect
  * CUDA C: extends ANSI C with minimal new  syntax

* CUDA accelerates applications drastically with little effort, has an ecosystem of highly optimized libraries for [DNN](https://developer.nvidia.com/cudnn), [BLAS](https://developer.nvidia.com/cublas), [graph analytics](https://developer.nvidia.com/nvgraph), [FFT](https://developer.nvidia.com/cufft), and more, and also ships with powerful [command line](http://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview) and [visual profilers](http://docs.nvidia.com/cuda/profiler-users-guide/index.html#visual).
* CUDA supports many, if not most, of the [world's most performant applications](https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/catalog/?product_category_id=58,59,60,293,98,172,223,227,228,265,487,488,114,389,220,258,461&search=) in, [Computational Fluid Dynamics](https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/catalog/?product_category_id=10,12,16,17,19,51,53,71,87,121,124,156,157,195,202,203,204,312,339,340,395,407,448,485,517,528,529,541,245,216,104,462,513,250,492,420,429,490,10,12,16,17,19,51,53,71,87,121,124,156,157,195,202,203,204,312,339,340,395,407,448,485,517,528,529,541,245,216,104,462,513,250,492,420,429,490,10,12,16,17,19,51,53,71,87,121,124,156,157,195,202,203,204,312,339,340,395,407,448,485,517,528,529,541,245,216,104,462,513,250,492,420,429,490&search=), [Molecular Dynamics](https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/catalog/?product_category_id=8,57,92,123,211,213,237,272,274,282,283,307,325,337,344,345,351,362,365,380,396,398,400,435,507,508,519,8,57,92,123,211,213,237,272,274,282,283,307,325,337,344,345,351,362,365,380,396,398,400,435,507,508,519,8,57,92,123,211,213,237,272,274,282,283,307,325,337,344,345,351,362,365,380,396,398,400,435,507,508,519,8,57,92,123,211,213,237,272,274,282,283,307,325,337,344,345,351,362,365,380,396,398,400,435,507,508,519&search=), [Quantum Chemistry](https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/catalog/?product_category_id=8,57,92,123,211,213,237,272,274,282,283,307,325,337,344,345,351,362,365,380,396,398,400,435,507,508,519,8,57,92,123,211,213,237,272,274,282,283,307,325,337,344,345,351,362,365,380,396,398,400,435,507,508,519&search=), [Physics](https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/catalog/?product_category_id=6,24,116,118,119,135,229,231,372,373,392,393,489,493,494,495,496,497,498,67,170,216,281,6,24,116,118,119,135,229,231,372,373,392,393,489,493,494,495,496,497,498,67,170,216,281,6,24,116,118,119,135,229,231,372,373,392,393,489,493,494,495,496,497,498,67,170,216,281,6,24,116,118,119,135,229,231,372,373,392,393,489,493,494,495,496,497,498,67,170,216,281,6,24,116,118,119,135,229,231,372,373,392,393,489,493,494,495,496,497,498,67,170,216,281&search=) and HPC.

#### Programming Model

##### CUDA Thread Hierarchy

* Programming model: SIMT
  * 三层抽象：grid, block, thread

* Kernel Execution

  * configuration ~ grid
  * block: Each block is executed by one SM and does not migrate.
    * Several concurrent blocks can reside on one SM depending on block’s memory requirement and the SM’s memory resources.

  * thread：uniquely identified by threadIdx和blockIdx
    * Idea: map threads to multi-dimensional data
    * At a high level, execution configuration allows programmers to specify the **thread hierarchy** for a kernel launch, which defines the number of thread groupings (called **blocks**), as well as how many **threads** to execute in each block.


![SIMT](./GPU/SIMT.png)

![image-20250224190231769](./GPU/image-20250224190231769.png)

```
gridDim.x: num of blocks
blockDim.x: num of threads in a block
blockIdx.x: block index
threadIdx.x: thread index
```

* [Grid-Stride Loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)
  * By using a loop with stride equal to the grid size, we ensure that all addressing within warps is unit-stride, so we get [maximum memory coalescing](https://developer.nvidia.com/blog/parallelforall/how-access-global-memory-efficiently-cuda-c-kernels/), just as in the monolithic version.
  * 所有warp同步寻址都是在一个grid里，这是最大显存聚合


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

##### 



##### Warp Specialization

《Singe: Leveraging Warp Specialization for High Performance on GPUs》

#### Host and Device Code

![workflow](./GPU/workflow.png)

* 细节：
  * CUDA Kernel argument space has a max limit of 4KB
  * [cudaDeviceSynchronize只需要在使用cudaStream时使用](https://stackoverflow.com/questions/11888772/when-to-call-cudadevicesynchronize)，平时“Although CUDA kernel launches are asynchronous, all GPU-related tasks placed in one stream (which is the default behavior) are executed sequentially.”

![image-20250226193631721](./GPU/image-20250226193631721.png)

![image-20250224190443112](./GPU/image-20250224190443112.png)

![image-20250224190455058](./GPU/image-20250224190455058.png)

![image-20250404200327944](./GPU/image-20250404200327944.png)



#### CUDA Compiler

[**NVIDIA CUDA Compiler**](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html), [documentation](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)

* nvcc (NVIDIA C compiler) is used to compile kernels into PTX
  * [`arch` flag](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation)
    * 使用 `-gencode arch=compute_XX,code=sm_YY` 这样的语法，你可以让 nvcc 生成针对虚拟架构 compute_XX 的 PTX 代码，并且同时为真实架构 sm_YY 生成 SASS。
    * PTX用于前向兼容性，SASS用于最佳性能

  * [virtual architecture features](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list) 

* Parallel Thread Execution (PTX) is a low-level VM & instruction set
* graphics driver translates PTX into executable binary code (SASS)

#### CUDA Libraries (called from device)

* CUTLASS、Thrust、CUB

#### CUDA Streams

> https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf

* 核心是一个GPU stream上执行的kernel，必定**按issue提交的顺序执行**
  * Kernels within any single stream must execute in order
  * Kernels in different, non-default streams can interact concurrently
  * The default stream is special: it blocks all kernels in all other streams 
    * 这一规则有副作用，因此推荐用non-default streams

* e.g.
  * nbody-raw.cu -> nbody-optimized.cu

#### CUDA cooperative group 协作线程组

<img src="./GPU/cooperative-groups.png" alt="cooperative-groups.png" style="zoom:100%;" />

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



![shuffle](./GPU/shuffle.png)







#### Case Study: Matmul

> Snippets/gpu-ops/triton-matmul.py

* Cutlass implementation of matrix multiplication on A100
  * https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf

#### Case Study: Reduce

> GPU Mode Lecture 9: Reductions https://www.youtube.com/watch?v=09wntC6BT5o
>
> cuda lecture：https://www.youtube.com/watch?v=D4l1YMsGNlU&t=1763s

* 什么是reduction

  * min, max, argmax, argmin, norm, sum, prod, mean, unique

  * ```python
    def reduce(data, identity, op):
        result = identity
        for element in data:
            result = op(result, element)
        return result
    ```

  * 应用：
    * Mean/Max pooling
    * Classification: Argmax
    * Loss calculations
    * Softmax、normalization

* parallel reduction tree
  * inactive warps
  
* resources/cuda-reduction.pdf
  
  * Parallel Reduction
  * Problem: Global Synchronization
    * Solution: decompose into multiple kernels
      * Kernel launch serves as a global synchronization point
      * Kernel launch has negligible HW overhead, low SW overhead
    * Solution: Kernel Decomposition
      * Recursive kernel invocation
  
* use [CUB](https://nvidia.github.io/cccl/cub/) to accelerate the reduce operations

### CUTLASS

> Open source: https://github.com/NVIDIA/cutlass
>
> • Documentation: https://github.com/NVIDIA/cutlass#documentation
>
> • Presented: [GTC’18](https://on-demand.gputechconf.com/gtc/2018/presentation/s8854-cutlass-software-primitives-for-dense-linear-algebra-at-all-levels-and-scales-within-cuda.pdf), [GTC’19](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf), [GTC’20](https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf), [GTC’21](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31883/), [GTC'22](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41996/) , [GTC’22](https://www.nvidia.com/en-us/on-demand/session/gtcfall22-a41131/), [GTC’23](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51413/), [GTC’24](https://www.nvidia.com/en-us/on-demand/session/gtc24-s61198/)
>
> • Come join the CUTLASS channel in our discord: https://discord.gg/CVEJqWtU

> [GPU Mode Lecture 15](https://www.youtube.com/watch?v=G6q719ck7ww&t=4s) by Eric Auld

#### Intro

![image-20250518040354290](./GPU/image-20250518040354290.png)

![image-20250518041015826](./GPU/image-20250518041015826.png)

#### 编程模型

##### Layout

* layout
  * shape
  * Stride

![image-20250518041548300](./GPU/image-20250518041548300.png)

![image-20250518042222533](./GPU/image-20250518042222533.png)

![image-20250518043150350](./GPU/image-20250518043150350.png)

##### Tile

![image-20250519021000133](./GPU/image-20250519021000133.png)

![image-20250520014916050](./GPU/image-20250520014916050.png)

![image-20250520015733984](./GPU/image-20250520015733984.png)

![image-20250520020742235](./GPU/image-20250520020742235.png)

#### WGMMA

https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/

#### [Speaking Tensor Cores —— GPU Mode Lecture 23](https://www.youtube.com/watch?v=hQ9GPnV0-50)

> **看到 14:20**

* Intro
  * CUDA C++ Template Library for High Performance Linear Algebra
  * Tensor core computations at all scopes and scales, **decomposed into their “moving parts”**
  * **Provides a native tile-based programming model for GPU kernels**
  * ![image-20250515020823498](./GPU/image-20250515020823498.png)

* 特点：

  * Public Tensor Core programming model for NVIDIA GPUs
    * Serve as a production grade example for the world
  * Extreme focus on developer productivity for custom kernels
    * Allow customizing any layer in the hierarchy while preserving composability with other layers
  * If it compiles, it will be correct – actionable static assert messages otherwise
    * Static asserts at every layer to ensure layout and dispatch compatibilities
  * Single, clear points of customization and dispatch to flatten the learning curve
    * Reduce API surface area with fewer named types
  * ![image-20250515021614827](./GPU/image-20250515021614827.png)

  * ![image-20250515021650578](./GPU/image-20250515021650578.png)





### CuTe

> CuTe包含于CUTLASS 3
>
> https://github.com/NVIDIA/cutlass/tree/main/include/cute

![image-20250515021815814](./GPU/image-20250515021815814.png)



### Triton

#### Intro

> https://openai.com/index/triton/

* Triton v.s. CUDA
  * pythonish
  
  * easy to write and debug

  * 二者均生成PTX
  
  * triton的核心思路：block操作，而不是SIMT
  
  * |                          | **CUDA** | **TRITON** |
    | ------------------------ | -------- | ---------- |
    | Memory Coalescing        | Manual   | Automatic  |
    | Shared Memory Management | Manual   | Automatic  |
    | Scheduling (Within SMs)  | Manual   | Automatic  |
    | Scheduling (Across SMs)  | Manual   | Manual     |
  
* Triton是OpenAI 推出的以python为编程语言基础，专门为深度学习研发和高性能计算而设计的编程语言和编译器，旨在简化和优化GPU编程的复杂操作，降低高性能优化的门槛。它允许开发者在Triton框架内更灵活地编写和优化自定义的算子（operators）或处理复杂的数据流程。
  * 生成PTX（Cuda Assembly）而不是cuda
  * Triton的初期版本以CUDA为起点而开发，为没有CUDA基础的编程者提供快速编写高效CUDA kernel的方案，而随着迭代已逐渐支持其他芯片和编程工具，如AMD的ROCm，并在继续支持其他的芯片，如Intel的CPU。
  * During the compilation, the Triton compiler tries to use clever tricks to **rearrange the parts of your program**
  * 利用ptx汇编可以将triton降级为ptx代码，在cuda上直接运行以达到极致计算性能的优化，Triton提供了块指针非常便捷的实现FA，对GPU IO感知类的实现进行了充分的支持。

#### 和 numba 对比

* numba：
  * python
  * 可以传入矩阵shape
  * debug
  * NUMBA_ENABLE_CUDASIM=1

#### Op优化情况

>  snippets/gpu-triton.py

* Square

> Snippets/gpu-triton-ops.py

* Matmul

![image-20250302192301222](./GPU/image-20250302192301222.png)

* Index select backward
  * ![image-20250423115630485](./GPU/image-20250423115630485.png)
  * 



#### Programming Model

* Triton v.s. CUDA
  * 只感知 blocks <-> CUDA两层抽象，blocks 、threads
    * Note on jargon: In triton lingo, each kernel (which processes a block) is called a "program". Therefore, "block_id" is often called "pid" (short for "program id"), but it's the same.
  * 处理tensor <-> 处理scalar
    * **All** operations in triton kernels are vectorized: Loading data, operating on data, storing data, and creating masks.
  * 不感知shared memory

![image-20250302005036701](./GPU/image-20250302005036701.png)

* one important limitation of Triton is that **each block must have a power-of-two number of elements**, so we need to internally “pad” each row and guard the memory operations properly if we want to handle any possible input shapes:

#### 优化原理

* shared memory
  * data can be automatically **stashed to shared memory by looking at the operands of computationally intensive block-level operations** (e.g., `tl.dot`)—and **allocated/synchronized using standard liveness analysis techniques.**
  * ![image-20250417144519785](./GPU/image-20250417144519785.png)

* parallel
  * (1) across SMs by executing different kernel instances concurrently
  * (2) within SMs by analyzing the iteration space of each block-level operation and partitioning it adequately across different SIMD units
  * ![image-20250417144848673](./GPU/image-20250417144848673.png)



#### Debugging

> snippets/gpu-triton-debugging.py

* `TRITON_INTERPRET=1 python interpret_triton_square.py`
  * 原理：CPU上运行，模拟GPU运行

* crash the kernel then get all the information

#### 经验和细节

* triton autotune目前对dynamic shape的支持不好，性能较差，原因是autotune会对每个新shape重新tune

### NCCL

> GPU Mode Lecture 17 NCCL: https://www.youtube.com/watch?v=T22e3fgit-A

#### Intro

> https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html

![image-20250515030141689](./GPU/image-20250515030141689.png)

* 应用于DDP
  * AllReduce gradients



#### 通信库

* allreduce && reducescatter ops are not actually implemented via ring reductions in NCCL in 2024. 
  * [double binary trees](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/) are used for sufficiently large allreduce ops
  * more recently a variant of Bruck’s algorithm is used for reduce-scatter calls in the [most recent release of NCCL](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-23-4.html#rel_2-23-4).
  * the use of [SHArP](https://network.nvidia.com/pdf/solutions/hpc/paperieee_copyright.pdf) in infiniband switches produces [indeterministic garbage](https://github.com/NVIDIA/nccl/issues/1497) under a scope I am ill-equipped to elaborate on.

* Also, none of this paragraph applies for any serious frontier lab, because all of them use more efficient collective communications libraries, e.g. [HFReduce](https://arxiv.org/pdf/2408.14158#page=6), [MSCCL++](https://github.com/microsoft/mscclpp), [NCCLX](https://www.reddit.com/r/MachineLearning/comments/1eb4xn4/p_ncclx_mentioned_in_llama3_paper/), etc. So, you know, I am just making unprincipled simplifications here, don’t read too hard into the assumptions or you’ll crack them like eggshells.

### GPU优化

#### Overview

> Nvidia Guide: https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html
>
> Getting good occupancy – balance resources

##### Roofline Model

* H20: 
  * 4.8 TB/s
  * FP32: 672 TFLOPS
  * FLOPS/Byte = 140

![image-20250404194640272](./GPU/image-20250404194640272.png)

![image-20250404195020449](./GPU/image-20250404195020449.png)

##### optimization workflow

![optimization-workflow](./GPU/optimization-workflow.png)

##### 细节

● Have 82 SM → many blocks = good

​	(for comparison Jetson Xavier has 8 Volta SM)

● Can schedule up to 1536 threads per SM

​	→ power of two block size <512 desirable

​	(some other GPUs 2048)

● Avoid divergence to execute an entire warp (32 threads) at each cycle

● Avoid FP64/INT64 if you can on Gx102 (GeForce / Workstation GPUs)

● Shared Memory and Register File → limits number of scheduled on SM

(use __launch_bounds__ / C10_LAUNCH_BOUNDS to advise compiler of # of threads for register allocation, but register spill makes things slow) 

● Use `torch.cuda.get_device_properties(<gpu_num>)` to get properties (e.g. max_threads_per_multi_processor)

​	[even more in CUDA than in PyTorch](https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__DEVICE_g5aa4f47938af8276f08074d09b7d520c.html)

#### Literature Review

* 访存瓶颈
  * compute speed has out-paced memory speed [61, 62, 63], and most operations in Transformers are bottlenecked by memory accesses [43]. 【FlashAttention】

#### 写好GPU程序的难点

* 思路

  * "if you do not care about performance, parallel programming is very easy"

    * designing parallel algorithms in practice harder than sequential algorithms
      * e.g. parallelizing recurrent computations requires nonintuitive thinking (like prefix sum)

    * speed is often limited by memory latency/throughput (memory bound)
      * e.g. llm token by token效率低，需要batching


  * perf of parallel programs can vary dramatically based on input data characteristics

  * not all apps are "embarassingly parallel" - synchronization imposes overhead (waits)

* 性能

  * Memory transfers from DRAM must be *coalesced* into large transactions to leverage the large bus width of modern memory interfaces.
  * Data must be manually stashed to SRAM prior to being re-used, and managed so as to minimize shared memory bank conflicts upon retrieval.
  * Computations must be partitioned and scheduled carefully, both across and within Streaming Multiprocessors (SMs), so as to promote instruction/thread-level parallelism and leverage special-purpose ALUs (e.g., tensor cores).

#### [CUDA Performance Checklist (GPU Mode Lecture 8)](https://www.youtube.com/watch?v=SGhfUhlowB4)

> https://docs.google.com/presentation/d/1cvVpf3ChFFiY4Kf25S4e4sPY6Y5uRUO-X-A4nJ7IhFE/edit?slide=id.p#slide=id.p

* Coalesced Global Memory Access

* Maximize occupancy

* *Understand if memory or compute bound*

* Minimize control divergence

* Tiling of reused data

* Privatization
  * local copy, avoid hitting global memory

* Thread Coarsening
  * compute bound通常每个thread做尽可能少的事情
  * memory bound每个thread做更多事情

* Bank conflicts
  
* Latency hiding
* *Rewrite your algorithm using better math*
  * Use high level language to write GPU kernels.


* Nsight System & Nsight Compute
* Use existing libraries, which are highly optimized, e.g. cublas, cudnn. 

* Choose the right metric:
  * GFLOP/s: for compute-bound kernels
  * Bandwidth: for memory-bound kernels

#### SM Occupancy

* **SM Occupancy：the ratio of the number of warps assigned to an SM to the maximum number it can support**
  * 提升 SM 占用率的关键通常在于 降低每个 Block 对 SM 资源的（寄存器、共享内存）需求 ，使得 SM 能够同时调度运行更多的 Block（以及更多的 Warp）。

* 限制因素：主要是资源约束
  *  **每个thread的register数量**
    - each SM has 65536 registers. To execute 2048 threads simultaneously, each thread can have a maximum of 32 registers (65536/2048 = 32). If a kernel needs 64 registers per thread, we can only run 1024 threads per SM,
    - --> resulting in 50% occupancy.
  *  每个SM的共享内存 / **Shared Memory per Block**
  *  **SM能够同时处理的线程块数量**
     - block size太小的情形
       - block size=32，总共需要执行 2048 个线程。因此总共需要 2048/32 = 64 个线程块来容纳这 2048 个线程
       - **每个 SM 在同一时刻最多只能处理 32 个 thread block**
       - --> 50% SM Occupancy

> https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html

![image-20250502214010222](./GPU/image-20250502214010222.png)

* 充分利用 SM
  * cuda occupancy calculator
  * 增大batch size

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
*  Unbalanced workload across blocks
*  Too few blocks launched

**Occupancy Optimization**

* Know the occupancy: NVIDIA Visual profiler / Nsight Compute 
* Adjust resource usage to increase theoretical occupancy
  * Change block size
  * **Limit register usage**
    * **Compiler option –maxregcount=n: per file**

    * **`__launch_bounds__`: per kernel** 
  * Limit shared memory usage.

* Launch enough load-balanced blocks to increase achieved occupancy

```c++
__global__ void
__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
MyKernel(...){
  ...
}
```

#### Warp效率

##### coalesce memory access

>  Improve memory access pattern to reduce wasted transactions，提高bus utilization

* per warp (coalesced, 32/64/128B)
  * 线程访问可以不连续，但内存需要连续
* in discrete chunks (transported in segments: L2 cache line, 32B or 128B)
  * 长度和index均需对齐

##### Warp Divergence

* 如果 warp 内的线程执行不同的分支，会出现分支分歧，显著降低性能
  * 线程块的线程数量是 32 的倍数，更好地组织线程能减少分支分歧的发生概率

##### intra-warp data sharing

![warp](./GPU/warp-sharing.png)

#### Shared Memory利用 —— Tiling

* Reduce redundant access: shared memory
  * Inter-block communication
  * User-managed cache to reduce redundant global memory accesses
  * Avoid non-coalesced access: shared memory没有cache line的概念，e.g. matrix-transposition.cu

![stencil](./GPU/stencil.png)

* 本质：In Matmul, each of the n² outputs uses 2n inputs
  * n^2 * 2n / (2n^2) = n，每个input重复读n次
  * 优化后：read each input only n/TILE_SIZE times from main memory
  * cuda技巧：双buffer的思路，prefetch和计算并行
    * cuda-sgemm.cu

![image-20250404222219317](./GPU/image-20250404222219317.png)

#### TensorCore利用

https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9926-tensor-core-performance-the-ultimate-guide.pdf

#### H2D Optimization

> https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/

* Host<->device data transfer has much lower bandwidth than global memory access.

  * 16 GB/s (PCIe x16 Gen3) vs 250 GB/s & 10.6 T inst/s (GP100) 

* Minimize transfer

  * Intermediate data can be allocated, operated, de-allocated directly on GPU

  * Sometimes it’s even better to re-compute on GPU

* Group transfer
  * One large transfer much better than many small ones
  * Overlap memory transfer with computation

#### Instruction Optimization

* Use float if precision allow
  * Adding “f” to floating literals (e.g. 1.0f) because the default is double 

* Fast math functions
  * Two types of runtime math library functions
    * func(): slower but higher accuracy (5 ulp or less)
    * __func(): fast but lower accuracy (SFU做，see prog. guide for full details) 
    * -use_fast_math: forces every func() to __func ()

* High-throughput function:
  * DP4A and DP2A  for int8 and int16 dot productions
    * Dot Product of 4/2 elements and Accumulate
  * Warp matrix function(WMMA) for tensor core operations

![control-flow](./GPU/control-flow.png)

### PMPP: Programming Massively Parallel Processors

> * 书：Programming Massively Parallel Processors (PMPP) 3rd edition
>
> * 视频：https://www.youtube.com/@pmpp-book/videos?view=0&sort=dd&shelf_id=2

* Main Goals
  * Parallel programming & computational thinking
  * Correct & reliable: debugging function & performance 
  * Scalability: regularize and localize memory access

#### Ch1-3 PPT

> GPU-Mode Lecture 2: https://www.youtube.com/watch?v=NQ-0D5Ti2dc

* Intro
  * motivation: GPU go brrr, more FLOPS please
    * Why? Simulation & world-models (games, weather, proteins, robotics)
    * Bigger models are smarter -> AGI (prevent wars, fix climate, cure cancer)
    * GPUs are the backbone of modern deep learning
  * classic software: sequential programs
    * higher clock rate trend for CPU slowed in 2003: energy consumption & heat dissipation
  * multi-core CPU came up
    * developers had to learn multi-threading (deadlocks, races etc.)
* Heterogeneous data parallel computing
  * Heterogeneous：GPU + CPU
  * CUDA C: extends ANSI C with minimal new syntax

* Multidimensional grids and data

#### Ch4-5 Compute and Memory Basics

> GPU-Mode Lecture 4: https://www.youtube.com/watch?v=lTmYrKwjSOU

* Chapter 4: Compute Architecture and Scheduling
  * aka: How to keep all of the GPU busy
* Chapter 5: Memory architecture and data locality
  * aka the basics of getting fast kernels

### GPU Profiling

> [tf-timeline](https://zhuanlan.zhihu.com/p/40156908)
>
> [nsight-systems](https://developer.nvidia.cn/nsight-systems)
>
> [nsight-compute](https://developer.nvidia.com/nsight-compute)

* Intro
  * cuda是async，因此用python的time模块，测的包含kernel launch时间，不包含execute时间

#### Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking

https://arxiv.org/pdf/1804.06826

#### Demystifying the Nvidia Ampere Architecture through Microbenchmarking and Instruction-level Analysis

microbenchmark using ptx

#### Nvidia Lecture 5: Introduction to Nsight Profiling Tools

![nsight-product](./GPU/nsight-product.png)


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



* nvtx记录kernel信息
  * "//tensorflow/core/profiler:nvtx_utils"
  * nvtxDomainRangeStartEx 和 nvtxDomainRangeEnd
  * export TF_ENABLE_NVTX_RANGES=1、export TF_ENABLE_NVTX_RANGES_DETAILED=1



* Key features
  * section is a group of metrics


![warp-scheduler](./GPU/warp-scheduler.png)

### Prod Ready GPU Library

#### Intro

#### [Build a Prod Ready CUDA library —— GPU Mode Lecture 10](https://www.youtube.com/watch?v=FHsEW0HpuoU)

* 讲了一些故事：
  * 没什么人懂gpu
  * GPU优化，不同的步骤放在不同的GPU上

![image-20250509235246521](./GPU/image-20250509235246521.png)

![image-20250509235815478](./GPU/image-20250509235815478.png)

![image-20250510000035990](./GPU/image-20250510000035990.png)

* prod ready cuda library
  * 均衡抽象和性能

##### GPU communication utility

> nccl: https://developer.nvidia.com/nccl
>
> https://developer.nvidia.com/deepstream-sdk

* 图像处理的场景：3GPU、输入60 frames/s * 4 cameras，输出6 * 60 frames/s
  * **soft realtime**：可接受延时
  * 由于driver等原因，只能windows，很坑
* 要点1：Pre-allocate all your memory
* 要点2:
  * h2d，经过pinned memory
  * ![image-20250510023351873](./GPU/image-20250510023351873.png)

![image-20250510023535472](./GPU/image-20250510023535472.png)

![image-20250510024754543](./GPU/image-20250510024754543.png)

* delay buffer + Iterative memory manager
  * 管理数据ownership，类似unique_ptr

![image-20250510030621333](./GPU/image-20250510030621333.png)

```C++
struct DataInfo {
  int numElements;
  int elemSizeInBytes;
  MemorySpace memSpace;
};
DataInfo producerDataInfo{1024, 4, HostPageable};
DataInfo consumerDataInfo{1024, 4, Device_1};

Data ptrToProduce(producerDataInfo);
Data ptrToConsume(consumerDataInfo);

enum Actions { ProducerProvides, ProducerTakes, ConsumerProvides, ConsumerTakes};

//Initialization
MemoryManager<ProducerProvides, ConsumerProvides> manager(producerDataInfo, consumerDataInfo);

int delay = manager.getTotalDelay(); // Query delay generated by the manager

manager.manage(ptrToProduce, ptrToConsume); // Usage

// Initialization
MemoryManager<ProducerTakes, ConsumerTakes> manager(producerDataInfo, consumerDataInfo);
auto [ptrToProduce, ptrToConsume] = manager.manage(); // Usage

// Initialization
MemoryManager<ProducerProvides, ConsumerTakes> manager(producerDataInfo, consumerDataInfo);
ptrToConsume = manager.manage(ptrToProduce); // Usage
```

![image-20250510032943928](./GPU/image-20250510032943928.png)



#### GemLite —— Low-Bit Triton Kernels

> /code-reading-gem-lite

* Intro

  - Support for various activation data types: fp16, int8 and fp8

  - Compatibility: works seamlessly with non-packed (e.g., int8, fp8) and packed formats (e.g., uint4, uint2, uint1)

  - Performance Optimization: includes optimized kernels and autotuning tools to achieve high performance across different hardware and batch sizes

  - Integration: Compatible with torch.compile and CUDA graphs, ensuring support for advanced features like tensor parallelism

* Kernel Selection
  * For batch size = 1, a GEMV kernel performs best,
    * for packed data, our experiments indicate that **loading scales and zero points only once per two consecutive blocks minimizes redundant operations**. Since these blocks share the same metadata, this approach results in:
      - 5–8% end-to-end inference speedup compared to the default GEMV kernel
      - 30–40% improvement over the traditional Split-K method
    * for non packed: GEMV_SPLITK
  * for larger batch sizes, GEMM kernels are more efficient.
  * For batch sizes between 2 and 64, when matrices are “skinny,” a GEMM-SPLITK kernel is used to enable better GPU utilization ([arXiv](https://arxiv.org/abs/2402.00025)).

* autotuning
  * [Autotuning](https://triton-lang.org/main/python-api/generated/triton.autotune.html) is critical for achieving optimal kernel performance. Since this process can be time-intensive, GemLite **provides tools to automatically save and load autotuning results for all kernels**. This ensures that the autotuning process is performed only once per GPU device, minimizing runtime, reducing repetitive overhead, and maintaining consistent performance across runs.
* Overcoming **Bit-Unpacking Bottlenecks**
  * To mitigate these, various bit-packing configurations were explored, including **packing along columns** versus rows and experimenting with different bit-packing widths (e.g., 8-bit vs. 32-bit). Notably, **transitioning from 32-bit to 8-bit packing** delivered performance improvements of up to 18% on the A100 and 6% on the H100

### 应用

#### Intro

* use [*GPU-Accelerated Libraries for Computing*](https://developer.nvidia.com/gpu-accelerated-libraries) to learn where you can use highly optimized CUDA libraries for tasks like:
  *  [basic linear algebra solvers](https://developer.nvidia.com/cublas) (BLAS)
  * [graph analytics](https://developer.nvidia.com/nvgraph)
  * [fast fourier transforms](https://developer.nvidia.com/cufft) (FFT)
  * [random number generation](https://developer.nvidia.com/curand) (RNG)

#### 图像处理

* [image and signal processing](https://developer.nvidia.com/npp)

* GPU做图像处理pipeline自动优化
  * Halide: a language and compiler for optimizing parallelism, locality, and recomputation in
    image processing pipelines.



