[toc]

### Low Latency Guide

https://rigtorp.se/low-latency-guide/

* Disable hyper-threading
  * Using the [CPU hot-plugging functionality](https://www.kernel.org/doc/html/latest/core-api/cpu_hotplug.html) to disable one of a pair of sibling threads. Use `lscpu --extended` or `cat /sys/devices/system/cpu/cpu*/topology/thread_siblings_list` to determine which “CPUs” are sibling threads.
* CAS Latency: https://en.wikipedia.org/wiki/CAS_latency
  * One byte of memory (from each chip; 64 bits total from the whole DIMM) is accessed by supplying a 3-bit bank number, a 14-bit row address, and a 13-bit column address.
* 绑核
  * `sched_getaffinity`、`sched_setaffinity`
* Uncore
  * [What is Uncore?](https://forums.tomshardware.com/threads/what-is-uncore-how-is-it-relevant-to-overclocking.2094084/)

### Arch with C++

```c++
constexpr size_t CACHE_LINE_SIZE =
#if __cplusplus >= 201703L and __cpp_lib_hardware_interference_size
  std::hardware_constructive_interference_size;
#else
  64;
#endif
```

`__buildin_prefetch`: https://www.daemon-systems.org/man/__builtin_prefetch.3.html

[Using the extra 16 bits in 64-bit pointers](https://stackoverflow.com/questions/16198700/using-the-extra-16-bits-in-64-bit-pointers)

### 各种微架构

#### Intel

[Xeon Gold 5118 - Intel](https://en.wikichip.org/wiki/intel/xeon_gold/5118)

[14 nm lithography process](https://en.wikichip.org/wiki/14_nm_lithography_process)

* [Skylake (server) - Microarchitectures - Intel](https://en.wikichip.org/wiki/intel/microarchitectures/skylake_(server))
  * [Mesh Interconnect Architecture - Intel](https://en.wikichip.org/wiki/intel/mesh_interconnect_architecture)
  * [The Intel Skylake-X Review: Core i9 7900X, i7 7820X and i7 7800X Tested](https://www.anandtech.com/show/11550/the-intel-skylakex-review-core-i9-7900x-i7-7820x-and-i7-7800x-tested/5)
  * skylake 的一些简单 latency 参数：https://www.7-cpu.com/cpu/Skylake.html
* [Sapphire Rapids](https://en.wikipedia.org/wiki/Sapphire_Rapids) for IDC, Alder Lake for public use
  * 它的上一代：[Whitley](https://www.asipartner.com/solutions/server/intel-whitley-platform/)

#### AMD

* [AMD CCD and CCX in Ryzen Processors Explained](https://www.hardwaretimes.com/amd-ccd-and-ccx-in-ryzen-processors-explained/#go-to-content)

  * **The basic unit of a Ryzen processor is a CCX or Core Complex**, a quad-core/octa-core CPU model with a shared L3 cache.

  * However, while CCXs are the basic unit of silicon dabbed, at an architectural level, a **CCD or Core Chiplet Die is your lowest level of abstraction**. A CCD consists of two CCXs paired together using the Infinity Fabric Interconnect. All Ryzen parts, even quad-core parts, ship with at least one CCD. They just have a differing number of cores disabled per CCX.

  * zen-3(milan)舍弃了一个CCD包含两个CCX的概念，8 cores (one CCD/CCX) 共享 32MB 的 L3 cache

  * Intel’s Monolithic Design and the Future
* [AMD Ryzen 5000 “Zen 3” Architectural Deep Dive](https://www.hardwaretimes.com/amd-ryzen-5000-zen-3-architectural-deep-dive/)
  * lower core-to-core latency
  * Branch Target Buffer (BTB)
  * Although the load/store bandwidths have effectively doubled, the L1 to L2 cache transfer speeds are unchanged at 32 bytes/cycle x 2. The L1 fetch width is also the same as Zen 2 at 32 bytes.

* [NUMA Configuration settings on AMD EPYC 2nd Generation](https://downloads.dell.com/manuals/common/dell-emc-dfd-numa-amd-epyc-2ndgen.pdf)

  * 1 socket ~ 4 CCX ~ 8 memory controller(~ memory channel)
    * Up to 2 DIMMs per channel
  * With this architecture, all cores on a single CCD are closest to 2 memory channels. The rest of the memory channels are across the IO die, at differing distances from these cores. Memory interleaving allows a CPU to efficiently spread memory accesses across multiple DIMMs. This allows more memory accesses to execute without waiting for one to complete, maximizing performance. 
  * Nodes Per Socket (NPS)
  * For additional tuning details, please refer to the Tuning Guides shared by AMD [here](For additional tuning details, please refer to the Tuning Guides shared by AMD here. For detailed discussions around the AMD memory architecture, and memory configurations, please refer to the Balanced Memory Whitepaper). For detailed discussions around the AMD memory architecture, and memory configurations, please refer to the [Balanced Memory Whitepaper](https://downloads.dell.com/Manuals/Common/dellemc-balanced-memory-2ndgen-amd-epyc-poweredge.pdf)

#### AMD 课程

ROCm：https://developer.amd.com/resources/rocm-learning-center/

### Cache 系列科普 ~ Latency

[UEFI和BIOS探秘 —— Zhihu Column](https://www.zhihu.com/column/UEFIBlog)

[interactive latency numbers](https://colin-scott.github.io/personal_website/research/interactive_latency.html)

数据基于 Skylake 架构

* L1 cache
  * 32KB
  * 4~5 cycles, L1D latency ~1ns
  * 算一下 load/store 指令占所有 instructions 的比例，小于 5 就没办法 hide latency，需要优化访存模式
* L2 cache
  * 512KB
  * ~12 cycles, ~4ns
* LLC (L3 cache)
  * 32MB
  * ~38 cycles, ~12ns
    * 与之对比，memory access 约 50~100ns (Intel 70ns, AMD 80ns)
  * 直接走 IMC，1.5MB/core
  * 分析：llc-load-miss * 64B per load / time elapsed，和内存带宽数据做对比
* L4: eDRAM，可作显存
* DRAM
  * ~100ns

* [L1，L2，L3 Cache究竟在哪里？](https://zhuanlan.zhihu.com/p/31422201)
  * [CPU Die and Socket](https://zhuanlan.zhihu.com/p/51354994): Intel Xeon 是一个 CPU Die 一个 Socket；而 AMD ECPY 的一个 Socket 由 4 个 CPU Die 组成。因此 AMD 8 个逻辑核共享 LLC，而 Intel 全部核心共享
  * [为什么Intel CPU的Die越来越小了？](https://zhuanlan.zhihu.com/p/31903866)
    * 晶体管数目增长落后于晶体管密度增长
    * Coffeelake 8700K，晶体管的密度不增反降，Pitch从70nm增加到了84nm。在可以提供更高频率支持的背后，代价就是对Die的大小造成负面影响
* [Cache是怎么组织和工作的？](https://zhuanlan.zhihu.com/p/31859105)
  * 全相联、组相联
* [Cache为什么有那么多级？为什么一级比一级大？是不是Cache越大越好？](https://zhuanlan.zhihu.com/p/32058808)
* [显存为什么不能当内存使？内存、Cache和Cache一致性](https://zhuanlan.zhihu.com/p/63494668)
  * 有 GDDR 和 PC DDR 设计初衷不同导致的问题
  * 为什么不能通过PCIe来扩展普通内存？
    * 为什么偷显存性能低的原因：显存不能保证被cache，或者说无法保证cache的一致性
  * Cache 一致性
    * 用硬件而非软件来做 cache coherency
    * CPU 片内总线架构演进：ring bus -> mesh
    * 模型：MESI protocol

![img](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/Computer-Architecture/MESI-protocol.jpg)

![img](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/Computer-Architecture/MESI-fsm.jpg)


SMP (Symmetric Multiprocessing): cache的发展

Bus Snooping (1983)

* 实现：Home Agent (HA)，在内存控制器端；Cache Agent (CA)，在L3 Cache端
  * [Intel 的两种 snoop 的方式](https://www.intel.ca/content/dam/doc/white-paper/quick-path-interconnect-introduction-paper.pdf)：Home Snoop 和 Source Snoop。它们的主要区别在于谁主导 snoop 消息的发送

![img](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/Computer-Architecture/bus-snooping.jpg)

* 缺点：在QPI总线上广播，带宽消耗大，scaling 有问题

* Write-invalidate
* Write-update



The two most common mechanisms of ensuring coherency are *[snooping](https://en.wikipedia.org/wiki/Bus_sniffing)* and *[directory-based](https://en.wikipedia.org/wiki/Directory-based_cache_coherence)*, each having their own benefits and drawbacks.

* Snooping based protocols tend to be faster, if enough [bandwidth](https://en.wikipedia.org/wiki/Memory_bandwidth) is available, since all transactions are a request/response seen by all processors. The drawback is that snooping isn't scalable. Every request must be broadcast to all nodes in a system, meaning that as the system gets larger, the size of the (logical or physical) bus and the bandwidth it provides must grow.
* Directories, on the other hand, tend to have longer latencies (with a 3 hop request/forward/respond) but use much less bandwidth since messages are point to point and not broadcast. For this reason, many of the larger systems (>64 processors) use this type of cache coherence.
  * 图解 slides: http://15418.courses.cs.cmu.edu/spring2017/lecture/directorycoherence/slide_015

* scalability of multi-thread applications can be limited by synchronization 
  * 延伸：PCIe 内部 memory （包括 PCIe 后面的显存、NvRAM 等）的割裂性在服务器领域造成了很大问题，CXL 的引入为解决这个问题提供了技术手段

* synchronization primitives: LOCK PREFIX、XCHG

内存拓扑的 unbalanced 问题

* 可能导致同一物理机上先启动的服务效率高
* 多 channel 的 64bit DRAM，ddr 频率在 2666 居多，单 channel 可以到 ～20GB/s，4～6 channel 比较常见

![mem-layout](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/Computer-Architecture/mem-layout.png)





Some conclusion and Advices

* 11 -> 1H: Hyper Threading could help on performance on such “lock” condition (But may not in the end, maybe depends on the total threads: C1-> CH)

* 22 -> 21: Lower Core-Count Topology helps for this circustances (Not the Benchmark Software threads)

* Increase of the hardware resource (along with a huge mount of OS threads) usually not help on the performance, but waste of the CPU and memory resources

* Intel’s “innovation” for high performance processors has been tired with maintaining the same performance for “unconstrained” end users.
  * Intel has been done very well, if you compare with ARM64 Enterprise and AMD...

* lock code is the RISK (pitfall from something beyond your source code, even from glibc, 3rd lib ...)

* Use the scaling tests to find your bottleneck, and improve the “lock”  components
  * Maybe from DISK I/O, Network layer
  * Rarely from the memory bandwidth layer, LLC cache size for the non-HPC workloads

### NUMA

* 搜到一段获取 cpu topology 的 C 语言代码：https://github.com/SANL-2015/SANL-2015/blob/8779af7939bcacebd74abfabba9873b68eaca304/SAND2015/liblock/liblock.c#L99

```shell
yum -y install numactl numastat

numactl -H
numastat
numactl -C 0-15 ./bin
numactl -N0 -m0 ./bin
```




### Hyper-threading

https://www.semanticscholar.org/paper/Hyper-Threading-Technology-Architecture-and-1-and-Marr-Binns/04b58af4fc0e5c3e8e614e2ddb0c41749cc9166c

https://pdfs.semanticscholar.org/04b5/8af4fc0e5c3e8e614e2ddb0c41749cc9166c.pdf?_ga=2.24705338.1691629142.1553869518-295966427.1553869518

https://www.slideshare.net/am_sharifian/intel-hyper-threading-technology/1

* 实测性能是 -20% ~ +20%，因为可能依赖内存带宽、抢占cache。数值计算任务，用到了AVX、SSE技术的，开Hyper-threading一般都会降性能；访存频繁的CPU任务，建议打开，因为circle、LRU是idle的，会有提升
* 禁止hyper-threading：offline、isolate

### 指令集

* AVX-512 throttling
  * [On the dangers of Intel's frequency scaling](https://blog.cloudflare.com/on-the-dangers-of-intels-frequency-scaling/)
  * [Gathering Intel on Intel AVX-512 Transitions](https://travisdowns.github.io/blog/2020/01/17/avxfreq1.html)
* [AVX-512_BF16 Extension](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-deep-learning-boost-new-instruction-bfloat16.html)
  * 点积 + 转换
* CLZ 等指令：https://en.wikipedia.org/wiki/Find_first_set

#### AMX

* AMX: [The x86 Advanced Matrix Extension (AMX) Brings Matrix Operations; To Debut with Sapphire Rapids](https://fuse.wikichip.org/news/3600/the-x86-advanced-matrix-extension-amx-brings-matrix-operations-to-debut-with-sapphire-rapids/)
  * AMX introduces a new matrix register file with eight rank-2 tensor (matrix) registers called “tiles”.
  * 独立单元，支持bf16，intel oneAPI DNNL指令集接口
* 功能：单元相比avx512、vnni，支持了reduce操作
* 实现：
  - 输入8位/16位，计算用32位（防溢出）
  - 扩展了tile config、tile data寄存器，内核支持（XFD，eXtended Feature Disable）, allos os to add states to thread on demand
* 测试：减少内存带宽，增频了（重计算指令减少）
  - 配合tf有automix策略，op可能为bf16+fp32混合计算
  - matmul相关运算全bf16
  - tf2.9合入intel大量patch, TF_ENABLE_ONEDNN_OPTS=1，检测cpuid自动打开；oneDNN 2.7
    - TF_ONEDNN_USE_SYSTEM_ALLOCATOR
    - jemalloc: MALLOC_CONF="oversize_threshold:96000000,dirty_decay_ms:30000,muzzy_decay_ms:30000"), mitigate page_fault, which benefit fp32 in addition
    - TF_ONEDNN_PRIM_CACHE_SIZE=8192 (for mix models deployment)
    - TF2.9 SplitV performance issue
* 其它SPR独立单元：
  - Intel DSA(data streaming accelerator): batched memcpy/memmove，减少CPU cycles
    - https://01.org/blogs/2019/introducing-intel-data-streaming-accelerator
  - Intel IAA(in-memory analytics accelerator): compress/decompress/scan/filter，也是offload cpu cores
    - https://www.intel.com/content/www/us/en/analytics/in-memory-data-and-analytics.html
    - 场景如presto：https://engineering.fb.com/2019/06/10/data-infrastructure/aria-presto/

### 存储

* NVMe
  * [NVMe vs SATA: What’s the difference and which is faster?](https://www.microcontrollertips.com/why-nvme-ssds-are-faster-than-sata-ssds/)
  * [Bandana: Using Non-volatile Memory for Storing Deep Learning Models, SysML 2019](https://arxiv.org/pdf/1811.05922.pdf)
    * https://www.youtube.com/watch?v=MSaD8DFsMAg
  * Persistent Memory
    * Optane DIMM: https://www.anandtech.com/show/12828/intel-launches-optane-dimms-up-to-512gb-apache-pass-is-here
      * Optane DC PMMs can be configured in one of these two modes: (1) memory mode and (2) app direct mode. In the former mode, the DRAM DIMMs serve as a hardware-managed cache (i.e., direct mapped write-back L4 cache) for frequently-accessed data residing on slower PMMs. The memory mode enables legacy software to leverage PMMs as a high-capacity volatile main memory device without extensive modifications. However, it does not allow the DBMS to utilize the non-volatility property of PMMs. In the latter mode, the PMMs are directly exposed to the processor and the DBMS directly manages both DRAM and NVM. In this paper, we configure the PMMs in app direct mode to ensure the durability of NVM-resident data.
      * pmem: https://pmem.io/pmdk/
      * DWPD: 衡量 SSD 寿命
      * 《Spitfire: A Three-Tier Buffer Manager for Volatile and Non-Volatile Memory》

### 显示器

一些外设概念

* OSD(On Screen Display) Menu
* 接口：DC-IN, HDMI 2.0 两个, DisplayPort, 耳机, USB 3.0 两个, Kensington 锁槽
* 170Hz 刷新率、130%sRGB、96%DCI-P3

### 显卡

[gpu-z 判断锁算力版本](https://zhuanlan.zhihu.com/p/385968761)

### 主板

* [PCI-E x1/x4/x8/x16](https://www.toutiao.com/i6852969617992712715)
  * PCI-E x16：22（供电）+142（数据）；用于显卡，最靠近 CPU
  * PCI-E x8：伪装成 x16
  * PCI-E x4：22+14；通常由主板芯片扩展而来，也有直连 CPU 的，用于安装 PCI-E SSD
  * PCI-E x1：独立网卡、独立声卡、USB 3.0/3.1扩展卡等
    * 另外一个形态，一般称为Mini PCI-E插槽，常见于 Mini-ITX 主板以及笔记本电脑上，多数用来扩展无线网卡，但由于其在物理结构上与 mSATA 插槽相同，因此也有不少主板会通过跳线或者 BIOS 设定让 Mini PCI-E 接口在 PCI-E 模式或者 SATA 模式中切换，以实现一口两用的效果。已经被 M.2 接口取代，基本上已经告别主流。

