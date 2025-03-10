[toc]

__主题：virtualization, concurrency, persistence__

[book](http://pages.cs.wisc.edu/~remzi/OSTEP/), [book-code](https://github.com/remzi-arpacidusseau/ostep-code), [projects](https://github.com/remzi-arpacidusseau/ostep-projects), [homework](https://github.com/remzi-arpacidusseau/ostep-homework/) [xxxzz's homework answer](https://github.com/xxyzz/ostep-hw), [my homework answer](https://github.com/huangrt01/ostep-homework)

### Kernel

* Meltdown patch

  * Meltdown patch 之后，Page Table Isolation (PTI)，syscall 会 flush TLB
  * Linux 4.14 之后，support PCID (process-context id)
    * Process-Context Identifiers (PCID) enables us to achieve the same goal of isolation much more efficiently by associating some data with each TLB entry which the processor uses to control access to the mappings. By changing the PCID during the mode switch, TLB entries with the kernel’s PCID will not be accessible from user space.
  * [Linux Meltdown patch: 'Up to 800 percent CPU overhead', Netflix tests show](https://www.zdnet.com/article/linux-meltdown-patch-up-to-800-percent-cpu-overhead-netflix-tests-show/)

  * [Meltdown: What’s the performance impact and how to minimise it?](https://www.opsian.com/blog/meltdown-benchmarks/)

### CPU性能优化 - KV Store

#### 《Put an Elephant into a Fridge: Optimizing Cache Efficiency for In-memory Key-value Stores》

Abstract

现状： Such an extremely low cache-to-memory ratio (less than 0.1%) poses a significant new challenge—the limited CPU cache is becoming a severe performance bottleneck that hinders us from fully exploiting the great potential of high-speed memory-based key-value stores.

问题：cache contention, thrashing, inability to scale,
https://en.wikipedia.org/wiki/Thrashing_(computer_science)

解决方案：By carefully reorganizing

the data layout in memory, redesigning the hash indexing structure, and offloading garbage collection, we can effectively improve the utilization of the limited cache space.

1.Introduction

1.1 Techinical Trend and Challenges

* CPU cache 贵、重要、对性能影响大、scalable能力有限
* 利用 CPU cache 的 challenges
* hardware challenges: 软件很难直接管理cpu cache
* software challenges: 数据冷热特点；value size比key大，caching a value可能逐出多个key；hash indexing structure很重要

1.2 Making Key-Value Store Cache Aware

software-only solution: best placement of a key-value item according to its temporal locality; key/value分离；新的hash indexing structure

2. Motivations and Challenges

* Issue #1: Disproportional key and value sizes
* Issue #2: Low cache utilization in hash indexing.
* Issue #3: Read amplification with key-values.

3.Mechanism
* page coloring: 本质上给LLC cache做了32个分片
* Gaining control on cache
* Mapping with Hugepage: 2-MB page = 16 columns (one column = 32 rows * 64 sets * 64 B)
* Mapping with pre-allocated pages
* get_pgcolor

4.Policy

4.1 Handling Hot and Cold Key-value Data

* we desire to see that each row is filled up with both cold and hot data, which compete for the space within the row, and upon eviction, the victims would be the cold ones.
* 手段：Memcached maintains an LRU list per slab class to track each key-value item’s relatively locality, while Redis maintains a pool of weak-locality (cold) key-values for eviction by sampling the dataset periodically.

4.2 Separating Key and Value Data
* placing them in separate rows
* key/value 读写分离的问题：
* parallel access: 维护两个指针，缺点是需要指针以及内存带宽的浪费
* concurrent access: 维护两个queue，效果更好

4.3 Cache-friendly Hash Indexing


case study 1: Upon inserting a key-value item, a slab from the slab class with the smallest slot size that can accommodate the item is selected.

首先，尽量让hot items（logical position接近的items）集中在某列（physical zone）

其次，每一个physical zone内部的hot item尽量不在同一个row（不同的color）

6. Case Study 1: MEMCACHED

* 6.1 Optimizations
  * head-to-head allocation比较有趣，在key-value分离的设计下，解决了不知道key and value内存比例的问题

7. Case Study 2: REDIS

* redis和memcached的区别：没有slab -> 不需要 LRU list 来做 repartition，用一个LRU clock，后台线程做eviction

8.Discussion

* 只要cashe sets够用，增加set数量不会有提升
* huge page由于是system-wide配置，有受到恶意攻击disturb the shared cache的可能

#### MICA: A Holistic Approach to Fast In-Memory Key-Value Storage, NSDI 2014

https://www.usenix.org/conference/nsdi14/technical-sessions/presentation/lim

MICA(Memory-store with Intelligent Concurrent Access) takes a holistic approach that encompasses all aspects of request handling, including parallel data access, network request handling, and data structure design, but makes unconventional choices in each of the three do- mains. First, MICA optimizes for multi-core architectures by enabling parallel access to partitioned data. Second, for efficient parallel data access, MICA maps client requests directly to specific CPU cores at the server NIC level by using client-supplied information and adopts a light-weight networking stack that bypasses the kernel. Finally, MICA’s new data structures—circular logs, lossy concurrent hash indexes, and bulk chaining—handle both read- and write-intensive workloads at low overhead.

3. Key Design Choices

3.1 Parallel Data Access

MICA’s parallel data access: MICA partitions data and mainly uses exclusive access to the partitions. MICA exploits CPU caches and packet burst I/O to disproportionately speed more loaded partitions, nearly eliminating the penalty from skewed workloads. MICA can fall back to concurrent reads if the load is extremely skewed, but avoids concurrent writes, which are always slower than exclusive writes. Section 4.1 describes our data access models and partitioning scheme.

3.2 Network Stack
* socket I/O 比较费，大量的read
* direct NIC access
* request direction
* Flow-level core affinity: 1) Receive-Side Scaling (RSS); 2) Flow Director (FDir)
* Object-level core affinity
* MICA's request direction: 利用Flow Director，在client去做object的编码，让NIC能理解

3.3 KV Data Structures

3.3.1 Memory Allocator

* cache mode: log structure
* store mode: segregated fits
3.3.2 Indexing: Read-oriented vs. Write-friendly
* lossy data structures:
* bulk chaining
* use memory allocator's eviction support to avoid evicting recently-used items (4.3.2)

4. MICA Design
    4.1.2 分析CREW(Concurrent Read Exclusive Write)模式 -> 4.3.1

  4.2 Network Stack

  ​	UDP
  4.2.1 Direct NIC Access

  ​	Intel's DPRK

  4.2.2 Client-Assisted Hardware Request Direction

4.3 Data Structure

MICA, in cache mode, uses circular logs to manage memory for key-value items and lossy concurrent hash indexes to index the stored items. Both data structures exploit cache semantics to provide fast writes and simple memory management. Each MICA partition consists of a single circular log and lossy concurrent hash index. MICA provides a store mode with straightforward extensions using segregated fits to allocate memory for key- value items and bulk chaining to convert the lossy concurrent hash indexes into lossless ones.

Hugepages (2 MiB in x86-64) use fewer TLB entries for the same amount of memory, which significantly reduces TLB misses during request processing.

hugepage必须预先一次分配2M或者1GB的内存空间，并且使用mmap的接口去分配。所以用malloc和free的库来管理大页是合理的（比如jemalloc去支持hugepage配置）。

4.3.2 Lossy Concurrent Hash Index

#### MEMC3: Compact and concurrent memcache with dumber caching and smarter hashing, NSDI 2013

* MemC3—Memcached with CLOCK and Concurrent Cuckoo hashing
  * optimistic cuckoo hashing, a compact LRU-approximating eviction algorithm based upon CLOCK, and comprehensive implementation of optimistic locking
  * cuckoo hashing: https://web.stanford.edu/class/archive/cs/cs166/cs166.1146/lectures/13/Small13.pdf

3. Optimistic Concurrent Cuckoo Hashing

* An optimistic version of cuckoo hashing that supports multiple-reader/single writer concurrent access, while preserving its space benefits;
  * 记录version，version不对（比如是奇数）就retry
  * 利用了 the atomic read/write for 64-bit aligned pointers on 64-bit machines 的特点，参考 APPENDIX
* A technique using a short summary of each key to improve the cache locality of hash table operations; 
  * An optimization for cuckoo hashing insertion that improves throughput
  * CLOCK-based approximate LRU，clock算法可以增强thread的scale能力（消除LRU synchronization的瓶颈）

#### Full-stack architecting to achieve a billion-requests-per-second throughput on a single key-value store server platform, 2016
![network-with-kv](OSTEP-Operating-Systems-Three-Easy-Pieces/network-with-kv.png)



* DDIO
  * https://www.intel.com/content/www/us/en/io/data-direct-i-o-technology.html
  * https://www.intel.com/content/www/us/en/io/data-direct-i-o-technology-brief.html





#### 杂项

* [Intel Introduces Thread Director For Heterogeneous Multi-Core Workload Scheduling](https://fuse.wikichip.org/news/6123/intel-introduces-thread-director-for-heterogeneous-multi-core-workload-scheduling/)
  * performance core and efficient core 的概念，OS会aware程序hints（切换core是否有损性能）、core的状态



### 内存优化

* garbage collection
  * 《Quantifying the performance of garbage collection vs.
    explicit memory management.》



### Intro

#### 1.Dialogue

I hear and I forget. I see and I remember. I do and I understand.    其实是荀子说的

#### 2.Introduction to Operating Systems
* Von Neumann model
* OS：并行，外设，resource manager
* 概念：virtualization, API, system calls, standard library
##### CRUX: how to virtualize resources
##### CRUX: how to build correct concurrent programs
* persistance
* write有讲究：1）先延迟一会按batch操作    2）protocol, such as journaling or copy-on-write    3）复杂结构B-tree
##### CRUX: how to store data persistently
* 目标：
  * performance: minimize the overheads
  * protection ~ isolation
  * reliability
  * security
  * mobility
  * energy-efficiency
* history: 
  * libraries
  * protection
  * trap handler            user/kernel mode
  * multiprogramming    minicomputer
  * memory protection    concurrency    ASIDE:UNIX
  * modern era: PC    Linus Torvalds: Linux

### Virtualization

#### 3.Dialogue

#### 4.the abstraction: The Process
##### CRUX: how to provide the illusion of many CPUs
* low level machinery    
  * e.g. context switch : register context
* policies 
  * high level intelligence
  * e.g. scheduling policy
* separating policy(which) and mechanism(how)
  * modularity
* Process Creation
  * load lazily: paging and swaping 
  * run-time stack; heap(malloc(),free())    
  * I/O setups; default file descriptors

<img src="OSTEP-Operating-Systems-Three-Easy-Pieces/001.jpg" alt="进程状态转移" style="zoom:50%;" />

* final态（在UNIX称作zombie state）等待子进程return 0，parent进程 wait()子进程
```shell
# 查找僵尸进程
ps -aux|grep Z
ps -ef|grep 子进程pid
kill -9 父进程pid
```



* xv6 process structure

```c++
// the registers xv6 will save and restore
// to stop and subsequently restart a process
struct context {
    int eip;int esp;
    int ebx;int ecx;
    int edx;int esi;
    int edi;int ebp;
};
// the different states a process can be in
enum proc_state { UNUSED, EMBRYO, SLEEPING,RUNNABLE, RUNNING, ZOMBIE };
// the information xv6 tracks about each process
// including its register context and state
struct proc {
    char*mem;                  // Start of process memory
    uint sz;                    // Size of process memory
    char*kstack;               // Bottom of kernel stack
                               // for this process
    enum proc_state state;      // Process state
    int pid;                    // Process ID
    struct proc*parent;        // Parent process
    void*chan;                 // If !zero, sleeping on chan
    int killed;                 // If !zero, has been killed
    struct file*ofile[NOFILE]; // Open files
    struct inode*cwd;          // Current directory
    struct context context;     // Switch here to run process
    struct trapframe*tf;       // Trap frame for the
                                // current interrupt
};
```

* Data Structure: process list，PCB(Process Control Block)

* HW:process-run.py
  * -I IO_RUN_IMMEDIATE      发生IO的进程接下来会有IO的概率大，所以这样高效

#### 5.Interlude: Process API
##### CRUX: how to create and control processes
* #include <unistd.h>，getpid()，fork()    不从开头开始运行
* scheduler的non-determinism，影响concurrency
* p3.c    利用execvp执行子程序wc
  * reinitialize the executable，transform原进程
  * 不会return
  * exec调用会把当前进程的机器指令都清除，因此前后的printf都不会执行
* fork+exec的意义： it lets the shell run code after the call to fork() but before the call to exec(); this code can alter the environment of the about-to-be-run program, and thus enables a variety of interesting features to be readily built.

* p4.c 
```c++
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <assert.h>
#include <sys/wait.h>

int main(int argc, char *argv[])
{
    int rc = fork();
  //  printf("STDOUT_FILENO的值是%d",STDOUT_FILENO);
    if (rc < 0) {
        // fork failed; exit
        fprintf(stderr, "fork failed\n");
        exit(1);
    } else if (rc == 0) {
	// child: redirect standard output to a file

	close(STDOUT_FILENO); 
	open("./p4.output", O_CREAT|O_WRONLY|O_TRUNC, S_IRWXU);

	// now exec "wc"...
        char *myargs[3];
        myargs[0] = strdup("wc");   // program: "wc" (word count)
        myargs[1] = strdup("p4.c"); // argument: file to count
        myargs[2] = NULL;           // marks end of array
        execvp(myargs[0], myargs);  // runs word count
    } else {
        // parent goes down this path (original process)
        int wc = wait(NULL);
	assert(wc >= 0);
    }
    return 0;
}

```
  * file descriptor的原理：按序搜索，因此需要close(STDOUT_FILENO); 
  * 类似的应用：UNIX的pipe()特性，grep -o foo file | wc -l

* 谁可以发送SIGINT信号给process=>signal(), process group, 引入user的概念
* RTFM：read the fucking manual

**HW:**

* 5.3 [用vfork()保证父进程后执行](https://www.cnblogs.com/zhangxuan/p/6387422.html)

**fork()和vfork()的区别：**

1. fork （）：子进程拷贝父进程的数据段，代码段
    * vfork（ ）：子进程与父进程共享数据段
2. fork （）父子进程的执行次序不确定
    * vfork 保证子进程先运行，在调用exec 或exit之前与父进程数据是共享的,在它调用exec或exit 之后父进程才可能被调度运行。
3. vfork （）保证子进程先运行，在她调用exec 或exit 之后父进程才可能被调度运行。如果在调用这两个函数之前子进程依赖于父进程的进一步动作，则会导致死锁。 

* 5.4 [不同的exec](https://en.wikipedia.org/wiki/Exec_(system_call)#C_language_prototypes)
  * execvp，p的含义是寻找路径，v：vector
* 5.5 如果child没有child，在child里用wait没有意义
* 5.6 waitpid()    [wait和waitpid的区别](https://www.cnblogs.com/yusenwu/p/4655286.html)
  * The pid parameter specifies the set of child processes for which to wait. If pid is -1, the call waits for any child process.  If pid is 0, the call waits for any child process in the process group of the caller.  If pid is greater than zero, the call waits for the process with process id pid.  If pid is less than -1, the call waits for any process whose process group id equals the absolute value of pid.
* 5.8    [注意子进程返回0](https://blog.csdn.net/beautysleeper/article/details/52585224)

##### 管道

* [stdio buffering](https://www.pixelbeat.org/programming/stdio_buffering/)
  * It should be noted here that changing the buffering for a stream can have unexpected effects. For example glibc (2.3.5 at least) will do a read(blksize) after every fseek() if buffering is on.
  * 期望环境变量控制
    * `tail -f access.log | BUF_1_=1 cut -d' ' -f1 | uniq`
    * 风险：Denial Of Service possibilities
* `fcntl(fileno(stdin), F_SETPIPE_SZ, pipe_page_num * 4096);`
  * 16MB is system max value by default for socket

```python
import fcntl
import platform

try:
    if platform.system() == 'Linux':
        fcntl.F_SETPIPE_SZ = 1031
        fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, size)
except IOError:
    print('can not change PIPE buffer size')
```


#### 6.Mechanism: Limited Direct Execution
##### CRUX: how to efficiently virtualize the cpu with control
* limited direct execution
##### CRUX: how to perform restricted operations
* aside: open() read()这些系统调用是trap call，写好了汇编，参数和系统调用number都放入well-known locations
  * 概念：trap into the kernel        return-from-trap        trap table    trap handler
  * be wary of user inputs in secure systems
* NOTE：
  1. x86用[per-process kernel stack](https://stackoverflow.com/questions/24413430/why-keep-a-kernel-stack-for-each-process-in-linux)，用于存进程的寄存器值，以便trap的时候寄存器够 
  2. 如何控制：set up trap table at boot time；直接进任何内核地址是very bad idea
  3. user mode不能I/O request

* system call，包括accessing the file system, creating and destroying processes, communicating with other processes, and allocating more memory（POSIX standard）
	* protection: user code中存在的是system call number，避开内核地址
	* 告诉硬件trap table在哪也是privileged operation

<img src="OSTEP-Operating-Systems-Three-Easy-Pieces/002.jpg" alt="LDE protocal" style="zoom:70%;" />

[stub code](https://www.zhihu.com/question/24844900/answer/35126766)

##### CRUX: how to regain control of the CPU
* problem #2:switching between processes
  * A cooperative approach: wait for system calls
  * [MacOS9 Emulator](http://www.columbia.edu/~em36/macos9osx.html#summary)
  * NOTE: only solution to infinite loops is to reboot the machine，reboot is useful。
  * 重启的意义：1）回到确定无误的初始状态；2）回收过期和泄漏的资源；3）不仅适合手动操作，也易于自动化
  * A Non-Cooperative Approach: The OS Takes Control
##### CRUX: how to gain control without cooperation
* a timer interrupt    interrupt handler
  * timer也可以关
* deal with malfeasance: in modern systems, the way the OS tries to handle such malfeasance is to simply terminate the offender.

* scheduler    context switch
<img src="OSTEP-Operating-Systems-Three-Easy-Pieces/003.jpg" alt="LDE protocal + timer interrupt" style="zoom:50%;" />

注意有两种register saves/restores:
* timer interrupt: 用hardware，kernel stack，implicitly，存user registers
* OS switch：用software，process structure，explicitly，存kernel registers

* e.g. xv6 context switch code

NOTE:
* 如何测time switch的成本：[LMbench](https://winddoing.github.io/post/54953.html)
* 为何这么多年操作系统速度没有明显变快：memory bandwidth

* 如何处理concurrency？=>    locking schemes，disable interrupts
  * 思考：baby-proof

HW: measurement
* [多核时代不宜用x86的RDTSC](http://www.360doc.com/content/12/0827/17/7851074_232649576.shtml)
* system call需要0.3 microseconds; context switch 0.6 microseconds; 单次记录用时1 microseconds
* [MacOS上没有sched.h](https://yyshen.github.io/2015/01/18/binding_threads_to_cores_osx.html) 

#### 7.Scheduling: Introduction

##### CRUX: how to develop scheduling policy

* workload assumptions
  * fully-operational scheduling discipline
  * 概念：jobs
* scheduling metrics：turnaround time

FIFO: convoy effect $\stackrel{\bf{允许长度不等}}{\longrightarrow}$ SJF(shortest job first) $\stackrel{\bf{允许来时不等}}{\longrightarrow} $ STCF(Shortest Time-to-Completion First )=PSJF     $\stackrel{\bf{允许不run\ to\ completion}}{\longrightarrow}$ a new metric: response time

* 概念：preemptive schedulers
* Round-Robin(RR) scheduling（轮转调度算法）
  * time slice    scheduling quantum
  * 时间片长：amortize the cost of context switching 
* 针对I/O：overlap

#### 8.Scheduling: The Multi-Level Feedback Queue(MLFQ) 多级反馈队列 
* Corbato图灵奖；和security有联系
##### CRUX: How to schedule without perfect knowledge
多个queue，每个queue对应一个priority，队内用RR => how to change priority

* Rule 1:If Priority(A)>Priority(B), A runs (B doesn’t).
* Rule 2:If Priority(A)=Priority(B), A & B run in RR.

attempt1: how to change priority

* Rule 3:When a job enters the system, it is placed at the highest priority (the topmost queue).
* Rule 4a:If a job uses up an entire time slice while running, its priority is reduced(i.e., it moves down one queue).
* Rule 4b:If a job gives up the CPU before the time slice is up, it stays at the same priority level.
* 思考：是否优先级越高的queue越倾向于用RR

MLFQ的问题：    
1. starvation
2. game the scheduler
3. change its behavior

attempt2: the priority boost

* Rule 5:After some time period S, move all the jobs in the system to the topmost queue.
  * 部分地解决1和3
  * 和思考一致：高优先级，把time slice调短
  * Solaris：Default values for the table are 60 queues, with slowly increasing time-slice lengths from 20 milliseconds (highest priority) to a few hundred milliseconds (lowest),and priorities boosted around every 1 second or so，

attempt3: better accounting

* Rule 4:Once a job uses up its time allotment at a given level (regardless of how many times it has given up the CPU), its priority is reduced (i.e., it moves down one queue).

其它可能的特性：
* 操作系统0优先级
* advice机制：As the operating system rarely knows what is best for each and every process of the system, it is often useful to provide interfaces to allow usersor administrators to provide some hints to the OS. We often call such hints advice, as the OS need not necessarily pay attention to it, but rather might take the advice into account in order to make a better decision. Such hints are useful in many parts of the OS, including the scheduler(e.g., with **nice**), memory manager (e.g.,**madvise**), and file system (e.g.,informed prefetching and caching [P+95])


* HW: iobump，io结束后把进程调到当前队列第一位，否则最后一位；io越多效果越好

#### 9.Scheduling: Proportional share

##### 应用： C++ 进程/线程优先级

[实时和非实时调度策略测试总结](https://www.cnblogs.com/mightycode/p/13930352.html)

* Linux内核的三种调度策略：
  * SCHED_OTHER 分时调度策略
  * SCHED_FIFO实时调度策略，先到先服务。一旦占用cpu则一直运行。一直运行直到有更高优先级任务到达或自己放弃
  * SCHED_RR实时调度策略，时间片轮转。当进程的时间片用完，系统将重新分配时间片，并置于就绪队列尾。放在队列尾保证了所有具有相同优先级的RR任务的调度公平
  * `pthread_setschedparam(id, policy, &param)`

* [nice 可用于设置 Linux 线程优先级](https://stackoverflow.com/questions/7684404/is-nice-used-to-change-the-thread-priority-or-the-process-priority)，因为 Linux Threads do not share a common nice value，violate 了 POSIX.1
  * `setpriority()`, `nice()`
  * 提高优先级需要 sudo 权限，有 setcap 的方法绕过，参考 [stackoverflow](https://stackoverflow.com/questions/41562834/linux-set-priority-function-is-not-taking-effect-in-my-test)




##### CRUX: how to share the CPU proportionally

Basic Concept: Tickets Represent Your Share
* 利用randomness：
  1. 避免corner case，LRU replacement policy (cyclic sequential)    
  2. lightweight    
  3. fast，越快越伪随机

NOTE：

如果对伪随机数限定范围，不要用rand，[用interval](https://stackoverflow.com/questions/2509679/how-to-generate-a-random-integer-number-from-within-a-range)

**机制：**

1. ticket currency，用户之间
2. ticket transfer，用户与服务器
3. ticket inflation，临时增加tickets，需要进程之间的信任

* unfairness metric
* stride scheduling    — deterministic
  * lottery scheduling 相对于 stride 的优势：no global state

```c++
curr = remove_min(queue);   // pick client with min pass
schedule(curr);             // run for quantum
curr->pass += curr->stride; // update pass using stride
insert(queue, curr);        // return curr to queue
```

9.7 The Linux Completely Fair Scheduler(CFS) 完全公平调度器
* 引入 vruntime的概念，记录进程运行时间
* 引入sched_latency=48ms， time slice=48/n，保证均分
  * min_granularity=6ms，防止n太大sched_latency过低的情况
  * Weighting (Niceness) --->time slice ; table对数性质，ratio一致
* 用红黑树储存进程节点
  * sleep，则从树中移去
  * 关于I/O，回来后设成树里的最小值

NOTE:
* 这个idea应用广泛，比如用于虚拟机的资源分配
* [why index-0?](https://www.cs.utexas.edu/users/EWD/ewd08xx/EWD831.PDF) 

#### 10.Multiprocessor Scheduling (Advanced)
* 概念：multicore processor       与threads配合

##### CRUX: how to schedule jobs on multiple CPUs

**Background: Multiprocessor Architecture**

单核与多核的区别：the use of hardware caches(e.g., Figure 10.1), and exactly how data is shared across multiple processors. 

Q: cache coherence问题，不同CPU的cache未及时更新，导致读错数据

A: 利用hardware, by monitoring memory accesses: bus snooping; write-back caches

**Don’t Forget Synchronization**

虽然解决了coherence问题，依然需要mutual exclusion primitives(locks)

**One Final Issue: Cache Affinity**

**Single-Queue Scheduling**

SQMS (single-queue multiprocessor scheduling)，存在的问题如下：

* lack of scalability: lock overhead随CPU数目增加而变高
* cache affinity: 需要用复杂的机制调度，比如将少量jobs migrating from CPU to CPU

->

MQMS (multi-queue multiprocessor scheduling): 分配jobs给CPU

* 优点：more scalable; intrinsically provides cache affinity
* 缺点：load imbalance

##### CRUX: how to deal with load imbalance?

migration: 将余数migrate

the tricky part: how should the system decide to enact such a migration?

e.g. work stealing: source queue经常peek其它的target queue，使两边work load均衡

* peek频率是超参数，过高会失去MQMS的意义，过低会有load imbalances

##### Linux Scheduling

* Linux Multiprocessor Schedulers

  * O(1) scheduler: priority-based scheduler

  * CFS (Completely Fair Scheduler): a deterministic proportional-share approach

  * BFS (BF Scheduler): also proportional-share, but based on a more complicated scheme known as Earliest Eligible Virtual Deadline First (EEVDF) 

  * BFS是single queue，前两个是multiple queues


* Linux中，每个CPU都有一个自己的本地队列（红黑树），用来存放等待这个CPU资源的，已就绪待运行的tasks。正在运行的task，主要有两种机制离开CPU。
  - 自愿抢占：通过调用sleep, lock, io等主动让出CPU；
  - 非自愿抢占: 被优先级更高的task抢占或者时间片到期，被动的离开CPU。
* 显然，task如果在CPU队列上等待的时间较长，一定会影响请求延迟。 但事实上Linux的CPU调度算法是非常优秀的，而且CPU是一种可以抢占的资源，因此通常即使在一个比较高的CPU利用率下，延迟也不会急剧上升（和磁盘IO调度相比）。

![fcfdaf2f-b76b-47d9-a49f-ca74513cb09f](OSTEP-Operating-Systems-Three-Easy-Pieces/cpu-scheduling.png)



#### 11.summary

#### 12.A dialogue on memory virtualization
every address generated by a user program is a virtual address
  * ease of use, isolation, protection

#### 13.The Abstraction: Address Spaces
* multiprogramming
* abstraction: address space
##### CRUX：how to virtualize memory
* virtual memory            
  * goals：transparency, efficiency (e.g. TLBs), protection
  
  * location of code : 0x105f40ec0
  
  * location of heap : 0x105f55000
  
  * location of stack: 0x7ffee9cbf8ac

*  [64bit系统下进程的内存分布](https://blog.csdn.net/chenyijun/article/details/79441166)
  * Linux64位操作系统仅使用低47位，高17位做扩展（只能是全0或全1）。所以，实际用到的地址为256TiB，空间为0x0000000000000000 ~ 0x00007FFFFFFFFFFF（user space）和0xFFFF800000000000 ~ 0xFFFFFFFFFFFFFFFF（kernel space）,其余的都是unused space。
  * 对于 32-bit Linux，一个进程的地址空间是 4GiB，其中用户态能访问 3GiB 左右， 而一个线程的默认栈(stack)大小是 10MB，心算可知，一个进程大约最多能同时启动 300 个线程

NOTE:
* 用microkernels的思想实现isolation，机制和策略的分离

#### 14.Interlude: Memory API
##### CRUX:how to allocate and manage memory
* 64bit UNIX系统，int和double都是8个字节
* automatic memory management         ~ garbage collector
* 其它calls：calloc()先置0，realloc()更大区域

**一些常见错误：**

* segmentation fault    =>用strdup    ###
* buffer overflow        e.g. 应该strlen(src)+1
* uninitialized read/undefined value    ###
* memory leak            针对long-running server，是OS层面的错误
* dangling pointer
* double free   ###
* invalid frees
* strcat的参数内存区域重复    ###
* （###：valgrind可检测的）
* 用[purify](https://www.cnblogs.com/Leo_wl/p/7699489.html)和valgrind检查内存泄漏
* 底层基础：
  * [C++ Memory Allocation/Deallocation for Data Processing](https://towardsdatascience.com/c-memory-allocation-deallocation-for-data-processing-1b204fb8a9c)
  * [brk](https://man7.org/linux/man-pages/man2/brk.2.html), sbrk: adjusting the Program Break which is the current heap limit, 不要自己调用
  * mmap 内存映射

HW:
* null.c    segmentation fault
* forget_free.c    lldb没反应
* dangling_pointer.c    直接run会print出0
* free_wrong.c    int *类型的+操作符重载过，直接加数字，不用乘sizeof(XXX)

#### 15.Mechanism: Address Translation
efficiency and control
##### CRUX: how to efficiently and flexibly virtualize memory
hardware-based address translation

dynamic (hardware-based) relocation=base and bounds 
  * physical address = virtual address + base
  * base and bounds register，本质上是MMU(memory management unit)
  * bounds register: 两种方式，一是size，二是physical address

<->

static (software-based) relocation: loader，不安全，难以再次换位

一些硬件要素：寄存器，异常，内核态，privileged instructions
* 硬件和protection联系紧密

**OS需要的数据结构：**

* free list(定长进程内存)
* PCB(or process structure)            储存base和bounds信息
* exception handlers: 掐掉过界的进程

<img src="OSTEP-Operating-Systems-Three-Easy-Pieces/004.jpg" alt="LDE+Dynamic Relocation" style="zoom:60%;" />

问题：internal fragmentation，内存利用率不高 => segmentation

<img src="OSTEP-Operating-Systems-Three-Easy-Pieces/005.jpg" alt="Dynamic Relocation" style="zoom:80%;" />

#### 16.Segmentation

##### CRUX:how to support a large address space
* 意义：节省内存，不赋予全部虚拟地址空间以实体
* 概念：sparse address spaces、segmentation、segmentation fault
* 实现
  * explicit：利用前两位；也可只利用一位，把code和heap合并
  * implicit：利用计组知识，比如PC生成的地址属于code区
  * 对于stack的特殊处理：negative offset

**support for sharing**

* code sharing    这是一个潜在的好处
* protection bits (硬件支持)

* fine-grained segmentation: segment table
* coarse-grained

**OS support**

* segment registers
* malloc
* manage free space 
  * 问题：external fragmentation
  * 方案1: compaction
    * 消耗大
    * makes requests to grow existing segments hard to serve
  * 方案2：free-list：best-fit，worst-fit，first-fit(address-based ordering，利于coalesce)，[buddy algorithm](https://blog.csdn.net/wan_hust/article/details/12688017) (块链表，合并) 



* [What is memory fragmentation?](https://stackoverflow.com/questions/3770457/what-is-memory-fragmentation)
  * virtual memory下的memory fragmentation问题没那么严重
  * 自己维护内存池可能还是有必要的，相当于利用了业务先验知识（某些内存块是一起分配一起释放的、以及能感知locality）
  * It's when you have mixtures of short-lived and long-lived objects that you're most at risk, but even then `malloc` will do its best to help.

* [How to solve Memory Fragmentation](https://stackoverflow.com/questions/60871/how-to-solve-memory-fragmentation)
  * perf **free** blocks in the heap的情况
  * logically divide the heap into sub arenas where the lifetimes are more similar
  * to attempt to make the allocation sizes more similar or identical
    * **unnecessary** because the allocator will already be doing a power of two size bucketing scheme





#### 17.Free-Space Management
##### CRUX: how to manage free space
* 重点是external fragmentation
* memory分配给user后，禁止compaction

Low-level mechanisms: 运用了以下机制
* Splitting and Coalescing

* Tracking the size of allocated regions
  
  * `void free(void*ptr) {header_t*hptr = (header_t*) ptr - 1;}`
  
  * size 和 magic；寻找 chunk 的时候需要加上头的大小
  
* Embedding A Free List  
  * `typedef struct __node_t {int size; struct __node_t *next;} node_t;`
* Growing The Heap

Other Approaches: (这些 approaches 的问题在于 lack of scaling)
* segregated list，针对高频的size
  * slab allocator： 利用了这个特性，object caches，pre-initialized state
* binary buddy allocator

并行优化：Hoard，jemalloc (radix trees 存 metadata)

#### 18.Paging: Introduction

另一条路径，page frames，never allocating memory in variable-sized chunks

##### CRUX: how to virtualize memory with pages
* flexibility and simplicity
* page table: store address translations
  * inverted page table不是per-process结构，用哈希页表，可以配合TLB

translate: virtual address= virtual page number(VPN) + offset

 => PFN(PPN): physical frame(page) number

18.2 Where are page tables stored?
* page table大，仅用hardware MMU难以管理，作为virtualized OS memory，存在physical memory里，甚至可以存进swap space

**PTE: page table entry**

* valid bit：x86的实现中，没有valid bit，由OS利用额外的结构决定，present bit=0的page，是否valid，即是否需要swapped back in 
* protection bits    
* present bit (e.g. swapped out)    
* dirty bit    
* reference bit (accessed bit) ~ page replacement    

* 实际存储：VirtualAddress：32=20(VPN)+12(Offset)；PTE内部：20(PFN)+3(empty)+9(flag)

<img src="OSTEP-Operating-Systems-Three-Easy-Pieces/006.jpg" alt="accessing memory with paging" style="zoom:80%;" />

#### 19.Paging: Faster Translations(TLBs)

##### CRUX: how to speed up address translation

**TLB: translation-lookaside buffer**

* 属于MMU，是address-translation cache
* TLB hit/miss 
* cache：spatial and temporal locality ; locality 是一种 heuristic
* [TLB是全相联cache](https://my.oschina.net/fileoptions/blog/1630855)

TLB Control Flow Algorithm
```c++
VPN = (VirtualAddress & VPN_MASK) >> SHIFT
(Success, TlbEntry) = TLB_Lookup(VPN) 
if(Success == True) // TLB Hit
    if (CanAccess(TlbEntry.ProtectBits) == True)
        Offset   = VirtualAddress & OFFSET_MASK
        PhysAddr = (TlbEntry.PFN << SHIFT) | Offset
        Register = AccessMemory(PhysAddr)
    else
        RaiseException(PROTECTION_FAULT)
else                  // TLB Miss
    PTEAddr = PTBR + (VPN*sizeof(PTE))
    PTE = AccessMemory(PTEAddr)
    if (PTE.Valid == False)
        RaiseException(SEGMENTATION_FAULT)
    else 
    		if (CanAccess(PTE.ProtectBits) == False)
        		RaiseException(PROTECTION_FAULT)
        else
            TLB_Insert(VPN, PTE.PFN, PTE.ProtectBits)
            RetryInstruction()
```
OS-handled，实现细节：
* 普通的return-from-trap回到下条指令，TLB miss handler会retry，回到本条指令
* 防止无限循环：trap handler放进physical memory，或者对部分entries设置wired translations

```c++
VPN = (VirtualAddress & VPN_MASK) >> SHIFT
(Success, TlbEntry) = TLB_Lookup(VPN)
if (Success == True)   // TLB Hit
    if (CanAccess(TlbEntry.ProtectBits) == True)
        Offset   = VirtualAddress & OFFSET_MASK
        PhysAddr = (TlbEntry.PFN << SHIFT) | Offset
        Register = AccessMemory(PhysAddr)
    else
        RaiseException(PROTECTION_FAULT)
else                  // TLB Miss
    RaiseException(TLB_MISS)
```

19.3 Who Handles The TLB Miss?

* Aside: RISC vs CISC
* CISC:    x86: hardware-managed    
  * multi-level page table
  * 硬件知道PTBR
  *  the current page table is pointed to by the CR3 register [I09]
* RISC:    MIPS: software-managed

ASIDE: TLB Valid Bit和Page Table Valid Bit的区别：
1. PTE和新进程密切相联。
2. context switch时把TLB valid bit置0

19.5 TLB Issue: Context Switches
##### CRUX: how to manage TLB contents on a context switch
* solution1: flush the TLB    
  * 对于硬件实现，PTBR的变化后flush the TLB
* solution2: ASID(address space identifier)  8bit ,性质上类似于32bit的PID 
* NOTE: 可能存在进程间的 sharing pages，比如库或者代码段

**Issue: cache replacement policy**
##### CRUX: how to design TLB replacement policy
* LRU， 会有corner-case behaviors
* random policy

* MIPS 的 TLBs 是 software-managed，一个 entry 64bit，有  32或64个 entry，会给OS预留，比如用于 TLB miss handler

<img src="OSTEP-Operating-Systems-Three-Easy-Pieces/007.jpg" alt="A MIPS TLB Entry" style="zoom:80%;" />

* MIPS TLB 相关的四个 privileged OS 命令：TLBP(probe), TLBR(read), TLBWI(replace specific), TLBWR(replace random)
* Culler's Law：TLB经常是性能瓶颈

* Issue: exceeding the TLB converge => larger pages，应用于DBMS
* Issue: physically-indexed cache成为bottleneck => virtually-indexed cache [W03]
  * in the CPU pipeline, with such a cache, address translation has to take place before the cache is accessed（计组知识）

* HW:测量NUMPAGES，UNIX: getpagesize()=4096

#### 20.Paging: Smaller Tables
##### CRUX: how to make page tables smaller?
bigger pages, multiple page sizes, DBMS

**hybrid approach: paging and segments**
* hybrid的思想，尤其针对看似对立的机制
* e.g. Multics
* base: physical address of the page table      , limit:how many valid pages
* 有三个page tables=>三个base registers而不是一个
* issue：
  1. page table的大小可变，与内存相联系，重新产生了external segmentation
  2. 不灵活，比如不针对堆很稀疏的情形

```c++
SN = (VirtualAddress & SEG_MASK) >> SN_SHIFT
VPN = (VirtualAddress & VPN_MASK) >> VPN_SHIFT
AddressOfPTE = Base[SN] + (VPN*sizeof(PTE))
```
**multi-level page tables**
* page directory    好处：加入了level of indirection，更灵活    坏处：Time-space trade-off
* PDE(page directory entry)
* 如何分组：page size/PTE size ~ n位    n位一组即可

**inverted page tables**
* 本质上是个数据结构问题
* size ～ 物理页数 < 进程数*虚拟页数
* swapping the page tables to disk: VAX/VMS

#### 21.Beyond Physical Memory: Mechanisms
##### CRUX: how to go beyond physical memory
* hard disk drive
* 与single address space对立的旧机制：memory overlays
* swap space：在硬盘上，disk address
* the present bit
  * 0的意义：page fault handler         
  * 为什么称作fault？是因为硬件无法处理，需要raise an exception交给OS

page replacement policy

* Page-Fault Control Flow Algorithm (Hardware)
```c++
VPN = (VirtualAddress & VPN_MASK) >> SHIFT
(Success, TlbEntry) = TLB_Lookup(VPN) 
if(Success == True) // TLB Hit
    if (CanAccess(TlbEntry.ProtectBits) == True)
        Offset   = VirtualAddress & OFFSET_MASK
        PhysAddr = (TlbEntry.PFN << SHIFT) | Offset
        Register =AccessMemory(PhysAddr)
    else
        RaiseException(PROTECTION_FAULT)
else                  // TLB Miss
    PTEAddr = PTBR + (VPN*sizeof(PTE))
    PTE =AccessMemory(PTEAddr)
    if (PTE.Valid == False)
        RaiseException(SEGMENTATION_FAULT)
    else 
        if (CanAccess(PTE.ProtectBits) == False)
            RaiseException(PROTECTION_FAULT)
        else if (PTE.Present == True)// assuming hardware-managed TLB
            TLB_Insert(VPN, PTE.PFN, PTE.ProtectBits)
            RetryInstruction()
        else if (PTE.Present == False)
            RaiseException(PAGE_FAULT)
```

* Page-Fault Control Flow Algorithm (Software)
```c++
PFN = FindFreePhysicalPage()
if(PFN == -1) // no free page found
    PFN = EvictPage()       // run replacement algorithm
DiskRead(PTE.DiskAddr, PFN)// sleep (waiting for I/O)
PTE.present = True          // update page table with present
PTE.PFN     = PFN           // bit and translation (PFN)
RetryInstruction()          // retry instruction 
```

when replacements really occurs
* swap(page) daemon的任务：可用页数低于LW(low watermark)时free到HW
  * cluster a number of pages
* daemon：守护进程
  * 可以救活coreaudiod这种进程
* idle time: background，比如把文件写入memory而非disk
* 总结：以上这些，对process是透明的

HW:

[vmstat命令](https://www.cnblogs.com/ftl1012/p/vmstat.html)

* vmstat 1 显示每秒状态
1. 运行多个，user time变大，idle time 变少
2. 运行1024MB，swpd不变，free减少，exit之后还原
3. cat /proc/meminfo 可用内存132GB，运行巨量mem会core dumped，in(中断时间)明显增加，偶尔会有sy(system time)
5. swapon -s，显示可供swap的大小 ，相当于 cat /proc/swaps

#### 22.Beyond Physical Memories: Policies
##### CRUX: how to decide which page to evict
* 本质上是cache management
* 评价指标average memory access time：$ AMAT = T_{M}+(P_{Miss} · T_{D}) $，hit rate很重要

the optimal replacement policy
* Belady: furtherest in the future

ASIDE: types of cache misses:    Three C’s: compulsory, capacity, conflict
* OS page cache是全相联，不会发生conflict miss

A simple policy: FIFO
* Belady’s Anomaly: FIFO，cache size高可能反而不好，因为没有stack property(N+1-cache和N-cache的包含关系)，不像LRU有这个性质

another simple policy : random

Using History: LRU    least-recently-used            
* frequency and recency            LFU
* 其它变种：scan resistance

workload有几种：随机，80-20（适合LRU），looping sequential（适合RAND）

##### CRUX: how to implement an LRU replacement policy
关于实现：approximating LRU
* use bit(reference bit) 
* clock algorithm: 循环数组，遇到1置为0，遇到0置换
  * 效果比其它的好，只比LRU差一点
  * 改进：考虑dirty bits，先evict unused and clean pages，否则swap损耗大

其它policy：
* page selection: demand paging/prefetching
* clustering(grouping) of writes

thrashing: memory is oversubscribed, demands > physical memory
* 方法一：admission control，控制working sets的大小
* 方法二：out-of-memory killer，       潜在的问题：kill X server

#### 23.Complete Virtual Memory Systems

##### CRUX: how to build a complete VM system

##### VAX/VMS virtual memory

DEC发明

存在的问题：
* 需要覆盖的机器类型range太宽，the curse of generality    
* 有inherent flaws

page+segmentation

Q：page太小，512bytes，如何解决内存压力？

1. hybrid approach: 引入segmentation    
2. 利用内核memory    
3. 与2联系，利用TLB缓解复杂机制带来的损耗 

<img src="OSTEP-Operating-Systems-Three-Easy-Pieces/008.jpg" alt="具体实现" style="zoom:50%;" />

NOTE：
* page 0: in order to provide some support for detecting null-pointer accesses
* kernel is mapped into each address space

有关page replacement:
*  no reference bit
* 针对memory hog：segmented FIFO，给每个进程规定一个RSS(resident set size)，超出这个范围要FIFO
* second-chance lists： clean-page list和dirty-page list，从clean开始evict
* clustering

other neat tricks:
* demand zeroing: 等到进程要用page，再交给OS给page置0
* COW: copy-on-write，和UNIX的fork() exec()机制结合
* 核心思想是be lazy: 好处一是提高responsiveness，二是可能obviate the need to do things at all

##### the Linux virtual memory system


内核、用户部分的内存分配，大体沿用以前，区别在于内核分为两部分：
* kernel logical addresses:
  * kmalloc，存page tables, per-process kernel stacks，不能swap到disk
  * direct mapping to physical addresses  : 1) simple to translate，0xC0000000变0x00000000；2) contiguous，适合于DMA
* kernel virtual addresses: vmalloc，不连续，用于large buffers，可以让32bit系统处理超过1GB的memory
* 0xC0000000开始是内核

64-bit x86：
<img src="OSTEP-Operating-Systems-Three-Easy-Pieces/009.jpg" alt="64-bit x86" style="zoom:50%;" />

large page support
* explicit的支持：mmap, shmget    => transparent huge page support
* 对 TLB 好处大，miss rate 和 path 都降低成本
* 缺点：internal fragmentation; swapping效果不好
* 体现了 incrementalism，慢慢引入特性并迭代

the page cache
* 主要来源：memory-mapped files, file data and metadata from devices , and heap and stack pages that comprise each process (anonymous memory)
* pdflush：背景线程，把dirty data写入backing store
* 2Q replacement：inactive list不时加入active list，解决大文件频繁访问的问题
* memory mapping 体现在 linux 的方方面面: 用 pmap 命令查看

**security相关**

[buffer overflow](https://en.wikipedia.org/wiki/Buffer_overflow)

[smashing the stack for fun and profit](https://insecure.org/stf/smashstack.html)
* 概念：privilege escalation
* 对策：NX bit page禁止执行，针对stack

return-oriented programming (ROP)
* 对策：address space layout randomization(ASLR).    even  KALSR
  * macOS上，调整[27:12]16位，随机

Other Security Problems: Meltdown And Spectre
* meltdownattack.com    spectreattack.com
* 思想：speculation execution会暴露内存信息
* 对策：kernel page-table isolation (KPTI)

《cloud atlas》quote: “My life amounts to no more than one drop in a limitless ocean. Yet what is any ocean, but a multitude of drops?”


#### 24.Summary

### Concurrency

#### 25.A Dialogue on Concurrency

#### 26.Concurrency: An Introduction

* Amdahl's Law: 1/(1-p)
  * 95% work ~ infinite concurrency, the theoretical speedup is limited to at most 20 times the single thread performance

概念：thread, multi-threaded, thread control blocks (TCBs)
  * thread-local: 栈不共用，在进程的栈区域开辟多块栈，不是递归的话影响不大
    * [关于线程栈和进程栈](https://www.cnblogs.com/luosongchao/p/3680312.html)
      * 线程栈是固定大小的（默认8KB），可以使用`ulimit -a` 查看，使用`ulimit -s` 修改
      * 进程栈大小时执行时确定的，与编译链接无关
      * 进程栈大小是随机确认的，至少比线程栈要大，但不会超过2倍

  * thread的意义：1) parallelism, 2) 适应于I/O阻塞系统、缺页中断（需要KLT），这一点类似于multiprogramming的思想，在server-based applications中应用广泛。

<img src="OSTEP-Operating-Systems-Three-Easy-Pieces/015.jpg" alt="015" style="zoom:50%;" />

NOTE：
* pthread_join与[detach](https://blog.csdn.net/heybeaman/article/details/90896663)
* disassembler: objdump -d -g main
* x86，变长指令，1-11个字节

问题：线程之间会出现data race，e.g. counter的例子
* 经典面试题：两个线程，全局变量i++各运行100次，问运行完i的最小值。
  * 答案是2
```
进程2取 i=0
进程1执行99次
进程2算 i+1=0+1=1
进程1取 i=1
进程2执行99次
进程1算 i+1=1+1=2
```

引入概念：critical section, race condition, indeterminate, mutual exclusion 

=> the wish for atomicity

* transaction: the grouping of many actions into a single atomic action 
* 和数据库, journaling、copy-on-write联系紧密
* 条件变量：用来等待而非上锁

##### CRUX: how to support synchronization
* the OS was the first concurrent program!
* Not surprisingly, pagetables, process lists, file system structures, and virtually every kernel data structure has to be carefully accessed, with the proper synchronization primitives, to work correctly.

HW26:
* data race来源于线程保存的寄存器和stack
* 验证了忙等待的低效

#### 27.Interlude: Thread API
##### CRUX: how to create and control threads
```c++
#include <pthread.h>
int pthread_create(pthread_t*thread,const pthread_attr_t*attr,void*(*start_routine)(void*),void*arg);

typedef struct
{
    int a;
    int b;
} myarg_t;
typedef struct
{
    int x;
    int y;
} myret_t;
void * mythread(void *arg)
{
    myret_t *rvals = Malloc(sizeof(myret_t));
    rvals->x = 1;
    rvals->y = 2;
    return(void *)rvals;
}
int main(int argc, char *argv[])
{
    pthread_t p;
    myret_t * rvals;
    myarg_t args = {10, 20};
    Pthread_create(&p, NULL, mythread, &args);
    Pthread_join(p, (void **)&rvals);
    printf("returned %d %d\n", rvals->x, rvals->y);
    free(rvals);
    return 0;
}
```
* pthread_create
  * thread: &p
  * attr：传参NULL或，pthread_attr_init
  *  arg和start_routine的定义保持一致；
  * void=any type
* pthread_join
  * simpler argument passing：`(void *)100, (void **)rvalue`
  * `(void **)value_ptr`，小心局部变量存在栈中，回传指针报错
* gcc -o main main.c -Wall -pthread

##### lock
```c++
pthread_mutex_t lock;
pthread_mutex_lock(&lock);
x = x + 1; // or whatever your critical section is
pthread_mutex_unlock(&lock);
```
* ` pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;`
* `int rc = pthread_mutex_init(&lock, NULL);		assert(rc == 0); // always check success!`
* pthread_mutex_trylock和timedlock

##### conditional variables
```c++
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  cond = PTHREAD_COND_INITIALIZER;
Pthread_mutex_lock(&lock);
while (ready == 0) Pthread_cond_wait(&cond, &lock);
Pthread_mutex_unlock(&lock);
```

HW:
* main-race.c:
  * `valgrind --tool=helgrind ./main-race`，结果给出了“Possible data race during write of size 4 at 0x30A014 by thread #1”
  * 全局变量存放在数据段
* 误判了main-deadlock-global-c，说明有瑕疵

* main-signal-cv.c  条件变量的用法示例
```c++
#include <stdio.h>
#include "mythreads.h"
// 
// simple synchronizer: allows one thread to wait for another
// structure "synchronizer_t" has all the needed data
// methods are:
//   init (called by one thread)
//   wait (to wait for a thread)
//   done (to indicate thread is done)
// 
typedef struct __synchronizer_t {
    pthread_mutex_t lock;
    pthread_cond_t cond;
    int done;
} synchronizer_t;

synchronizer_t s;

void signal_init(synchronizer_t *s) {
    Pthread_mutex_init(&s->lock, NULL);
    Pthread_cond_init(&s->cond, NULL);
    s->done = 0;
}

void signal_done(synchronizer_t *s) {
    Pthread_mutex_lock(&s->lock);
    s->done = 1;
    Pthread_cond_signal(&s->cond);
    Pthread_mutex_unlock(&s->lock);
}

void signal_wait(synchronizer_t *s) {
    Pthread_mutex_lock(&s->lock);
    while (s->done == 0) Pthread_cond_wait(&s->cond, &s->lock);
    Pthread_mutex_unlock(&s->lock);
}

void* worker(void* arg) {
    printf("this should print first\n");
    signal_done(&s);
    return NULL;
}

int main(int argc, char *argv[]) {
    pthread_t p;
    signal_init(&s);
    Pthread_create(&p, NULL, worker, NULL);
    signal_wait(&s);
    printf("this should print last\n");

    return 0;
}

```

#### 28.Locks
* 概念：lock, owner, 
* locks: 利用OS来schedule线程，套在critical section两边，有available(unlocked, free)和acquired(locked, held)两种状态
* POSIX的mutex

##### CRUX: how to build a lock
evaluating locks
* correctness (mutual exclusion), fairness (starve), performance
* 需要考虑starve，设不设置contending
* 多核问题
* 定义锁的精细化程度：coarse和fine-grained lock

解决互斥问题应遵循的条件
1. 任何两个进程不能同时处于临界区
2. 不应对CPU的速度和数量做任何假设 
3. 临界区外运行的进程不得阻塞其他进程 
4. 不得使进程无限期等待进入临界区

##### 禁止中断 interrupt masking
* 优点：简单
* 把禁止中断的权利交给用户进程导致系统可靠性较差
* 不适用于多处理器(违反条件2)
* 可能错过其它interrupts，比如disk的read request
* 对于现代CPU，这个实现速度慢
* 应用场景：OS内部数据结构的互斥访问

##### 共享锁变量 just using loads/stores
```c++
while(lock==1);
lock=1;
//critical region
lock=0;
//non_critical region;
```
* 违反条件1（interleaving）
* 忙等待（spin-waiting）
* Murphy's law

##### Building Working Spin Locks with Test-And-Set
`while (TestAndSet(&lock->flag, 1) == 1);`
* test-and-set(atomic exchange)：把返回原值+修改值这两个操作绑定
  * xchg(x86), ldstub(SPARC)
  * 可以test-and-test-and-set，只有当flag为0才改变锁。
* spin lock，要求preemptive scheduler，抢占式调度
  * 满足correctness
  * 不满足fairness和performance
  * 在多核处理器上表现良好，因为当前的线程可以很快通过critical section，不需要多次spin，上下文切换

##### DEKKER’S AND PETERSON’S ALGORITHMS
```c++
#define FALSE 0
#define TRUE  1
#define N     2                                           /* 进程数量 */
  
int turn=0;                                 /* 现在轮到谁？*/
int interested[N];               /* 所有值初始化为0（FALSE）*/

void enter_region(int process)               /* 进程是0或1 */
{
	interested[process] = TRUE;              /* 表名所感兴趣的*/
	turn = 1-process;                            /* 设置标志 */
	while(turn == 1-process && interested[1-process] ==TRUE); /* 空语句 */
}

void leave_region(int process)                    
{
	interested[process] = FALSE;             /* 表示离开临界区*/ 
}
```
##### Compare-And-Swap
* 这是SPARC上的叫法，在x86上称作compare-and-exchange
* 对于简单的同步问题，效果和test-and-set一样
* [利用CAS实现lock-free和wait-free](https://www.jianshu.com/p/baaf53d69b51)
  * 实现了lock-free的ATM存钱问题
  * ABA(AtomicStampedReference)问题：需要用版本戳version标记
  * 相关论文：[Wait-Free Synchronization](http://www.cs.utexas.edu/users/lorenzo/corsi/cs380d/papers/p124-herlihy.pdf)


##### Load-Linked and Store-Conditional
* [MIPS处理器](http://groups.csail.mit.edu/cag/raw/documents/R4400_Uman_book_Ed2.pdf)

```c++
typedef struct __lock_t{
	int flag;
} lock_t;

void init(lock_t *lock){
	//0:lock is available, 1:lock is held
	lock->flag = 0;
}

int LoadLinked(int *ptr)
{
	return *ptr;
}
int StoreConditional(int*ptr, int value) {
	if (no update to *ptr since LoadLinked to this address) {
		*ptr = value;
		return 1; // success!
	} 
	else {
		return 0; // failed to update
	}
}
void lock(lock_t *lock){
	while (LoadLinked(&lock->flag) ||!StoreConditional(&lock->flag, 1));
}
void unlock(lock_t *lock){
	lock->flag = 0;
}
```

##### Fetch-And-Add
* ticket lock
  * 优点：ensure progress for all threads, 线程一定会被调度到
```c++
typedef struct __lock_t {
    int ticket;
    int turn;
} lock_t;
void lock_init(lock_t*lock) {
    lock->ticket = 0;
    lock->turn   = 0;
}
void lock(lock_t*lock) {
    int myturn = FetchAndAdd(&lock->ticket);
    while (lock->turn != myturn); // spin
}
void unlock(lock_t*lock) {
    lock->turn = lock->turn + 1;
}
```

##### CRUX: how to avoid spinning

##### 方法一：just yield
* 效率问题没解决，并且仍然有starvation问题

```c++
void lock() {
    while (TestAndSet(&flag, 1) == 1)
        yield(); // give up the CPU, running->ready
}
```

##### 方法二：Using Queues: Sleeping Instead Of Spinning
* OS support: park(), unpark() (Solaris)
* 利用guard，虽然也有一定的spin lock损耗，但不涉及critical section，损耗较小
* Q1：wakeup/waiting race：在park之前切换上下文
* A1: 1）利用setpark(); 2)guard传入内核，可能类似后面futex的实现
* Q2：[priority inversion](https://en.wikipedia.org/wiki/Priority_inversion): 高优先级线程waiting低优先级线程，可能因为spin lock或者存在**中优先级线程**而无法运行。
* A2: 1）priority inheritance; 2）所有线程平等；3）Priority ceiling protocol；4）Random boosting；5）Avoid blocking

##### 方法三：Linux的futex，更多内核特性
* OS support: per-futex in-kernel queue
* [nptl库](http://ftp.gnu.org/gnu/glibc/)中lowlevellock.h的代码片段：
  * mutex的巧妙设计

```c++
void mutex_lock (int*mutex) {
    int v;
    /*Bit 31 was clear, we got the mutex (the fastpath)*/
    if (atomic_bit_test_set (mutex, 31) == 0)
        return;
    atomic_increment (mutex);
    while (1) {
        if (atomic_bit_test_set (mutex, 31) == 0) {
            atomic_decrement (mutex);
            return;
        }
        /*We have to wait
        First make sure the futex value 
        we are monitoring is truly negative (locked).*/
        v =*mutex;
        if (v >= 0)
            continue;
        futex_wait (mutex, v);
    }
}
void mutex_unlock (int*mutex) {
    /*Adding 0x80000000 to counter results in 0 if and
    only if there are not other interested threads*/
    if (atomic_add_zero (mutex, 0x80000000))
        return;
    /*There are other threads waiting for this mutex,
    wake one of them up.*/
    futex_wake (mutex);
}
```

##### 方法四：Two-Phase Locks
* 在futex之前spin不止一次，可以spin in a loop
* 思考：这又是一个hybrid approach（上一个是paging and segments）

#### 29.Lock-based Concurrent Data Structures
##### CRUX: how to add locks to data structures

**Concurrent Counters**
* 概念：thread safe, perfect scaling

**Scalable Counting： approximate counter**
* local counter和global counter，一个CPU配一个锁，再加上一个global锁
* threshold S: scalable的程度，local到global的transfer间隔
* 实现见[书上本章P5](http://pages.cs.wisc.edu/~remzi/OSTEP/threads-locks-usage.pdf)

[LWN上的文章](https://lwn.net/Articles/170003/) : 
* atomic_t变量, SMP-safe (SMP:Symmetrical Multi-Processing)，缺点在于锁操作的损耗、cache line频繁在CPU之间跳动
* approximate counter：缺点在于耗内存、低于实际值
* 进一步引入 local\_t，每个GPU设两个counter

**Concurrent Linked Lists**
* malloc error之后接unlock，这样的代码风格容易出问题。实际实现时推荐只在update数据结构的时候加锁，因为malloc具有thread safe特性。
* Tip：be wary of control flow changes that lead to function returns, exits, or other similar error conditions that halt the execution of a function
* hand-over-hand locking(lock coupling)：并发性强，但锁操作频繁，实际性能不见得好 

**Concurrent Queue**
* Michael and Scott Concurrent Queue: 1）头尾两个锁；2）头节点法
* A more fully developed bounded queue, that enables a thread to wait if the queue is either empty or overly full, is the subject of our intense study in the next chapter on condition variables.

**Concurrent Hash**
基于concurrent lists

```c++
#define BUCKETS (101)
typedef struct __hash_t {
    list_t lists[BUCKETS];
} hash_t;
void Hash_Init(hash_t*H) {
    int i;
    for (i = 0; i < BUCKETS; i++)
        List_Init(&H->lists[i]);
}
int Hash_Insert(hash_t*H, int key) {
    return List_Insert(&H->lists[key % BUCKETS], key);
}
int Hash_Lookup(hash_t*H, int key) {
    return List_Lookup(&H->lists[key % BUCKETS], key);
}
```

**premature optimization** (Knuth's Law)

* linux、Sun OS这样成熟的OS，为了规避这一问题，也是一开始只用 big kernel lock(BKL)，等多核瓶颈出现后再做优化。 《Understanding the Linux Kernel 》

#### 30.Condition Variables

##### CRUX: how to wait for a condition

* 概念：condition variable, wait/signal on the condition
* wait(): unlock, 然后让线程睡眠
* signal(): lock，返回caller

```c++
int done  = 0;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t c  = PTHREAD_COND_INITIALIZER;
void thr_exit() {
    Pthread_mutex_lock(&m);
    done = 1;
    Pthread_cond_signal(&c);
    Pthread_mutex_unlock(&m);
}
void*child(void*arg) {
    printf("child\n");
    thr_exit();
    return NULL;
}
void thr_join() {
    Pthread_mutex_lock(&m);
    while (done == 0)
        Pthread_cond_wait(&c, &m);
    Pthread_mutex_unlock(&m);
}
int main(int argc, char*argv[]) {
    printf("parent: begin\n");
    pthread_t p;
    Pthread_create(&p, NULL, child, NULL);
    thr_join();
    printf("parent: end\n");
    return 0;
}
```

* 代码中变量done的意义：state varibale，使程序的正确性不受两个线程运行先后顺序影响
* [关于条件变量需要互斥量保护的问题](https://www.zhihu.com/question/53631897)，pthread_cond_wait内部先解锁再等待，之所以加锁是防止cond_wait内部解锁后时间片用完。https://blog.csdn.net/zrf2112/article/details/52287915

##### The Producer/Consumer (Bounded Buffer) Problem
* bounded buffer的应用场景：HTTP requests的work queue；pipe

实现一：用一个条件变量+if实现
* 问题：Mesa semantics: there is no guarantee that when the woken thread runs, the state will still be as desired <-> Hoare semantics；前者广泛采用

实现二：一中的if改成**while**，尽量不被遗漏
* 多线程程序尽量用while来check条件，可以避免if条件满足时一次唤醒多个线程，spurious wakeups，资源不足
* 问题：signal不确定唤醒的是生产者还是消费者

实现三：while+两个条件变量


**Covering Conditions**：指需要唤醒过多的满足条件的线程的情形
* eg1: 针对memory allocator问题，直接用`pthread_cond_broadcast`唤醒所有wait中的线程，这是最简洁有效的思路
* eg2: 生产者消费者问题的实现一，也有这个问题，但可以从原理上进行改进，而eg1不方便进行原理上的改进，只能broadcast

HW:

2.`./main-two-cvs-while -l 10 -m 10 -p 1 -c 1 -v -t -C 0,0,0,0,0,0,1`

3.Linux switchs more often between producer and consumer than Mac

4.5.改m之后，由12/13秒到7秒

6.7.均是5秒，因为睡眠的时候释放了锁

9.`./main-one-cv-while -l 100 -p 1 -c 2 -m 1 -v -t`

#### 31.Semaphores

##### CRUX: how to use semaphores?

信号量和锁/条件变量的互相转换问题

```c++
#include <semaphore.h>
sem_t s;
sem_init(&s, 0, 1);
// second arg set to 0:the semaphore is shared between threads in the same process
// third arg: initial value

sem_wait(&m);
//critical section
sem_post(&m);

```
* 信号量初始化的值如何选取：consider the number of resources you are willing to give away immediately after initialization
* 信号量为负值时，绝对值是正在等待的线程数
* sem_post(&m)运行不停滞

信号量的应用

* Binary Semaphores

* Semaphores For Ordering
* The Producer/Consumer (Bounded Buffer) Problem
  * mutual exclusion
  * mutex在内层，否则会deadlock 

```c++
void*producer(void*arg) {
	int i;
	for (i = 0; i < loops; i++) {
		sem_wait(&empty);       // Line P1
		sem_wait(&mutex);       // Line P1.5 (MUTEX HERE)
		put(i);                 // Line P2
		sem_post(&mutex);       // Line P2.5 (AND HERE)
		sem_post(&full);        // Line P3
	}
}

void*consumer(void*arg) {
	int i;
	for (i = 0; i < loops; i++) {
		sem_wait(&full);        // Line C1
		sem_wait(&mutex);       // Line C1.5 (MUTEX HERE)
		int tmp = get();        // Line C2
		sem_post(&mutex);       // Line C2.5 (AND HERE)
		sem_post(&empty);       // Line C3
		printf("%d\n", tmp);
	}
}
```

* Reader-Writer Locks
  * 如果要保证公平竞争：设置互斥信号量S，加在读者/写者的acquire_lock函数上
  * 引申到设计理念，复杂往往低效，可能简单的spin lock更好；比如说CPU的cache设计，全相联比组相连效率高，部分是因为全相联实现的lookups更快

```c++
typedef struct _rwlock_t {
	sem_t lock;      // binary semaphore (basic lock)
	sem_t writelock; // allow ONE writer/MANY readers
	int   readers;   // #readers in critical section
} rwlock_t;

void rwlock_init(rwlock_t *rw) {
	rw->readers = 0;
	sem_init(&rw->lock, 0, 1);
	sem_init(&rw->writelock, 0, 1);
}

void rwlock_acquire_readlock(rwlock_t *rw) {
	sem_wait(&rw->lock);
	rw->readers++;
	if (rw->readers == 1) // first reader gets writelock
		sem_wait(&rw->writelock);
	sem_post(&rw->lock);
}

void rwlock_release_readlock(rwlock_t *rw) {
	sem_wait(&rw->lock);
	rw->readers--;
	if (rw->readers == 0) // last reader lets it go
		sem_post(&rw->writelock);
	sem_post(&rw->lock);
}

void rwlock_acquire_writelock(rwlock_t *rw) {
	sem_wait(&rw->writelock);
}
void rwlock_release_writelock(rwlock_t *rw) {
	sem_post(&rw->writelock);
}
```

* The Dining Philosophers
  * 需要解决的问题：死锁，哲学家同时拿到左手的餐具，**资源依赖成环**
  * 方法一：对每个fork设置信号量；如下面代码所示，修改其中一位哲学家的get_forks()避免成环
  * 方法二：对每个人设置信号量，定义test函数，在test的外围加互斥锁
```c++
#define NUM 5
while(1){
	think();
	get_forks();
	eat();
	put_forks();
}

int left(int p) {return p;}
int right(int p) {return (p+1)%NUM;}

void put_forks(int p){
	sem_post(&forks[left(p)]);
	sem_post(&forks[right(p)]);
}
void get_forks(int p){
	if(p==NUM){
		sem_wait(&forks[right(p)]);
		sem_wait(&forks[left(p)]);
	} else{
		sem_wait(&forks[left(p)]);
		sem_wait(&forks[right(p)]);
	}
}

```

* thread throttling

admission control, 比如针对memory-intensive region，避免thrashing(swap pages)  

* 如何实现信号量？
  * 用条件变量和互斥锁
  * 用信号量实现条件变量很难，书中有提到论文
```c++
typedef struct __Zem_t {
	int value;
	pthread_cond_t cond;
	pthread_mutex_t lock;
} Zem_t;

// only one thread can call this
void Zem_init(Zem_t *s, int value) {
	s->value = value;
	Cond_init(&s->cond);
	Mutex_init(&s->lock);
}

void Zem_wait(Zem_t *s) {
	Mutex_lock(&s->lock);
	while (s->value <= 0)
		Cond_wait(&s->cond, &s->lock);
	s->value--;
	Mutex_unlock(&s->lock);
}
void Zem_post(Zem_t *s) {
	Mutex_lock(&s->lock);
	s->value++;
	Cond_signal(&s->cond);
	Mutex_unlock(&s->lock);
}
```

HW:

5.reader-write-nostarve

```c++
void rwlock_acquire_readlock(rwlock_t *rw) {
    sem_wait(&rw->S);
    sem_post(&rw->S);
    sem_wait(&rw->lock);
    rw->readers++;
    if(rw->readers==1)
        sem_wait(&rw->writelock);
    sem_post(&rw->lock);
}

void rwlock_release_readlock(rwlock_t *rw) {
    sem_wait(&rw->lock);
    rw->readers--;
    if(rw->readers==0)
        sem_post(&rw->writelock);
    sem_post(&rw->lock);
}

void rwlock_acquire_writelock(rwlock_t *rw) {
    sem_wait(&rw->S);
    sem_wait(&rw->writelock);
}

void rwlock_release_writelock(rwlock_t *rw) {
    sem_post(&rw->S); //这行代码的位置有讲究，书上是放在这里，我觉得放在sem_post(&rw->writelock)后面或者sem_wait(&rw->writelock)前面好像都行
    sem_post(&rw->writelock);
}
```



6.no-starve-mutex

很难的问题，参考[The Little Book of Semaphores](https://www.docin.com/p-424286179.html) 4.3节

weak semaphore和strong semaphore：strong semaphore能确保在一个wait线程之前唤醒的线程数量有界

no-starve-mutex的目的是基于weak semaphore实现no starving，具体实现非常巧妙，设两个room，轮流全部倒出，这样就不会出现单线程的loop。

t1、t2和mutex三个信号量，状态转移图如下：

<img src="OSTEP-Operating-Systems-Three-Easy-Pieces/no-starve-mutex.jpeg" alt="进程状态转移" style="zoom:50%;" />

#### 32.Common Concurrency Problems

##### CRUX: how to handle common concurrency bugs?

##### Non-Deadlock Bugs

1.atomicity-violation bugs

```c++
Thread 1::
if (thd->proc_info) {
	fputs(thd->proc_info, ...);
}

Thread 2::
thd->proc_info = NULL;
```

2.Order-Violation Bugs

用条件变量解决

##### Deadlock Bugs

一个死锁tutorial：https://deadlockempire.github.io/

##### CRUX: how to deal with deadlock?

为什么会有出现死锁？
* large code bases, complex dependencies
* encapsulation，底层细节，比如Java Vector class `v1.AddAll(v2)`需要multi-thread safe，获取锁的顺序随机

**死锁条件**

* **Mutual exclusion:** Threads claim exclusive control of resources that they require (e.g., a thread grabs a lock)
* **Hold-and-wait: **Threads hold resources allocated to them (e.g., locks that they have already acquired) while waiting for additional resources (e.g., locks that they wish to acquire)
* **No preemption: **Resources (e.g., locks) cannot be forcibly removed from threads that are holding them
* **Circular wait: **There exists a circular chain of threads such that each thread holds one or more resources (e.g., locks) that are being requested by the next thread in the chain.

**Prevention**:分别针对上面的条件

Circular  wait 

* total ordering 固定锁的唤醒顺序

* partial ordering: linux filemap.c
* 小Tip：Enforce Lock Ordering by Lock Address

```c++
if (m1 > m2) { // grab in high-to-low address order
  pthread_mutex_lock(m1);
  pthread_mutex_lock(m2);
} else {
  pthread_mutex_lock(m2);
  pthread_mutex_lock(m1);
}// Code assumes that m1 != m2 (not the same lock)
```

Hold-and-wait

* 用一个prevention锁包住所有锁的获取，意义不大

No Preemption

* a deadlock-free, ordering-robust lock acquisition protocol

```c++
top:
	pthread_mutex_lock(L1);
	if (pthread_mutex_trylock(L2) != 0) {
    pthread_mutex_unlock(L1);
    goto top;
  }
```
* 可能有livelock，但太巧了，可以用随机性处理
* 存在encapsulation的问题，但这个解法至少对某些场合有效

Mutual Exclusion

* lock-free系列方法，见第28节或笔记文件夹内threads-bugs.cpp文件

* 感觉这类方法适用于分布式系统，backup、试错、no bounded loop



**Deadlock Avoidance via Scheduling**

* 全局信息=>更优决策
* Dijkstra’s Banker’s Algorithm [D64]
* 应用不多：场景有限，比如嵌入式系统；限制了并行性

* 设计理念： “Not everything worth doing is worth doing well”

  

**Detect and Recover**

A deadlock detector runs periodically, building a resource graph and checking it for cycles.  常应用于数据库

**HW**

`./vector-*** -n 8 -d -l 100000 -t`

`vector-try-wait`比`vector-global-order`速度略快

`vector-avoid-hold-and-wait`用global锁住local锁的获取，单线程有优势，但不支持并行操作

`vector-nolock`利用fetch-and-add，效率较低



#### 33.Event-based Concurrency (Advanced)

##### CRUX: how to build concurrent servers without threads?

**Event-based Concurrency** 
* event handler独占时间，explicit control over scheduling
* event-based servers中一定不要block！

```c++
while (1) {
	events = getEvents();
	for (e in events)
		processEvent(e);
}
```

**an important API: select() (or poll())**

```c++
int select(int nfds,fd_set *restrict readfds,fd_set *restrict writefds,fd_set *restrict errorfds,struct timeval *restrict timeout);
```
* 这个api的意义是monitor各种fd是否“可用”（比如有新信息可读、有新空间可写）
* timeout参数使用灵活，NULL表示允许无限block，0表示立刻返回，类似于`waitpid`的参数`WNOHANG`
* `pselect`针对pthread做sigmask处理

**A Problem: Blocking System Calls**

=> no blocking calls are allowed => AIO

**A Solution: Asynchronous I/O**

思考：对于一个特殊的问题场景，需要厘清可用操作的边界，必要时可能引入新的概念，例如这里的asynchronous I/O、CSAPP p801的async-signal-safe functions

AIO control block: 利用`aio_error`配合signal机制interrupt，这一思想也用于I/O devices中

* "Flash"这篇论文用hybrid思想，events are used to process network packets, and a thread pool is used to manage outstanding I/Os

**Another Problem: State Management**

manual stack management => use an old programming language construct known as a **continuation**, 用hash table这种数据结构存continuation信息

**What Is Still Difficult With Events**

1. 不适用于多CPU
2. 和systems activity配合不行，比如paging，是implicit blocking
3. 不容易manage over time，改api的routine
4. 这个系统的实现并不容易，hybrid















### Appendix

#### Linux 系统文件

* `/etc/fstab` file is a system configuration *file* that contains all available disks, disk partitions and their options

* `cat /proc/interrupts | grep "TLB shootdowns"`
* /proc/meminfo 可用内存
* /proc/swaps 可swap内存

* /proc/pid/fd/ 查找持有的fd，查文件泄漏
  * `/proc/$tid` 或 `/prod/$pid/task/$tid` 内核任务调度id
  * /proc/pid/status 线程数目
* /proc/[pid]/uid_map 和  /proc/[pid]/gid_map
* /proc/self/maps
* /proc/sys/kernel/ns_last_pid
* /proc/version 查看系统版本





TODO：
* 19.physically-indexed cache
* [Linux堆内存管理深入分析]([https://introspelliam.github.io/2017/09/10/Linux%E5%A0%86%E5%86%85%E5%AD%98%E7%AE%A1%E7%90%86%E6%B7%B1%E5%85%A5%E5%88%86%E6%9E%90%EF%BC%88%E4%B8%8A%EF%BC%89/](https://introspelliam.github.io/2017/09/10/Linux堆内存管理深入分析（上）/))

