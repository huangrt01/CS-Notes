[toc]
## OSTEP-Operating-Systems-Three-Easy-Pieces

__主题：virtualization, concurrency, persistence__

[book](http://pages.cs.wisc.edu/~remzi/OSTEP/), [book-code](https://github.com/remzi-arpacidusseau/ostep-code), [projects](https://github.com/remzi-arpacidusseau/ostep-projects), [homework answer](https://github.com/xxyzz/ostep-hw)

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

<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/master/Notes/OSTEP-Operating-Systems-Three-Easy-Pieces/001.jpg" alt="进程状态转移" style="zoom:50%;" />
* final态（在UNIX称作zombie state）等待子进程return 0，parent进程 wait()子进程

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
* RTFM：read the fucking manual* 

##### HW
* 5.3 [用vfork()保证父进程后执行](https://www.cnblogs.com/zhangxuan/p/6387422.html)

fork()和vfork()的区别：
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

<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/master/Notes/OSTEP-Operating-Systems-Three-Easy-Pieces/002.jpg" alt="LDE protocal" style="zoom:50%;" />

[stub code](https://www.zhihu.com/question/24844900/answer/35126766)

##### CRUX: how to regain control of the CPU
* problem #2:switching between processes
  * A cooperative approach: wait for system calls
  * [MacOS9 Emulator](http://www.columbia.edu/~em36/macos9osx.html#summary)
  * NOTE: only solution to infinite loops is to reboot the machine，reboot is useful
  * A Non-Cooperative Approach: The OS Takes Control
##### CRUX: how to gain control without cooperation
* a timer interrupt    interrupt handler
  * timer也可以关
* deal with malfeasance: in modern systems, the way the OS tries to handle such malfeasance is to simply terminate the offender.

* scheduler    context switch
<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/master/Notes/OSTEP-Operating-Systems-Three-Easy-Pieces/003.jpg" alt="LDE protocal + timer interrupt" style="zoom:50%;" />

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
    多核时代不宜用x86的RDTSC http://www.360doc.com/content/12/0827/17/7851074_232649576.shtml
    system call需要0.3 microseconds; context switch 0.6 microseconds; 单次记录用时1 microseconds
    MacOS上没有sched.h    https://yyshen.github.io/2015/01/18/binding_threads_to_cores_osx.html

#### 7.Scheduling: Introduction

##### CRUX: how to develop scheduling policy

* workload assumptions
  * fully-operational scheduling discipline
  * 概念：jobs
* scheduling metrics：turnaround time

FIFO: convoy effect  <img src="https://www.zhihu.com/equation?tex=%5Cstackrel%7B%5Cbf%7B%E5%85%81%E8%AE%B8%E9%95%BF%E5%BA%A6%E4%B8%8D%E7%AD%89%7D%7D%7B%5Clongrightarrow%7D" alt="\stackrel{\bf{允许长度不等}}{\longrightarrow}" class="ee_img tr_noresize" eeimg="1">  SJF(shortest job first)  <img src="https://www.zhihu.com/equation?tex=%5Cstackrel%7B%5Cbf%7B%E5%85%81%E8%AE%B8%E6%9D%A5%E6%97%B6%E4%B8%8D%E7%AD%89%7D%7D%7B%5Clongrightarrow%7D%20" alt="\stackrel{\bf{允许来时不等}}{\longrightarrow} " class="ee_img tr_noresize" eeimg="1">  STCF(Shortest Time-to-Completion First )=PSJF      <img src="https://www.zhihu.com/equation?tex=%5Cstackrel%7B%5Cbf%7B%E5%85%81%E8%AE%B8%E4%B8%8Drun%5C%20to%5C%20completion%7D%7D%7B%5Clongrightarrow%7D" alt="\stackrel{\bf{允许不run\ to\ completion}}{\longrightarrow}" class="ee_img tr_noresize" eeimg="1">  a new metric: response time

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
  * Solaris：Default values for the table are 60 queues, with slowly increasing time-slice lengths from 20 milliseconds (highest priority) to a few hundred milliseconds (lowest),and priorities boosted around every 1 second or so，和思考一致：高优先级，把time slice调短

attempt3: better accounting

* Rule 4:Once a job uses up its time allotment at a given level (regardless of how many times it has given up the CPU), its priority is reduced (i.e., it moves down one queue).

其它可能的特性：
* 操作系统0优先级
* advice机制：As the operating system rarely knows what is best for each and every process of the system, it is often useful to provide interfaces to allow usersor administrators to provide some hints to the OS. We often call such hints advice, as the OS need not necessarily pay attention to it, but rather might take the advice into account in order to make a better decision. Such hints are useful in many parts of the OS, including the scheduler(e.g., with **nice**), memory manager (e.g.,**madvise**), and file system (e.g.,informed prefetching and caching [P+95])


* HW: iobump，io结束后把进程调到当前队列第一位，否则最后一位；io越多效果越好

#### 9.Scheduling: Proportional share

##### CRUX: how to share the CPU proportionally

Basic Concept: Tickets Represent Your Share
* 利用randomness：
  1. 避免corner case，LRU replacement policy (cyclic sequential)    
  2. lightweight    
  3. fast，越快越伪随机

NOTE：

如果对伪随机数限定范围，不要用rand，[用interval](https://stackoverflow.com/questions/2509679/how-to-generate-a-random-integer-number-from-within-a-range)

 机制：
1. ticket currency，用户之间
2. ticket transfer，用户与服务器
3. ticket inflation，临时增加tickets，需要进程之间的信任

* unfairness metric
* stride scheduling    — deterministic
  * lottery scheduling相对于stride的优势：no global state

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

#### 10.Multiprocessor Scheduling
* 概念：multicore processor        threads
* 还没看

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

location of code : 0x105f40ec0
location of heap : 0x105f55000
location of stack: 0x7ffee9cbf8ac        [64bit系统下进程的内存分布](https://blog.csdn.net/chenyijun/article/details/79441166)

NOTE:
* 用microkernels的思想实现isolation，机制和策略的分离

### Concurrency

#### 25.A Dialogue on Concurrency

#### 26.Concurrency: An Introduction

概念：thread, multi-threaded, thread control blocks (TCBs)
  * thread-local: 栈不共用，在进程的栈区域开辟多块栈，不是递归的话影响不大
  * thread的意义：1) parallelism, 2) 适应于I/O阻塞系统、缺页中断（需要KLT），这一点类似于multiprogramming的思想，在server-based applications中应用广泛。

<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/master/Notes/OSTEP-Operating-Systems-Three-Easy-Pieces/015.jpg" alt="015" style="zoom:50%;" />

NOTE：
* pthread_join与[detach](https://blog.csdn.net/heybeaman/article/details/90896663)
* disassembler: objdump -d -g main
* x86，变长指令，1-11个字节

问题：线程之间会出现data race，e.g. counter的例子

引入概念：critical section, race condition, indeterminate, mutual exclusion 

=> the wish for atomicity

* transaction: the grouping of many actions into a single atomic action 
* 和数据库, journaling、copy-on-write联系紧密
* 条件变量：用来等待而非上锁

##### CRUX: how to support synchronization
* the OS was the first concurrent program!
* Not surprisingly, pagetables, process lists, file system structures, and virtuallyevery kernel datastructure has to be carefully accessed, with the proper synchronizationprimitives, to work correctly.

HW26:
* data race来源于线程保存的寄存器和stack，
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

* [关于条件变量需要互斥量保护的问题](https://www.zhihu.com/question/53631897)
* pthread_cond_wait内部先解锁再等待，之所以加锁是防止cond_wait内部解锁后时间片用完。https://blog.csdn.net/zrf2112/article/details/52287915

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










