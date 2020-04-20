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

### Concurrency

#### 25.A Dialogue on Concurrency

#### 26.Concurrency: An Introduction

概念：thread, multi-threaded, thread control blocks (TCBs)
  * thread-local: 栈不共用，在进程的栈区域开辟多块栈，不是递归的话影响不大
  * thread的意义：1) parallelism, 2) 适应于I/O阻塞系统、缺页中断（需要KLT），这一点类似于multiprogramming的思想，在server-based applications中应用广泛。

<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/master/Data/OSTEP-Operating-Systems-Three-Easy-Pieces/015.jpg" alt="015" style="zoom:50%;" />

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










