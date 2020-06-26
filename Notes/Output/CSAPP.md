## CSAPP- Computer Systems: A Programmer’s Perspective, Third Edition

#### Preface

#### 1.A Tour of Computer Systems

### Part I Program Structure and Execution


### Part II Running Programs on a System

#### 8.Exceptional Control Flow

##### 8.1 Exceptions

ECF：e.g. hardware timer, packet arrived, request data from disk and sleep until the data are ready, 子进程terminate

*  nonlocal jumps (provided in C via the `setjump` and `longjump`)
* 概念event：change in state
* handler的返回方式：1.返回到 <img src="https://www.zhihu.com/equation?tex=I_%7Bcurr%7D" alt="I_{curr}" class="ee_img tr_noresize" eeimg="1">     2.返回到 <img src="https://www.zhihu.com/equation?tex=I_%7Bnext%7D" alt="I_{next}" class="ee_img tr_noresize" eeimg="1"> 	3.abort
  * interrupt - 2.   
  * trap - 2.
  * fault - 1. or 3.
  * abort - 3.
  
* 思考：返回到 <img src="https://www.zhihu.com/equation?tex=I_%7Bnext%7D" alt="I_{next}" class="ee_img tr_noresize" eeimg="1"> 的情形值得注意，例如一个空的handler可以中断sleep。是否可能会丢失一些需要curr指令处理的状态，然后出现不可预知的问题？
* exception number, exception table base register

exception和procedure call的区别

* The processor also pushes some additional processor state onto the stack that will be necessary to restart the interrupted program when the handler returns. For example, an x86-64 system pushes the EFLAGS register containing the current condition codes, among other things, onto the stack.
* When control is being transferred from a user program to the kernel, all of these items are pushed onto the kernel’s stack rather than onto the user’s stack.    e.g.  timer_interrupt存user register到kernel stack
* Exception handlers run in *kernel mode* (Section 8.2.4), which means they have complete access to all system resources.

exceptions的类别：interrupts, traps, faults, aborts，分为同步和异步，只有interrupts是异步的

* trap是syscall
* faults例如page fault handler

汇编：All arguments to Linux system calls are passed through general-purpose registers rather than the stack. By convention, register %rax contains the syscall number, with up to six arguments in %rdi, %rsi, %rdx, %r10, %r8, and %r9. The first argument is in %rdi, the second in %rsi

##### 8.2 Processes

特点：
* An independent logical control flow that provides the illusion that our pro- gram has exclusive use of the processor.
* A private address space that provides the illusion that our program has exclu- sive use of the memory system.

概念：
* parallel flow: 运行在不同的cores或computers上

Linux x86-64进程地址空间



<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/master/Notes/CSAPP/process.jpg" alt="process address space" style="zoom:50%;" />

mode bit (in control register) is set => 内核态

Linux能获取内核信息的文件系统

* /proc	/proc/cpuinfo		/proc/process-id/maps
* /sys: additional low-level information about system buses and devices



**context switches**

进程上下文：

* kernel's data structure: page/process/file table
* values of objects: the general-purpose registers, the floating-point registers, the program counter, user’s stack, status registers, kernel’s stack



##### 8.3 System Call Error Handling

处理error的，`<errno.h>`

* error-reporting function
* error-handling wrappers

##### 8.4 Process Control

设置exit status的方式：1）输入exit参数	2）main function的return value

fork的其中一个特性是share files，比如共享stdout file，print到同一个地方

画process graph辅助理解nested fork的行为

8.4.3: reap child processes, p779, `waitpid`的详细介绍

如果parent不reap，由kernel创建的pid为1的`init`进程reap zombie children



**execve**(p786)

unlike fork, which is called once but returns twice, execve is called once and never returns.

load file-name之后，call the start-up code

p788: user-stack结构图		%rdi %rsi %rdx: argc argv envp

env相关的函数

* `char *getenv(const char *name); `
* `int setenv(const char *name, const char *newvalue, int overwrite);`
* `void unsetenv(const char *name);`

##### 8.5 Signals

p793: Linux Signals		`man 7 signal`

signal体现了kernel exception handlers和user space的交互

pending signal，后续的相同signal会被discard，process可以block signal
* linux中，`pending`和`block` vector

process groups: `getpgrp(), setgpid(pid,pgid)`

kill和alarm函数 p798

install the handler -> catch the signal -> handle the signal

e.g. 一个空return handler可以让sleep进程return


**实现handler的关键细节**

p801: block机制，async-signal-safe functions，特征：reentrant，或者不会被signal handler interrupt

Declare global variables with volatile: 这样不会被编译器优化，确保handler更新的global值能被main知道

p811: Signal wrapper

p817: 用sigsuspend处理spin loop问题

##### 8.6 Nonlocal Jumps

user-level层面的ECF，`setjmp`类比catch，sigsetjmp类比throw

```c++
#include <setjmp.h>
int setjmp(jmp_buf env);
int sigsetjmp(sigjmp_buf env, int savesigs);
//Returns: 0 from setjmp, nonzero from longjmps
```
















