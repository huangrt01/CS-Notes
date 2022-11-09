**Advanced Programming in the UNIX Envinronment, 2013**

[toc]

* [signal(SIGPIPE, SIG_IGN)](https://blog.csdn.net/weiwangchao_/article/details/38901857)

  * 在linux下写socket的程序的时候，如果尝试send到一个disconnected socket上，就会让底层抛出一个SIGPIPE信号。这个信号的缺省处理方法是退出进程，大多数时候这都不是我们期望的。因此我们需要重载这个信号的处理方法。

  * 进阶的处理：

    * ```c++
      struct sigaction sa = {};
      sa.sa_handler = [](int) {};
      sa.sa_flags = 0;
      sigemptyset(&sa.sa_mask);
      sigaction(SIGPIPE, &sa, nullptr);
      ```

    * 不用SIG_IGN: swallow the signal to enable SIGPIPE in children to behave normally.

    * `sa.sa_flags = 0` prevents SA_RESTART from restarting syscalls after the handler completed. This is important for code using SIGPIPE to interrupt syscalls in other threads.


### chpt 14 Advanced I/O

* nonblocking I/O, record locking, I/O multiplexing (the select and poll functions), asynchronous I/O, the readv and writev functions, and memory-mapped I/O (mmap)

* 14.2 Nonblocking I/O
  * slow system calls 的概念
  * O_NONBLOCK flag
  * 失败返回-1，the errno of 35 is EAGAIN
  * 14.4 I/O multiplexing with a nonblocking descriptor is a more efficient way
  * Chpt11/21 用多线程绕过non-blocking I/O的问题

* 14.3 Record Locking

  * Record locking is the term normally used to describe the ability of a process to prevent other processes from modifying a region of a file while the first process is reading or modifying that portion of the file
    * Better Name: byte-range locking
  * fcntl Record Locking
    * 分成读写锁两类
    * `F_GETLK, F_SETLK, F_SETLKW`
  * Implied Inheritance and Release of Locks
    * Locks are associated with a process and a file.
      * whenever a descriptor is closed, any locks on the file referenced by that descriptor for that process are released
    * Locks are never inherited by the child across a fork.
    * Locks are inherited by a new program across an exec.
    * 数据结构分析：每个v_node维护一个lockf_entry list，而fd映射到v_node是多对一
  * Locks at End of File
    * 处理并行写的问题
  * Advisory versus Mandatory Locking
    * Mandatory locking causes the kernel to check every open, read, and write to verify that the calling process isn’t violating a lock on the file being accessed. Mandatory locking is sometimes called *enforcement-mode locking*.
    * Mandatory locking is enabled for a particular file by turning on the set-group-ID bit and turning off the group-execute bit.
  * Example: 使用vi时fork进程设置mandatory locking，但这样不能work，因为编辑器通常打开文件后就会close fd

* 14.4 I/O Multiplexing

  * 处理多路io的常见思路：

    * polling
    * async I/O
      * limited forms
      * use only one signal per process (SIGPOLL or SIGIO)
    * I/O multiplexing

  * I/O multiplexing

    * select and pselect Functions

      * ```c
        #include <sys/select.h>
        int select(int maxfdp1, fd_set *restrict readfds, fd_set *restrict writefds, fd_set *restrict exceptfds, struct timeval *restrict tvptr);
        ```

      * `FD_SET`

      * `maxfdp1`

    * poll function

      * On return, the revents member is set by the kernel, thereby specifying which events have occurred for each descriptor.

* 14.5 Asynchronous I/O

  * We incur additional complexity when we use the POSIX asynchronous I/O interfaces:

    * three sources of errors for every asynchronous operation
    * The interfaces themselves involve a lot of extra setup and processing rules compared to their conventional counterparts
    * Recovering from errors can be difficult

  * System V Asynchronous I/O

  * BSD Asynchronous I/O

  * POSIX Asynchronous I/O

    * ```c++
      struct aiocb {
        int             aio_fildes;
        off_t           aio_offset;
        volatile void  *aio_buf;
        size_t          aio_nbytes;
        int             aio_reqprio;
        struct sigevent aio_sigevent;
        int             aio_lio_opcode;  /* operation for list I/O */
      };
      
      struct sigevent;
      
      #include <aio.h>
      int aio_read(struct aiocb *aiocb);
      int aio_write(struct aiocb *aiocb);
      int aio_fsync(int op, struct aiocb *aiocb);
      ssize_t aio_return(const struct aiocb *aiocb);
      int aio_suspend(const struct aiocb *const list[], int nent, const struct timespec *timeout);
      int aio_cancel(int fd, struct aiocb *aiocb);
      
      int lio_listio(int mode, struct aiocb *restrict const list[restrict], int nent, struct sigevent *restrict sigev);
      ```

    * e.g. 书 P517

      * we use eight buffers, so we can have up to eight asynchronous I/O requests pending. Surprisingly, this might actually reduce performance — if the reads are presented to the file system out of order, it can defeat the operating system’s read-ahead algorithm.
      * When all AIO control blocks are in use, we wait for an operation to complete by calling aio_suspend.
      * 这个例子有特殊性，由于read和write的字节数相等，每个aio操作的offset是互不影响的，能简化程序

  