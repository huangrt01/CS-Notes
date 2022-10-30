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