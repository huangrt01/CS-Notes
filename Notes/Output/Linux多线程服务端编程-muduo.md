[toc]

muduo 是一个基于非阻塞 IO 和事件驱动的现代 C++ 网络库，原生支持 one loop per thread 这种 IO 模型。muduo 适合开发 Linux 下的面向业务的多线程服务端网络应用程序

### Part I: C++ 多线程系统编程

#### chpt 1 线程安全的对象生命期管理

* 基本问题是处理析构，用智能指针解决
* 依据 [JCP]，一个线程安全的 class 应当满足以下三个条件:
  * 多个线程同时访问时，其表现出正确的行为。
  * 无论操作系统如何调度这些线程，无论这些线程的执行顺序如何交织(interleaving)。
  * 调用端代码无须额外的同步或其他协调动作。
* 对象创建：不要在构造函数泄漏this指针，最后一行也不行（因为接着可能执行派生类的代码）
* 析构函数
  * 作为数据成员的 mutex 不能保护析构
  * 同时读写一个 class 的两个对象，有潜在的死锁可能
    * 为了保证始终按相同的顺序加锁，我 们可以比较 mutex 对象的地址，始终先加锁地址较小的 mutex
* 线程安全的 Observer 有多难
  * 对象的关系主要有三种:composition、aggregation、 association
  * 如果对象 x 注册了任何非静态成员函数回调，那么必然在某处持有了指向 x 的指针，这就暴露在了 race condition 之下
* 原始指针有何不妥
  * 垃圾回收的原理，所有人都用不到的东西一定是垃圾
  * 解决思路：引入另外一层间接性(another layer of indirection)
* 关于智能指针
  * shared_ptr/weak_ptr 的线程安全级别与 std::string 和 STL 容器一样（$1.9）
    * 并发读写shared_ptr（它本身，不是指它指向的对象），要加锁
    * 销毁行为移出临界区：利用local_ptr和global_ptr做swap()
  * 原子操作，性能不错
* 各种内存错误
  * 缓冲区溢出(buffer overrun)。
  * 空悬指针/野指针。
  * 重复释放(double delete)。
  * 内存泄漏(memory leak)。
  * 不配对的 new[]/delete。
  * 内存碎片(memory fragmentation)  --> §9.2.1 和 §A.1.8 
* Observer_safe
  * 锁争用 和 死锁，update()函数有隐含要求（比如不能unregister）
  * 倾向于使用不可重入的 mutex，例如 Pthreads 默认提供的那个，因为 “要求 mutex 可重入”本身往往意味着设计上出了问题(§2.1.1)。Java 的 intrinsic lock 是可重入的，因为要允许 synchronized 方法相互调用(派生类调用基类的同名 synchronized 方法)，我觉得这也是无奈之举
* shared_ptr 技术与陷阱
  * 意外延长对象的生命期：意外、lambda函数捕获
  * 函数参数：最外层持有一个实体时，可以传常引用
  * 析构动作在创建时被捕获
    * 二进制兼容性
    * 讲解deleter传参的实现，模版相关：https://www.artima.com/articles/my-most-important-c-aha-momentsemeverem，泛型编程和面向对象编程的一次完美结合
  * 析构所在的线程：我们可以用一个单 独的线程来专门做析构，通过一个 `BlockingQueue<shared_ptr<void> >` 把对象的析 构都转移到那个专用线程，从而解放关键线程
  * 现成的 RAII handle
    * 注意避免循环引用，通常的做法是 owner 持有指向 child 的 shared_ptr，child 持有指向 owner 的 weak_ptr
* 对象池
  * 释放对象：weak_ptr
  * 解决内存泄漏：shared_ptr初始化传入delete函数
  * this指针线程安全问题：enable_shared_from_this
  * Factory生命周期延长：弱回调
    * 通用的弱回调封装见 recipes/thread/WeakCallback.h，用到了 C++11 的 variadic template 和 rvalue reference
    * 在事件通知中非常有用
* 替代方案
  * 全局facade，对象访问都加锁，代价是性能，可以像Java的ConcurrentHashMap分buckets降低锁的代价
* Observer 之谬
  * recipes/thread/SignalSlot.h

##### Note

* 临界区在 Windows 上是 struct CRITICAL_SECTION，是可重入的;在 Linux 下是 pthread_mutex_t，默认是不可重入的

* 在 Java 中，一个 reference 只要不为 null，它一定指向有效的对象

* 如果这几种智能指针是对象 x 的数据成员，而它的模板参数 T 是个 incomplete 类型，那么 x 的析构函数不能是默认的或内联的，必须在 .cpp 文件里边 显式定义，否则会有编译错或运行错(原因见 §10.3.2)
  * 智能指针参考

* [function/bind的救赎（上） by 孟岩](https://blog.csdn.net/myan/article/details/5928531)

* * 过程范式、函数范式、对象范式
  * 对象范式的两个基本观念：
    - 程序是由**对象**组成的；
    - 对象之间互相**发送消息**，协作完成任务；
    - 请注意，这两个观念与后来我们熟知的面向对象三要素“封装、继承、多态”根本不在一个层面上，倒是与再后来的“组件、接口”神合
  * C++的静态消息机制：不易开发windows这种动态消息场景；“面向类的设计”过于抽象
  * Java和.NET中分别对C++最大的问题——缺少对象级别的delegate机制做出了自己的回应

* [My Most Important C++ Aha! Moments... Ever Opinion by Scott Meyers](https://www.artima.com/articles/my-most-important-c-aha-momentsemeverem)

  * Realizing that C++’s “special” member functions may be declared private, 1988
  * Understanding the use of non-type template parameters in Barton’s and Nackman’s approach to dimensional analysis, 1995
    * https://learningcppisfun.blogspot.com/2007/01/units-and-dimensions-with-c-templates.html
  * Understanding what problem Visitor addresses, 1996 or 1997
  * Understanding why remove doesn’t really remove anything, 1998
    * remove不改变容器的元素个数，本质上是因为在STL中，container 和 algorithm (基于iterator) 在设计上分离的概念，比如当 algorithm 遇到了 array 类型，"couldn't change the size"
  * Understanding how deleters work in Boost’s shared_ptr, 2004.



#### chpt 2 线程同步精要

* 并发编程有两种基本模型，一种是 message passing，另一种是 shared memory

* 线程同步的四项原则，按重要性排列
  * 首要原则是尽量最低限度地共享对象，减少需要同步的场合。一个对象能不暴 露给别的线程就不要暴露;如果要暴露，优先考虑 immutable 对象;实在不行 才暴露可修改的对象，并用同步措施来充分保护它。
  * 其次是使用高级的并发编程构件，如 TaskQueue、Producer-Consumer Queue、 CountDownLatch 等等。
  * 最后不得已必须使用底层同步原语(primitives)时，只用非递归的互斥器和条件变量，慎用读写锁，不要用信号量。
  * 除了使用 atomic 整数之外，不自己编写 lock-free 代码，也不要用“内核级” 同步原语 。不凭空猜测“哪种做法性能会更好”，比如 spin lock vs. mutex。

* 互斥器 mutex
  
  * Mutex
  
    * 核心原则：利用RAII保证各项细则
    * Scoped Locking：不手工调用 lock() 和 unlock() 函数，一切交给栈上的 Guard 对象的构造和析构函数负责
    * 在每次构造 Guard 对象的时候，思考一路上(调用栈上)已经持有的锁，防止因加锁顺序不同而导致死锁(deadlock)
    * 必要的时候可以考虑用 PTHREAD_MUTEX_ERRORCHECK 来排错
  
  * 只使用非递归的 mutex
  
    * 概念：recursive or reentrant 
  
    * recursive mutex 会隐藏代码问题：recipes/thread/test/NonRecursiveMutex_test.cc
  
      * post-mortem
  
    * ```c++
      void postWithLockHold(const Foo& f) {
      	assert(mutex.isLockedByThisThread()); // muduo::MutexLock 提供了这个成员函数
      	// ... 
      }
      ```
  
    * 性能: 
  
      * Linux 的 Pthreads mutex 采用 futex(2) 实现，不必每次加锁、解锁都陷入系统调用，效率不错
        * [futex](https://akkadia.org/drepper/futex.pdf) TODO
  
      * Windows 的 [CRITICAL_SECTION](http://msdn.microsoft.com/en-us/library/windows/desktop/ms682530(v=vs.85).aspx) 也是类似的，不过它可以嵌入一小段 spin lock。在多 CPU 系统上，如果不能立刻拿到锁，它会先 spin 一小段时间，如果还不能拿到锁，才挂起当前线程
  
  * 死锁
  
    * `recipes/thread/test/SelfDeadLock.cc`
    * `recipes/thread/test/MutualDeadLock.cc`
  
  * false sharing and CPU cache TODO
  
    * http://www.aristeia.com/TalkNotes/ACCU2011_CPUCaches.pdf
    * http://www.akkadia.org/drepper/cpumemory.pdf
    * http://igoro.com/archive/gallery-of-processor-cache-effects/
    * http://simplygenius.net/Article/FalseSharing
  
* 条件变量

  * Java Object 内置的 wait()、notify()、 notifyAll() 是条件变量，以容易用错著称，一般建议用 java.util.concurrent 中的同步原语

  * ```c++
    // 经典应用 BlockingQueue
    muduo::MutexLock  mutex;
    muduo::Condition  cond(mutex);
    std::deque<int>   queue;
    int dequeue()
    {
      MutexLockGuard lock(mutex);
      while (queue.empty()) // 必须用循环;必须在判断之后再 wait()
      {
        cond.wait(); // 这一步会原子地 unlock mutex 并进入等待，不会与 enqueue 死锁
      // wait() 执行完毕时会自动重新加锁
      }
      assert(!queue.empty());
      int top = queue.front();
      queue.pop_front();
      return top;
    }
    
    void enqueue(int x)
    {
    	MutexLockGuard lock(mutex);
      queue.push_back(x);
    	cond.notify(); // 可以移出临界区之外
    }
    ```

  * broadcast should generally be used to indicate state change rather than resource availability

  * Mutex 和 Condition，就像与非门和 D 触发器构成了数字电路设计所需的全部基础元件，可以完成任何组合和同步时序逻辑电路设计一样

  * [spurious wakeup](https://en.wikipedia.org/wiki/Spurious_wakeup): spurious wakeups can happen whenever there's a race and possibly even in the absence of a race or a signal

    * 最简单的场景应该是有人给你的程序发了个Signal，当前正在执行的各种各样的系统调用都可能被中断。

    * `cv.wait(lk, []{ return whether the event has occurred; });` 预防spurious wakeup

      * ```
        while (!pred()) {
            wait(lock);
        }
        ```

  * notify_one() 的细节，[讨论要不要放在锁里](https://en.cppreference.com/w/cpp/thread/condition_variable/notify_one)

    * "hurry up and wait" scenario
    * when precise scheduling of events is required

* 不要用读写锁和信号量

  * 读锁并不比 Mutex 高效
  * reader lock 可能允许提升(upgrade)为 writer lock，也可能不允许提升
  * 通常 reader lock 是可重入的，writer lock 是不可重入的。但是为了防止 writer 饥饿，writer lock 通常会阻塞后来的 reader lock，因此 **reader lock 在重入的时候可能死锁**。另外，在追求低延迟读取的场合也不适用读写锁，见 p. 55。
    * 注：java的ReentrantReadWriteLock实现，允许非公平锁，[读锁重入不会死锁](https://heapdump.cn/article/3957407)，而Go会死锁
  * 性能问题怎么办？
    * 2.8 copy-on-write
    * read-copy-update
  * 信号量：哲学家就餐问题，“平权不如集权”
    * 如果要控制并发度，可以考虑用 muduo::ThreadPool
  * barrier原语：不如 CountDownLatch

* 封装 MutexLock、MutexLockGuard、Condition

* 线程安全的 Singleton 实现

  * C++：pthread_once、DCL（见本节Note）、Memory Barrier、Eager Initialization

* sleep(3) 不是同步原语

  * 生产代码中线程的等待可分为两种:一种是等待资源可用(要么等在 select/ poll/epoll_wait 上，要么等在条件变量上;一种是等着进入临界区(等在 mutex 上)以便读写共享数据。后一种等待通常极短，否则程序性能和伸缩性就会有问题
  * 在用户态做轮询(polling)是低效的

* 总结

  * 本文没有考虑 signal 对多线程编程的影响(§4.10)，Unix 的 signal 在多线程下的行为比较复杂，一般要靠底层的网络库(如 Reactor)加以屏蔽，避免干扰上层应用程序的开发

* 借 shared_ptr 实现 copy-on-write

  * 用普通 mutex 替换读写锁的一个例子



##### Note

* [Real-world Concurrency](https://queue.acm.org/detail.cfm?id=1454462) 

  * history context
    * it was the introduction of the Burroughs B5000 in 1961 that proffered the idea that ultimately proved to be the way forward: disjoint CPUs concurrently executing different instruction streams but sharing a common memory
    * 1980: cache coherence protocols、prototyped parallel operating systems、parallel databases
    * 1990: symmetric multiprocessing，硬件+软件，uniprocessors to multiprocessors
      * microprocessor architects incorporated deeper (and more complicated) pipelines, caches, and prediction units
      * Many saw these two trends—the rise of concurrency and the futility of increasing clock rate—and came to the logical conclusion: instead of spending transistor budget on “faster” CPUs that weren’t actually yielding much in terms of performance gains (and had terrible costs in terms of power, heat, and area), why not take advantage of the rise of concurrent software and use transistors to effect multiple (simpler) cores per die?
      * That it was the success of concurrent software that contributed to the genesis of chip multiprocessing is an incredibly important historical point and bears reemphasis.
  * Concurrency is for Performance
    *  Just as no programmer felt a moral obligation to eliminate pipeline stalls on a superscalar microprocessor, no software engineer should feel responsible for using concurrency simply because the hardware supports it.
      * problems不一定应该/值得并行
    * to hide latency, for example, a disk I/O operation or a DNS lookup
      * 不一定值得
      * one can often achieve the same effect by employing nonblocking operations (e.g., asynchronous I/O) and an event loop (e.g., the poll()/select() calls found in Unix) in an otherwise sequential program.
    * to increase throughput need *not* consist exclusively (or even largely) of multithreaded code
      * e.g. typical MVC (model-view-controller) application
        * [为什么我不再推荐使用MVC框架？](https://www.toutiao.com/article/6763420080542843399/)
      * it is concurrency by architecture instead of by implementation.
  * Illuminating the Black Art
    * oral tradition in lieu of formal writing has left the domain shrouded in mystery
    * Know your cold paths from your hot paths.
    * Intuition is frequently wrong—be data intensive  ~ 压测
      * timeliness is more important than absolute accuracy: the absence of a perfect load simulation should not prevent you from simulating load altogether
      * Understanding scalability inhibitors on a production system requires the ability to safely dynamically instrument its synchronization primitives.
      * breaking up a lock is not the only way to reduce contention, and contention can be (and often is) more easily reduced by decreasing the hold time of the lock. This can be done by algorithmic improvements (many scalability improvements have been achieved by reducing execution under the lock from quadratic time to linear time!) or by finding activity that is needlessly protected by the lock
        * 经典例子：deallocate放到锁外
      * Be wary of readers/writer locks.
        * a readers/writer lock will use a single word of memory to store the number of readers
      * Consider per-CPU locking
        * if one were implementing a global counter that is frequently updated but infrequently read, one could implement a per-CPU counter protected by its own lock. Updates to the counter would update only the per-CPU copy, and in the uncommon case in which one wanted to read the counter, all per-CPU locks could be acquired and their corresponding values summed.
        * be sure to have a single order for acquiring all locks in the cold path
      * Know when to broadcast—and when to signal
        * Broadcast ~ *state change*, Signal ~ *resource availability*
        * *thundering herd*
      * Learn to debug postmortem
      * Second (and perhaps counterintuitively), one can achieve concurrency and composability by having no locks whatsoever. In this case, there must be no global subsystem state—subsystem state must be captured in per-instance state, and it must be up to consumers of the subsystem to assure that they do not access their instance in parallel. By leaving locking up to the client of the subsystem, the subsystem itself can be used concurrently by different subsystems and in different contexts
        * A concrete example of this is the AVL tree implementation used extensively in the Solaris kernel. As with any balanced binary tree, the implementation is sufficiently complex to merit componentization, but by not having any global state, the implementation may be used concurrently by disjoint subsystems—the only constraint is that manipulation of a single AVL tree instance must be serialized.
      * Don’t use a semaphore where a mutex would suffice.
        * unlike a semaphore, a mutex has a notion of *ownership*
        * First, there is no way of propagating the blocking thread’s scheduling priority to the thread that is in the critical section. This ability to propagate scheduling priority—*priority inheritance*—is critical in a realtime system, and in the absence of other protocols, semaphore-based systems will always be vulnerable to priority inversions.
      * Consider memory retiring to implement per-chain hash-table locks.
        * 见 【code-reading笔记】illumos部分
      * Be aware of false sharing.
        * This most frequently arises in practice when one attempts to defract contention with **an array of locks**
        * it can be expected to be even less of an issue on a multicore system (where caches are more likely to be shared among CPUs)
        * In this situation, array elements should be padded out to be a multiple of the coherence granularity.
      * Consider using nonblocking synchronization routines to monitor contention.
        * 见【code-reading笔记】illumos部分的per-cpu cache
      * When reacquiring locks, consider using generation counts to detect state change.
      * Use wait- and lock-free structures only if you absolutely must.
      * Prepare for the thrill of victory—and the agony of defeat.
  * The Concurrency Buffet
    * Those practitioners who are implementing a database or an operating system or a virtual machine will continue to need to sweat the details of writing multithreaded code
  
* [The "Double-Checked Locking is Broken" Declaration](http://www.cs.umd.edu/~pugh/java/memoryModel/DoubleCheckedLocking.html)

  * ```java
    / Broken multithreaded version
    // "Double-Checked Locking" idiom
    class Foo { 
      private Helper helper = null;
      public Helper getHelper() {
        if (helper == null) 
          synchronized(this) {
            if (helper == null) 
              helper = new Helper();
          }    
        return helper;
      }
      // other functions and members...
    }
    ```

  * Unfortunately, that code just does not work in the presence of either optimizing compilers or shared memory multiprocessors. There is *no way* to make it work without requiring each thread that accesses the helper object to perform synchronization.

    * obj产生和构造函数在inline后可能乱序，getHelper()拿到未构造完成的对象

  * A fix that doesn't work

    * The rule for a monitorexit (i.e., releasing synchronization) is that actions before the monitorexit must be performed before the monitor is released. However, there is no rule which says that actions after the monitorexit may not be done before the monitor is released. 

  * 必须加 memory_barrier，但还不够

    * The problem is that on some systems, the thread which sees a non-null value for the `helper` field also needs to perform memory barriers.
    * 本质上是需要 cache coherence instruction

  * Making it work for static singletons

  * It will work for 32-bit primitive values

    * it does not work for long's or double's, since unsynchronized reads/writes of 64-bit primitives are not guaranteed to be atomic.

    * ```c++
      // Lazy initialization 32-bit primitives
      // Thread-safe if computeHashCode is idempotent
      class Foo { 
        private int cachedHashCode = 0;
        public int hashCode() {
          int h = cachedHashCode;
          if (h == 0) {
            h = computeHashCode();
            cachedHashCode = h;
            }
          return h;
          }
        // other functions and members...
      }
      ```

  * ```c++
    // C++ implementation with explicit memory barriers
    // Should work on any platform, including DEC Alphas
    // From "Patterns for Concurrent and Distributed Objects",
    // by Doug Schmidt
    template <class TYPE, class LOCK> TYPE *
    Singleton<TYPE, LOCK>::instance (void) {
        // First check
        TYPE* tmp = instance_;
        // Insert the CPU-specific memory barrier instruction
        // to synchronize the cache lines on multi-processor.
        asm ("memoryBarrier");
        if (tmp == 0) {
            // Ensure serialization (guard constructor acquires lock_).
            Guard<LOCK> guard (lock_);
            // Double check.
            tmp = instance_;
            if (tmp == 0) {
                    tmp = new TYPE;
                    // Insert the CPU-specific memory barrier instruction
                    // to synchronize the cache lines on multi-processor.
                    asm ("memoryBarrier");
                    instance_ = tmp;
            }
        }
        return tmp;
    }
    ```

  * Fixing Double-Checked Locking using Thread Local Storage

  * Under the new Java Memory Model

    * Fixing Double-Checked Locking using Volatile
    * Double-Checked Locking Immutable Objects

* [Locks Aren't Slow; Lock Contention Is](https://preshing.com/20111118/locks-arent-slow-lock-contention-is/)

  * durations的角度
    * For short lock durations, up to say 10%, the system achieved very high parallelism. Not perfect parallelism, but close. Locks are fast!
    * once the lock duration passes 90%, there’s no point using multiple threads anymore
  * lock frequency的角度：
    * As my [next post](http://preshing.com/20111124/always-use-a-lightweight-mutex) shows, a pair of lock/unlock operations on a Windows Critical Section takes about **23.5 ns** on the CPU used in these tests
    * us 级别时，综合性能表现较好
  * the lock around the memory allocator in a game engine will often achieve excellent performance. 
  * 本文没有结合 CPU usage 做分析，最好能定量 CPU 损失以及连带的吞吐影响

* [Please Don’t Rely on Memory Barriers for Synchronization!](http://www.thinkingparallel.com/2007/02/19/please-dont-rely-on-memory-barriers-for-synchronization/) TODO

* [Lock-Free Code: A False Sense of Security](http://www.talisman.org/~erlkonig/misc/herb+lock-free-code/p1-lock-free-code--a-false-sense-of-security.html) TODO

* [Read-Copy-Update](https://en.wikipedia.org/wiki/Read-copy-update) 看了眼好复杂，没空读了



#### chpt 3 多线程服务器的适用场合与常用编程模型

* 进程与线程

  * 《Erlang 程序设计》[ERL] 把“进程” 比喻为“人”，我觉得十分精当，为我们提供了一个思考的框架。
    * 每个人有自己的记忆(memory)，人与人通过谈话(消息传递)来交流，谈话既可以是面谈(同一台服务器)，也可以在电话里谈(不同的服务器，有网络通信)。
    * 面谈和电话谈的区别在于，面谈可以立即知道对方是否死了(crash, SIGCHLD)，而电话谈只能通过周期性的心跳来判断对方是否还活着。
  * 线程的特点是共享地址空间，从而可以高效地共享数据。一台机器上的多个进程 能高效地共享代码段(操作系统可以映射为同样的物理内存)，但不能共享数据。如果多个进程大量共享内存，等于是把多进程程序当成多线程来写，掩耳盗铃。

* 单线程服务器的常用编程模型

  * Reactor模式：“non-blocking IO + IO multiplexing”

    * lighttpd，单线程服务器。(Nginx与之类似，每个工作进程有一个eventloop。) 

    * libevent，libev。ACE，Poco C++ libraries。

    * Java NIO，包括 Apache Mina 和 Netty。POE(Perl)。Twisted (Python)。

    * 优点：不仅可以用于读写 socket， 连接的建立(connect(2)/accept(2))甚至 DNS 解析 4 都可以用非阻塞方式进行，以 提高并发度和吞吐量(throughput)，对于 IO 密集的应用是个不错的选择

    * 缺点：它要求事件回调函数必须是非阻塞 的。对于涉及网络 IO 的请求响应式协议，它容易割裂业务逻辑，使其散布于多个回调函数之中，相对不容易理解和维护

    * ```c++
      while (!done) {
        int timeout_ms = max(1000, getNextTimedCallback());
        int retval = ::poll(fds, nfds, timeout_ms);
        if (retval < 0) {
          处理错误，回调用户的 error handler
        } else {
          处理到期的 timers，回调用户的 timer handler
            if (retval > 0) {
              处理 IO 事件，回调用户的 IO event handler }
        	}
      	}
      }
      ```

  * Proactor模式：

    * Boost.Asio 和 Windows I/O Completion Ports
    * 和Reactor模式的区别在于由内核完成IO操作（同步IO可以模拟异步IO），用户业务逻辑无阻塞

* 多线程服务器的常用编程模型

  * non-blocking IO + one loop per thread：处理IO和定时器
    * Event loop 代表了线程的主循环，需要让哪个线程干活，就把 timer 或 IO channel (如 TCP 连接)注册到哪个线程的 loop 里即可。对实时性有要求的 connection 可以单独用一个线程;数据量大的 connection 可以独占一个线程，并把数据处理任务分摊到另几个计算线程中(用线程池);其他次要的辅助性 connections 可以共享一个线程。
    * 线程安全很重要
  * 线程池：处理计算
    * 任务队列 或 生产者消费者数据队列
      * `concurrent_queue<T>`
    * “阻抗匹配”（p80）

* 进程间通信只用 TCP

  * pipe 也有一个经典应用场景，那就是写 Reactor/event loop 时用来[异步唤醒 select (或等价的 poll/epoll_wait)调用](https://www.zhihu.com/question/39752285/answer/82906915)
    * 在 Linux 下，可以用 eventfd(2) 代替，效率更高
    * elf pipe trick，说白了也很简单，windows下select只能针对socket套接字，不能针对管道，一般用构造两个互相链接于[localhost](https://www.zhihu.com/search?q=localhost&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A82906915})的socket来模拟之。不过win下select最多支持同时wait 64个套接字，你摸拟的[pipe](https://www.zhihu.com/search?q=pipe&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A82906915})占掉一个，就只剩下63个可用了。所以java的nio里[selector](https://www.zhihu.com/search?q=selector&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A82906915})在windows下最多支持62个套接字就是被self pipe trick占掉了两个，一个用于其它线程调用notify唤醒，另一个留作jre内部保留，就是这个原因。
  * Note
    * 用 socket_pair(2) 做双向通信
    * 消息格式：推荐Protobuf
    * TCP 的 local 吞吐量不低
    * 另外，除了点对点的通信之外，应用级的广播协议也是非常有用的，可以方便地构建可观可控的分布式系统，见 §7.11
  * 分布式系统中使用 TCP 长连接通信
    * 容易定位分布式系统中的服务之间的依赖关系
    * 通过接收和发送队列的长度也较容易定位网络或程序故障

* 多线程服务器的适用场合

  * 两种方式：宝贵的原生线程（reactor模式，pthread_create），廉价的“线程”（阻塞io，语言runtime调度）

    * Pthreads 是 NPTL(Native POSIX Thread Library) 的，每个线程由 clone(2) 产生，对应一个内核的 task_struct

  * 使用速率为 50MB/s 的数据压缩库、在进程创建销毁 的开销是 800μs、线程创建销毁的开销是 50μs 的前提下，考虑如何执行压缩任务:

    - 如果要偶尔压缩 1GB 的文本文件，预计运行时间是 20s，那么起一个进程去做是合理的，因为进程启动和销毁的开销远远小于实际任务的耗时。
    - 如果要经常压缩 500kB 的文本数据，预计运行时间是 10ms，那么每次都起进程似乎有点浪费了，可以每次单独起一个线程去做。
    - 如果要频繁压缩 10kB 的文本数据，预计运行时间是 200μs，那么每次起线程似乎也很浪费，不如直接在当前线程搞定。也可以用一个线程池，每次把压缩任务交给线程池，避免阻塞当前线程(特别要避免阻塞 IO 线程)。

  * 必须用单线程的场合

    * 程序可能会 fork(2)
      * 立刻执行 exec()，变身为另一个程序。例如 shell 和 inetd; 又比如 lighttpd fork() 出子进程，然后运行 fastcgi 程序。或者集群中运行在计算节点上的负责启动 job 的守护进程(即所谓的“看门狗进程”)。
      * 不调用 exec()，继续运行当前程序。要么通过共享的文件描述符与父进程通信，协同完成任务;要么接过父进程传来的文件描述符，独立完成工作，例如 20 世纪 80 年代的 Web 服务器 NCSA httpd。
    * 限制程序的 CPU 占用率
      * 因此对于一些辅助性的程序，如果它必须和主要服务进程运行在同一台机器的话 (比如它要监控其他服务进程的状态)，那么做成单线程的能避免过分抢夺系统的计算资源。比方说如果要把生产服务器上的日志文件压缩后备份到 NFS 上，那么应该使用普通单线程压缩工具(gzip/bzip2)。它们对系统造成的影响较小，在 8 核服务器上最多占满 1 个 core。

  * 单线程程序的优缺点

    * Event loop 有一个明显的缺点，它是非抢占的(non-preemptive)。这个缺点可以用多线程来克服
    * [IOCP , kqueue , epoll ... 有多重要？](https://blog.codingnow.com/2006/04/iocp_kqueue_epoll.html)
      * 这篇blog讲逻辑服务器前面加一个gateway；gateway定时发数据利于逻辑服务器调试

  * 适用多线程程序的场景

    * 提高响应速度，让 IO 和“计算”相互重叠，降低latency。虽然多线程不能提高绝对性能，但能提高平均响应性能。
    * 多线程间有需要修改的共享数据，提供非均质的服务（对于高优任务防止优先级反转）
    * latency 和 throughput 同样重要，利用异步操作
    * 性能可预测，多线程能有效地划分责任与功能

  * 例子：

    * master-slave，master多线程
      * 4 个用于和 slaves 通信的 IO 线程。
      * 1 个 logging 线程。
      * 1 个数据库 IO 线程。
      * 2 个和 clients 通信的 IO 线程。
      * 1 个主线程，用于做些背景工作，比如 job 调度。
      * 1 个 pushing 线程，用于主动广播机群的状态。
    * TCP聊天服务器：转发连接，更多功能
      * 见 §6.6 的方案 9，以及 p. 260 的实现

  * “多线程服务器的适用场合”例释与答疑

    * Linux 能同时启动多少个线程?

      * 对于 32-bit Linux，一个进程的地址空间是 4GiB，其中用户态能访问 3GiB 左右， 而一个线程的默认栈(stack)大小是 10MB，心算可知，一个进程大约最多能同时启动 300 个线程

    * 多线程能提高并发度吗?

      * thread per connection 不适合高并发场合，其 scalability 不佳。one loop per thread 的并发度足够大，且与 CPU 数目成正比。

    * 多线程能提高吞吐量吗?

      * 对于计算密集型服务，不能。
      * 根据 Amdahl’s law，即便算法的并行度高达 95%，8 核的加速比也只有 6，计算 时间为 0.133s，这样会造成吞吐量下降
      * 线程池也不是万能的，如果响应一次请求需要做比较多的计算(比如计算的时间占整个 response time 的 1/5 强)，那么用线程池是合理的，能简化编程。如果在一次请求响应中，主要时间是在等待 IO，那么为了进一步提高吞吐量，往往要用其他编程模型，比如 Proactor，见问题 8

    * 多线程能降低响应时间吗?

      * 多线程处理输入：并行化IO部分，降低平均延时（减少串行时某些任务的IO等待）
      * 多线程分担负载

    * 多线程程序如何让 IO 和“计算”相互重叠，降低 latency?

      * 所有的网络写操作都可以这么异步地做，不过这也有一个缺点，那就是每次 asyncWrite() 都要在线程间传递数据。其实如果 TCP 缓冲区是空的，我们就可以在本线程写完，不用劳烦专门的 IO 线程。Netty 就使用了这个办法来进一步降低延迟。

    * 第三方库不一定能很好地适应并融入这个 event loop framework

      * 但是检测串口上的某些控制信号(例如 DCD)只能用轮询(ioctl(fd, TIOCMGET, &flags))或阻塞等待(ioctl(fd, TIOCMIWAIT, TIOCM_CAR));要想融入 event loop，需要单独起一个线程来查询串口信 号翻转，再转换为文件描述符的读写事件(可以通过 pipe(2))
      * libmemcached 只支持同步操作

    * 什么是线程池大小的阻抗匹配原则?

      * T = C/P ：密集计算所占的时间比重为 P (0 < P ≤ 1)，而 系统一共有 C 个 CPU，为了让这 C 个 CPU 跑满而又不过载，线程池大小的经验公式

    * 除了你推荐的 Reactor + thread poll，还有别的 non-trivial 多线程编程模型吗?

      * Proactor 模式依赖操作系统或库来高效地调度这些子任务，每个子任务都不会阻

        塞，因此能用比较少的线程达到很高的 IO 并发度。

      * Proactor 能提高吞吐，但不能降低延迟，所以我没有深入研究。另外，在没有语 言直接支持的情况下 26，Proactor 模式让代码非常破碎，在 C++ 中使用 Proactor 是 很痛苦的。因此最好在“线程”很廉价的语言中使用这种方式，这时 runtime 往往会 屏蔽细节，程序用单线程阻塞 IO 的方式来处理 TCP 连接

    * 模式 2 和模式 3a 该如何取舍?

      * 可以根据工作集(work set)的大小来取舍。 工作集是指服务程序响应一次请求所访问的内存大小
      * memcached 这个内存消耗大户用多线程服务端就比在同一台机器上运行多个 memcached instance 要好。(但是如果你在 16GiB 内存的机器上运行 32-bit memcached，那么此时多 instance 是必需的。)
        * 地址空间4GiB，堆栈大小受限；单进程hack成多份地址空间No，多进程指定共享内存Yes

#### chpt 4 C++ 多线程系统编程精要

* 基本线程原语的选用

  * Thread + MutexLock + Condition
  * pthread_once，封装为 muduo::Singleton。其实不如直接用全局变量。
  * pthread_key*，封装为 muduo::ThreadLocal。可以考虑用 __thread 替换之。
  * 不建议使用:
    * pthread_rwlock，读写锁通常应慎用。muduo 没有封装读写锁，这是有意的。
    * sem\_*，避免用信号量(semaphore)。它的功能与条件变量重合，但容易用错。
    * pthread\_{cancel, kill}。程序中出现了它们，则通常意味着设计出了问题。

* C/C++ 系统库的线程安全性

  * 线程的出现立刻给系统函数库带来了冲击，破坏了 20 年来一贯的编程传统和假定。例如：

    * errno 不再是一个全局变量，因为每个线程可能会执行不同的系统库函数。

      * ```c++
        extern int *__errno_location(void);
        // return a lvalue
        #define errno (*__errno_location())
        ```

    * 有些“纯函数”不受影响，例如 memset/strcpy/snprintf 等等。

    * 有些影响全局状态或者有副作用的函数可以通过加锁来实现线程安全，例如malloc/free、printf、fread/fseek 等等。

      * 非线程安全的性能更好的版本：fread_unlocked、fwrite_unlocked 等等，见 `man unlocked_stdio`
      * 例如 fseek() 和 fread() 都是安全的，但是对某个文件“先 seek 再 read”这两步操作中间有可能 会被打断，其他线程有可能趁机修改了文件的当前位置，让程序逻辑无法正确执行。 在这种情况下，我们可以用 flockfile(FILE*) 和 funlockfile(FILE*) 函数来显式地 加锁。并且由于 FILE* 的锁是可重入的，加锁之后再调用 fread() 不会造成死锁。
      * 如果程序直接使用 lseek(2) 和 read(2) 这两个系统调用来随机读取文件，也存 在“先 seek 再 read”这种 race condition，但是似乎我们无法高效地对系统调用加 锁。解决办法是改用 pread(2) 系统调用，它不会改变文件的当前位置。

    * POSIX标准列出 [非线程安全函数的黑名单](https://pubs.opengroup.org/onlinepubs/9699919799/functions/V2_chap02.html#tag_15_09)

      * 有些返回或使用静态空间的函数不可能做到线程安全，因此要提供另外的版本，例如 asctime_r/ctime_r/gmtime_r、stderror_r、strtok_r 等等。

    * 传统的 fork() 并发模型不再适用于多线程程序(§4.9)。

    * 我们不必担心系统调用的线程安全性，因为系统调用对于用户态程序来说是原子的。但是要注意系统调用对于内核状态的改变可能影响其他线程，这个话题留到 §4.6 再细说。

  * 编写线程安全程序的一个难点在于线程安全是不可组合的(composable)

    * e.g. `tzset()` 是全局的，会影响其它线程的时区状态 ---> `muduo::TimeZone`
    * 一个基本思路是尽量把 class 设计成 immutable 的，这样用起来就不必为线程安全操心了
    * C++ 标准库中的绝大多数泛型算法是线程安全的，因为这些都是无状态纯函数。只要输入区间是线程安全的，那么泛型函数就是线程安全的
      * `std::random_shuffle()` 可能是个例外，它用到了随机数发生器
      * 随机数发生不是线程安全的，因为`time(NULL)`随机种子可能一样，导致产生的随机数一样。因此best practice是`static thread_local std::mt19937 rng(std::random_device{}());`
    * C++ 的 iostream 不是线程安全的，因为流式输出是多个operator <<函数调用
      * 如果用`printf`等于加了全局锁

* Linux 上的线程标识

  * pthread_t 并不适合用作程序中对线程的标识符。
  * 在 Linux 上，我建议使用 gettid(2) 系统调用的返回值作为线程 id
    * 在现代 Linux 中，它直接表示内核的任务调度 id，因此在 /proc 文件系统中可以轻易找到对应项:`/proc/ <img src="https://www.zhihu.com/equation?tex=tid%60%20%E6%88%96%20%60/prod/" alt="tid` 或 `/prod/" class="ee_img tr_noresize" eeimg="1"> pid/task/$tid`
    * 任何时刻都是全局唯一的，并且由于 Linux 分配新 pid 采用递增轮回办法，短时间内启动的多个线程也会具有不同的线程 id。
    * 0 是非法值，因为操作系统第一个进程 init 的 pid 是 1

* 线程的创建与销毁的守则
  * 几条创建的原则
    * 程序库不应该在未提前告知的情况下创建自己的“背景线程”。
      * 一旦程序中有不止一个线程，就很难安全地 fork() 了 (§4.9)
    * 尽量用相同的方式创建线程，例如 muduo::Thread。
      * bookkeeping，线程数目可以从 `/proc/pid/status` 拿到
    * 在进入 main() 函数之前不应该启动线程。
      * C++ 保证在进入 main() 之前完成全局对象的构造
        * "全局对象"也包括 namespace 级全局对象、文件级静态对象、class 的静态对象，但不包 括函数内的静态对象
      * 如果一个库需要创建线程，那么应该进入 main() 函数之后再调用库的初始化函数去做
    * 程序中线程的创建最好能在初始化阶段全部完成。
  * 线程的销毁有几种方式
    * 自然死亡。从线程主函数返回，线程正常退出。
    * 非正常死亡。从线程主函数抛出异常或线程触发 segfault 信号等非法操作
    * 自杀。在线程中调用 pthread_exit() 来立刻退出线程。
    * 他杀。其他线程调用 pthread_cancel() 来强制终止某个线程。
      * pthread_kill() 是往线程发信号，留到 §4.10 再讨论
      * 不要他杀！
      * 如果确实需要强行终止一个耗时很长的计算任务，而又不想在计算期间周期性 地检查某个全局退出标志，那么可以考虑把那一部分代码 fork() 为新的进程，这样杀(kill(2))一个进程比杀本进程内的线程要安全得多。当然，fork() 的新进程与 本进程的通信方式也要慎重选取，最好用文件描述符(pipe(2)/socketpair(2)/TCP socket)来收发数据，而不要用共享内存和跨进程的互斥器等 IPC，因为这样仍然有死锁的可能。
  * pthread_cancel 与 C++
    * [Cancellation and C++ Exceptions](https://udrepper.livejournal.com/21541.html)
      * catch-all cases must rethrow
      * `#define CATCHALL catch (abi::__forced_unwind&) { throw; } catch (...)`
  * exit(3) 在 C++ 中不是线程安全的
    * exit(3) 函数在 C++ 中的作用除了终止进程，还会析构全局对象和已经构造完的函数静态对象。这可能导致：死锁、其它线程调用已经被析构的全局对象
    * 如果确实需要主动结束 线程，则可以考虑用 _exit(2) 系统调用。它不会试图析构全局对象，但是也不会执 行其他任何清理工作，比如 flush 标准输出。
* 善用 __thread 关键字
  * 比 `pthread_key_t` 快很多
  * `__thread` 使用规则 27:只能用于修饰 POD 类型，不能修饰 class 类型，因为无法 自动调用构造函数和析构函数。`__thread` 可以用于修饰全局变量、函数内的静态变 量，但是不能用于修饰函数的局部变量或者 class 的普通成员变量。另外，`__thread` 变量的初始化只能用编译期常量。
  * 注意与C++ `threadlocal` 关键字比较
  * 书里举了一些应用的例子（p97）
* 多线程与 IO
  * 网络IO
    * 多个线程同时操作同一个 socket 文件描述符确实很麻烦，chenshuo认为是得不偿失的
    * 各种read/write/connect/clost/listen的情况，太复杂了，而且read/write返回字节数也是不定的
  * 磁盘IO
    * 要避免 lseek(2)/ read(2) 的 race condition(§4.2)
    * 每块磁盘都有一个操作队列，多个线程的读写请求 到了内核是排队执行的。只有在内核缓存了大部分数据的情况下，多线程读这些热数据才可能比单线程快
  * epoll
    * epoll 也遵循相同的原则。Linux 文档并没有说明:当一个线程正阻塞在 epoll_ wait() 上时，另一个线程往此 epoll fd 添加一个新的监视 fd 会发生什么。
    * `muduo::EventLoop::wakeup()`
  * 为了简单起见，我认为多线程程序应该遵循的原则是:每个文件描述符只由一个线程操作，从而轻松解决消息收发的顺序性问题，也避免了关闭文件描述符的各种 race condition
  * 这条规则有两个例外:
    * 对于磁盘文件，在必要的时候多个线程可以同时调用 pread(2)/pwrite(2) 来读写同一个文件;
    * 对于 UDP，由于协议本身保证消息的原子性，在适当的条件下(比如消息之间彼此独立)可以多个线程同时读写同一个 UDP 文件描述符。--->  [相关讨论](https://www.zhihu.com/question/39185963)
* 用 RAII 包装文件描述符
  * POSIX 标准要求每次新打开文件(含 socket)的时候必须使用当前最小可用的文件描述符号码。在多线程程序中，这样很容易串话
  * 在 C++ 里解决这个问题的办法很简单:RAII
  * 引申问题:为什么服务端程序不应该关闭标准输出(fd=1)和标准错误(fd=2)? 
    * 因为有些第三方库在特殊紧急情况下会往 stdout 或 stderr 打印出错信息，如果我们 的程序关闭了标准输出(fd=1)和标准错误(fd=2)，这两个文件描述符有可能被网络连接占用，结果造成对方收到莫名其妙的数据。正确的做法是把 stdout 或 stderr 重定向到磁盘文件(最好不要是 /dev/null)，这样我们不至于丢失关键的诊断信息。 当然，这应该由启动服务程序的看门狗进程完成，对服务程序本身是透明的
  * muduo 使用 shared_ptr 来管理 TcpConnection 的生命期。这是唯一一个采用引用计数方式管理生命期的对象。如果不用 shared_ptr，我想不出其他安全且高效的办法来管理多线程网络服务端程序中的并发连接
* RAII 与 fork()
  * 子进程会继承地址空间和文件描述符，因此用于管理动态内存和文件描述符的 RAII class 都能 正常工作。但是子进程不会继承:
    * 父进程的内存锁，mlock(2)、mlockall(2)。
    * 父进程的文件锁，fcntl(2)。
    * 父进程的某些定时器，setitimer(2)、alarm(2)、timer_create(2) 等等。
    * 其他，见 man 2 fork。
* 多线程与 fork()
  * 多线程与 fork() 的协作性很差。这是 POSIX 系列操作系统的历史包袱，因为以前长期是单线程的设计
    * 无法forkall
  * 在 fork() 之后，子进程就相当于处于 signal handler 之中，你不能调用线程安全的函数(除 非它是可重入的)，而只能调用异步信号安全(async-signal-safe)的函数
  * 唯一安全的做法是在 fork() 之后立即调用 exec() 执行另一个程序， 彻底隔断子进程与父进程的联系。
    * 不得不说，同样是创建进程，Windows 的 CreateProcess() 函数的顾虑要少得多，因为它创建的进程跟当前进程关联较少。
* 多线程与 signal
  * 单线程时代：由于 signal 打断了正在运行的 thread of control，在 signal handler 中只能调用 async-signal-safe 的函数，即 所谓的“可重入(reentrant)”函数，就好比在 DOS 时代编写中断处理例程([ISR](https://en.wikipedia.org/wiki/Interrupt_handler))一样。不是每个线程安全的函数都是可重入的。
    * the **First-Level Interrupt Handler** (**FLIH**) and the **Second-Level Interrupt Handlers** (**SLIH**). FLIHs are also known as *hard interrupt handlers* or *fast interrupt handlers*, and SLIHs are also known as *slow/soft interrupt handlers*, or [Deferred Procedure Calls](https://en.wikipedia.org/wiki/Deferred_Procedure_Call) in Windows.
      * SLIH use kernel threads
    * 如果 signal handler 中需要修改全局数据，那么被修改的变量必须是 `sig_atomic_t`
  * 多线程时代
    * 发送给某一线程(SIGSEGV)，发送给进程中的任一线程(SIGTERM)
    * 在多线程程序中，使用 signal 的第一原则是**不要使用 signal**
    * 不主动处理各种异常信号(SIGTERM、SIGINT 等等)，只用默认语义:结束进程。 有一个例外:SIGPIPE，服务器程序通常的做法是忽略此信号 40，否则如果对方 断开连接，而本机继续 write 的话，会导致程序意外终止
    * 在没有别的替代方法的情况下(比方说需要处理 SIGCHLD 信号)，把异步信号转换为同步的文件描述符事件。现代 Linux 的做法是采用 signalfd(2) 把信号直接转换为文件描述符事件，从而从根本上避免使用 signal handler
      * 例子见 http://github.com/chenshuo/muduo-protorpc 中 Zurg slave 示例的 [ChildManager class](https://github.com/chenshuo/muduo-protorpc/blob/cpp11/examples/zurg/slave/ChildManager.cc)
* Linux 新增系统调用的启示
  * 大致从 Linux 内核 2.6.27 起，凡是会创建文件描述符的 syscall 一般都增加了额外的 flags 参数，可以直接指定 O_NONBLOCK 和 FD_CLOEXEC
    * accept4 - 2.6.28, eventfd2 - 2.6.27, inotify_init1 - 2.6.27, pipe2 - 2.6.27, signalfd4 - 2.6.27, timerfd_create - 2.6.25
  * 另外，以下新系统调用可以在创建文件描述符时开启 FD_CLOEXEC 选项:
    * 以前需要 `fcntl(fd, F_SETFD, FD_CLOEXEC);`
    * open, dup3 - 2.6.27, epoll_create1 - 2.6.27, socket - 2.6.27
    * [Secure File Descriptor Handling](https://udrepper.livejournal.com/20407.html)
      * 例子：web fork出plugins执行exec，不希望主进程已有的私密文件泄露给第三方
      * fork()完立刻set flag并不安全，因为fork()是signal-safe的 (i.e., it can be called from a signal handler).

* Note
  * 在多 CPU 机器上，假设主板上两个物理 CPU 的距离为 15cm，CPU 主频是 2.4GHz，电信号在电路中 的传播速度按 2 × 108m/s 估算，那么在 1 个时钟周期(0.42ns)之内，电信号不能从一个 CPU 到达另一个 CPU。因此对于每个 CPU 自己这个观察者来说，它看到的事件发生的顺序没有全局一致性
  * 在现代 Linux glibc 中，fork(3) 不是直接使用 fork(2) 系统调用，而是使用 clone(2) syscall