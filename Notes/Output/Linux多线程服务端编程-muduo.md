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
    	MutexLockGuard lock(mutex); queue.push_back(x);
    	cond.notify(); // 可以移出临界区之外
    }
    ```

  * broadcast should generally be used to indicate state change rather than resource availability

  * Mutex 和 Condition，就像与非门和 D 触发器构成了数字电路设计所需的全部基础元件，可以完成任何组合和同步时序逻辑电路设计一样

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