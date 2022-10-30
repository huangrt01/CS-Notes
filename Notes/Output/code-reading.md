
[toc]

### Code Reading Tools

* 代码行数统计工具 [cloc](https://github.com/AlDanial/cloc)

* grep, [ack](https://beyondgrep.com/), [ag](https://github.com/ggreer/the_silver_searcher) and [rg](https://github.com/BurntSushi/ripgrep)
  * grep -R can be improved in many ways, such as ignoring .git folders, using multi CPU support, &c

```shell
# Find all python files where I used the requests library
rg -t py 'import requests'
# Find all files (including hidden files) without a shebang line
rg -u --files-without-match "^#!"
# Find all matches of foo and print the following 5 lines
rg foo -A 5
# Print statistics of matches (# of matched lines and files )
rg --stats PATTERN
```

* [C++阅码神器cpptree.pl和calltree.pl的使用 - satanson的文章 - 知乎](https://zhuanlan.zhihu.com/p/339910341)



### sponge (CS144 TCP Lab)

原理和细节参考我的【Computer-Networking-Lab-CS144-Stanford.md】笔记

https://github.com/huangrt01/TCP-Lab

#### abstract

* shared_ptr的使用
  * 只有FDWrapper（event_loop传入多个duplicate）和Buffer用了shared_ptr
  * muduo库作为对比，只有TcpConnection用了shared_ptr
  
* O_NONBLOCK的使用
  
  * writev系统调用只check以下情形，因为写入失败会返回-1，能handle：
  
  * ```c++
    const ssize_t bytes_written = SystemCall("writev", ::writev(fd_num(), iovecs.data(), iovecs.size()));
    if (bytes_written == 0 and buffer.size() != 0) {
      throw runtime_error("write returned 0 given non-empty input buffer");
    }
    
    if (bytes_written > ssize_t(buffer.size())) {
      throw runtime_error("write wrote more than length of input buffer");
    }
    ```
  

#### apps

linux中一切即文件的思想

* bidirectional_stream_copy
  * 这个函数很有意思，衔接了 stdin ~ socket ~ stdout，适用于socket的应用和测试
  * Copy socket input/output to stdin/stdout until finished
    * 是 `man2::poll` 的封装，https://man7.org/linux/man-pages/man2/poll.2.html

  * 将stdin FD、stdout FD、socket都设为unblocking模式
    * 调用`fcntl`设置，几个util类都有set_blocking的方法
* tcp_sponge_socket
  * 重要方法都会调用`_tcp_loop(...)`，_abort强行结束loop，`timestamp_ms()`推动时间
    * `std::atomic_bool _abort{false};  //!< Flag used by the owner to force the TCPConnection thread to shut down`

  * data_socket_pair is a pair of connected AF_UNIX SOCK_STREAM sockets
    * (LocalStreamSocket, _thread_data)

  * `CS144TCPSocket::CS144TCPSocket() : TCPOverIPv4SpongeSocket(TCPOverIPv4OverTunFdAdapter(TunFD("tun144"))) {}`
    * tcp_helpers/ipv4_datagram.*
    * tcp_helpers/tcp_over_ip.*


* tcp_ipv4
  * `LossyTCPOverIPv4SpongeSocket tcp_socket(LossyTCPOverIPv4OverTunFdAdapter( TCPOverIPv4OverTunFdAdapter(TunFD(tun_dev_name == nullptr ? TUN_DFLT : tun_dev_name))));`
* lab7
  * "10" in last two binary digits marks a private Ethernet address
  * router + network_interfaces


#### libsponge

* util
  * file_descriptor
  * buffer
    * BufferList和BufferViewList，用来存多个字符串，支持转化为iovec提供给FileDescriptor写入
  * eventloop
    * Waits for events on file descriptors and executes corresponding callbacks.
    * 附加了 interest() 执行条件和 cancel()
    * EventLoop::wait_next_event

  ```c++
  enum class Direction : short {
    In = POLLIN,   //!< Callback will be triggered when Rule::fd is readable.
    Out = POLLOUT  //!< Callback will be triggered when Rule::fd is writable.
  };
  ```

  * parser
  * socket
    * class Socket : public FileDescriptor
    * Socket有三种，LocalStreamSocket (AF_UNIX) 、 TCPSocket (AF_INET)、UDPSocket (AF_INET, SOCK_DGRAM)
  * util
    * SystemCall的封装，如果系统不允许调用，可以print error
    * InternetChecksum
    * ```c++
      uint64_t timestamp_ms() {
        using time_point = std::chrono::steady_clock::time_point;
        static const time_point program_start = std::chrono::steady_clock::now();
        const time_point now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - program_start).count();
      }
      
      // https://en.cppreference.com/w/cpp/language/parameter_pack
      template <typename... Targs>
      void DUMMY_CODE(Targs &&... /* unused */) {}
    

* byte_stream
  * Buffered i/o，用 BufferList 实现性能较好


* stream_assembly
  * 重要数据结构，A class that assembles a series of excerpts from a byte stream (possibly out of order, possibly overlapping) into an in-order byte stream.
* tcp_receiver
  * 核心数据结构：StreamAssembler、syn和fin的状态
  * TCPSegment
    * payload + TCPHeader
      * payload用Buffer存，浅拷贝
      * TCPHeader 用到 NetParser/NetUnparser (util/parser.*)
    * InternetChecksum (util.*)
  * 主要是 window 相关的逻辑，考虑 seq space 和 payload space 的各种细节
  * 用到 WrappingInt32 数据结构，主要是找到距离 checkpoint 最近的 uint32，以及一些操作符重载
* tcp_sender
  *  “automatic repeat request” (ARQ).
  * fill_window, ack_received 方法，类似于单线程的生产和消费

* network_interface

  * 缓存datagram、5s后再重发ARP Request

* router

  * ```c++
    class AsyncNetworkInterface : public NetworkInterface {
        std::queue<InternetDatagram> _datagrams_out{};
    
      public:
        // 派生类中用using改变基类成员的可访问性
        using NetworkInterface::NetworkInterface;
      	...
    };
    ```

  * Router

    * longest-prefix-match route
    * `dgram.header().ttl`

* tcp_helper

  * ethernet_frame: header + payload
  * ehternet_header: dsc + src + type
  * ipv4_datagram
  * ipv4_header

  ```c++
  ParseResult IPv4Header::parse(NetParser &p) {
    	// 很好地利用了Buffer的特性，string的底层存储不变，不同对象的offset可变
      Buffer original_serialized_version = p.buffer();
    	...
      p.remove_prefix(hlen * 4 - IPv4Header::LENGTH);
  	  InternetChecksum check;
      check.add({original_serialized_version.str().data(), size_t(4 * hlen)});
    	...
  }
  ```

  * arp_message

  * tun

    * TUN device (expects IP datagrams)
    * TAP device (expects Ethernet frames)

  * tuntap_adapter

    * TCPOverIPv4OverTunFdAdapter 
    * TCPOverIPv4OverEthernetAdapter
      * TapFD: Raw Ethernet connection

  * tcp_over_ip: 基于FdAdaptor，互相转化TCPSegment和InternetDatagram

  * tcp_sponge_socket

    * `TCPSpongeSocket : public LocalStreamSocket`

    * Set up the event loop, There are four possible events to handle:

      * Incoming datagram received (needs to be given to `TCPConnection::segment_received` method)
      * Outbound bytes received from local application via a write() call (needs to be read from the local stream socket and given to `TCPConnection::write` method)

      * Incoming bytes reassembled by the TCPConnection (needs to be read from the `inbound_stream` and written to the local stream socket back to the application)
      * Outbound segment generated by TCP (needs to be given to underlying datagram socket)

    * `CS144TCPSocket`

      * `using TCPOverIPv4SpongeSocket = TCPSpongeSocket<TCPOverIPv4OverTunFdAdapter>;`
      * `CS144TCPSocket::CS144TCPSocket() : TCPOverIPv4SpongeSocket(TCPOverIPv4OverTunFdAdapter(TunFD("tun144"))) {}`

    * `FullStackSocket`

      * `FullStackSocket::FullStackSocket() :   TCPOverIPv4OverEthernetSpongeSocket(TCPOverIPv4OverEthernetAdapter(TapFD("tap10"), random_private_ethernet_address(), Address(LOCAL_TAP_IP_ADDRESS, "0"), Address(LOCAL_TAP_NEXT_HOP_ADDRESS, "0"))) {}`

      * `using TCPOverIPv4OverEthernetSpongeSocket = TCPSpongeSocket<TCPOverIPv4OverEthernetAdapter>;`
      

#### shell

```shell
sudo apt install tshark
# window1
./apps/tcp_ipv4 -l 169.254.144.9 9090
# window2
sudo tshark -Pw /tmp/debug.raw -i tun144
# window3
./apps/tcp_ipv4 -d tun145 -a 169.254.145.9 169.254.144.9 9090
```

* 测试 clean shutdown：先关一下client，再关server，再观察client是否linger 10s
* 测试 window_size=1：`-w 1`

* 借助 router + network interface 通信

  ```
  ./apps/lab7 server cs144.keithw.org 3000 [debug]
  ./apps/lab7 client cs144.keithw.org 3001 [debug]
  ```

  * /dev/urandom和/dev/random的区别在于前者不会中断，随机数质量可能更低


```shell
dd if=/dev/urandom bs=1M count=1 of=/tmp/big.txt
./apps/lab7 server cs144.keithw.org 3000 < /tmp/big.txt
</dev/null ./apps/lab7 client cs144.keithw.org 3001 > /tmp/big-received.txt
sha256sum /tmp/big.txt
sha256sum /tmp/big-received.txt
```

* tcpcopy压力测试

### muduo

https://github.com/chenshuo/muduo





#### shell

```shell
# 找到长连接的client，grep tcp6查ipv6监听
netstat -tp(n)a | grep :port

```



### muduo-protorpc

https://github.com/chenshuo/muduo-protorpc

* examples
  * zurg
    * slave/ChildManager: 用signalfd处理SIGCHLD，调用wait4，`onExit` enqueue callback函数


### recipes

https://github.com/chenshuo/recipes https://github.com/huangrt01/recipes

#### datetime

* TimeZone

  * ```c++
    class TimeZone
    {
      public:
        explicit TimeZone(const char* zonefile);
        struct tm toLocalTime(time_t secondsSinceEpoch) const;
        time_t fromLocalTime(const struct tm&) const;
        // default copy ctor/assignment/dtor are okay.
        // ...
    };
    const TimeZone kNewYorkTz("/usr/share/zoneinfo/America/New_York");
    const TimeZone kLondonTz("/usr/share/zoneinfo/Europe/London");
    time_t now = time(NULL);
    struct tm localTimeInNY = kNewYorkTz.toLocalTime(now);
    struct tm localTimeInLN = kLondonTz.toLocalTime(now);
    ```

#### thread

* Atomic

  * boost::detail::AtomicIntegerT<int32_t>

* Mutex.h

  * MutexLock

    * `isLockedByThisThread()`
    * `MutexLockGuard guard(mutex)`

  * 这段代码没有达到工业强度:
    * mutex 创建为 `PTHREAD_MUTEX_DEFAULT` 类型，而不是我们预想的 `PTHREAD_MU- TEX_NORMAL` 类型(实际上这二者很可能是等同的)，严格的做法是用 mutexattr 来显示指定 mutex 的类型。
    * 没有检查返回值。这里不能用 `assert()` 检查返回值，因为 `assert()` 在 release build 里是空语句。我们检查返回值的意义在于防止 `ENOMEM` 之类的资源不足情况，这一般只可能在负载很重的产品程序中出现。一旦出现这种错误，程序必须立刻清理现场并主动退出，否则会莫名其妙地崩溃，给事后调查造成困难。 这里我们需要 non-debug 的 assert，或许 google-glog 的 `CHECK()` 宏是个不错的思路。
    
  * 没有 try_lock
    * trylock 的一个用途是用来观察 lock contention，见 [RWC] “Consider using nonblocking synchroniza- tion routines to monitor contention. ”

  * [why mutable mutex](https://stackoverflow.com/questions/4127333/should-mutexes-be-mutable)
    

* Conditional.h

  * 区分 signal 和 broadcast
  * `waitForSeconds` 接口

* BlockingQueue

  * ```c++
    // always use a while-loop, due to spurious wakeup
    while (queue_.empty()) {
      notEmpty_.wait();
    }
    ```

* BoundedBlockingQueue

  * 两个条件变量empty和full
  
  * boost::circular_buffer 作为数据结构
  
* CountDownLatch

  * `notifyAll`

* Exception

  * `backtrace`，没有demangle

* Thread

  * CurrentThread 利用 `__thread` 线程局部存储关键字标识当前线程tid和名字
    * 万一程序执行了 fork(2)，那么子进程会不会看到 stale 的缓存结果呢? 解决办法是用 `pthread_atfork()` 注册一个回调，用于清空缓存的线程 id。
    * 对比`boost::this_thread::get_id()`
  * ThreadData 存一个 weak_ptr 的 wkTid 对象作为回调，往外传 tid
  * `ThreadNameInitializer init;` 全局变量初始化，为了调用 `pthread_atfork`

* ThreadLocal

  * `pthread_key_create` `pthread_key_delete` `pthread_getspecific` `pthread_setspecific`

* ThreadLocalSingleton

  * See muduo/base/ThreadLocalSingleton.h for how to delete it automatically.
  * 显式初始化static变量  `template<typename T> __thread T* ThreadLocalSingleton<T>::t_value_ = 0;`

* ThreadPool

  * 简洁，很像BlockingQueue，融入了 [Github ThreadPool](https://github.com/progschj/ThreadPool/blob/master/ThreadPool.h) 的实现
  * 捕获异常，`#include "Exception.h"`
  * 如何stop ThreadPool: 用户传进去latch任务主动感知threadpool内的任务是否结束，再调用stop接口
  * 一些可迭代的地方：
    * 加一个`parallel_for`接口，增加任务并行度的参数，融入CountDownLatch的逻辑
    * 增加返回 future 的接口
    * `wait_all_done`，记录在跑的线程数，弄一个conditional
    * 绑核、调度优先级

* SignalSlotTrival.h

* SignalSlot.h: 
  * 线程安全的 Signal/Slots，并且在 Slot 析构时自动 unregister
  * SignalImpl的实现：Copy-on-write，多个shared_ptr指向一份 SlotList，保证 clean 操作的线程安全

* Singleton.h

  * 利用 `PTHREAD_ONCE_INIT`

    * 现代C++写法：

      * ```c++
        std::shared_ptr<Object> object() {
          static std::shared_ptr<Object> object(
              new Object);
          static std::once_flag init_flag;
          std::call_once(init_flag, define_object, object);
          return object;
        }
        
        void define_object(std::shared_ptr<Object> object) {
          ...
        }
        ```

      * 

  * 这个 Singleton 只能调用默认构造函数，如果用户想要指定 T 的构造方式， 我们可以用模板特化(template specialization)技术来提供一个定制点，这需要引入另一层间接(another level of indirection)。

  * `::atexit(destroy);` 聊胜于无

* WeakCallback.h:
  * 利用variadic template和weak_ptr实现弱回调

* test/Observer_safe.cc:
  *  用weak_ptr + enable_shared_from_this()实现observer设计模式

* test/Factory.cc: 对象池的迭代
  * 释放对象：weak_ptr
  * 解决内存泄漏：shared_ptr初始化传入delete函数
  * this指针线程安全问题：enable_shared_from_this
  * Factory生命周期延长：弱回调

* 讨论一下对象池的实现：
  * 一种实现是只增不减，维护一个pool存裸指针，shared_ptr传入deleter，deleter将裸指针塞回pool

* test/一些死锁的例子

  * NonRecursiveMutex_test.cc
  * SelfDeadLock.cc
  * MutualDeadLock.cc

* test/CopyOnWrite 的例子

  * CopyOnWrite_test.cc
  * RequestInventory_test.cc
    * 解决 Request 的析构问题：析构利用智能指针，process完立刻unregister

  * Customer.cc
    * 这里实际上是一个非常特殊的场景（写入量太少了），分布式kv不能这样搞，会增加绝对计算量影响性能
    * 本质上是read-intensive、写入操作少的业务（比如交易业务），可以用CopyOnWrite范式。延伸来说，也可以将读多写少的业务抽象为类似的模型，将写入操作按分钟级聚合为batch。比如借鉴progressive rehash的思路，维护两份hashtable，一份存近期增量，一份存旧数据，新的查找来临时先查bloom filter决定是否查新table，再查旧table。但这样会增加写入的cpu消耗（多查一次hashtable），不一定划算。

### boost

https://github.com/boostorg/boost

Documentation: https://www.boost.org/doc/libs/1_80_0/

[Getting Started](https://www.boost.org/doc/libs/1_80_0/more/getting_started/unix-variants.html)

#### thread

* 如何实现 `this_thread::get_id`
  * `thread/include/boost/thread/detail/thread.hpp`
    * `get_id()`
    * 用 `thread::id` 存 `detail::thread_data_ptr`，id类对指针做比较操作
  * `thread/src/pthread/thread.cpp`
    * 用`pthread_getspecific`实现ThreadLocal对象存储
    * `make_external_thread_data()`

#### smart_ptr

https://github.com/boostorg/smart_ptr

* detail

  * sp_counted_base_std_atomic.hpp

    * 一些atomic操作：increment, decrement, conditional_increment
    * virtual function: despose(), destroy(), get_deleter(sp_typeinfo_ const & ti)
      * sp_type_info_.hpp: `typedef std::type_info sp_typeinfo_;`

  * sp_counted_impl.hpp

    * ```c++
      template<class X> class BOOST_SYMBOL_VISIBLE sp_counted_impl_p: public sp_counted_base
      {
      private:
      
          X * px_;
        	...
      }
      
      void dispose() BOOST_SP_NOEXCEPT BOOST_OVERRIDE
      {
        boost::checked_delete( px_ );
      }
      ```

* shared_ptr.hpp

  * deleter 如何传：
    * sp_deleter_construct -> boost::detail::sp_enable_shared_from_this
    * sp_enable_shared_from_this
      * 将 Deleter 推导为 `enable_shared_from_raw* pe;` 
    * enable_shared_from_raw.hpp
      * `pe->_internal_accept_owner( ppx, const_cast< Y* >( py ) )`
      * `detail::esft2_deleter_wrapper * pd = boost::get_deleter<detail::esft2_deleter_wrapper>( shared_this_ );`
      * `pd->set_deleter( *ppx );`
      * 然后用 `shared_this_` 生成 ppx，相当于给 ppx 内部的 shared_count 注册 deleter (`sp_counted_base pi_;`)

* shared_count.hpp
  * `sp_counted_impl_pd` 初始化 `pi_`
  * 析构调用 sp_counted_impl::dispose



* 读文档：https://www.boost.org/doc/libs/1_80_0/libs/smart_ptr/doc/html/smart_ptr.html#introduction

  * [handle-body idiom](https://www.cs.vu.nl/~eliens/online/tutorials/objects/patterns/handle.html) 可以 [用 scoped_ptr 或 shared_ptr 来做](https://www.boost.org/doc/libs/1_80_0/libs/smart_ptr/example/scoped_ptr_example.hpp)
  * shared_ptr
    * 有一些atomic相关的member function
    * 为什么统一shared_ptr的实现：否则a reference counted pointer (used by library A) cannot share ownership with a linked pointer (used by library B.)
    * `make_shared` `allocated_shared` 可读性、control block内存一起分配
  * `make_unique`, `allocated_unique`
  * intrusive_ptr: Managing Objects with Embedded Counts
    * Some existing frameworks or OSes provide objects with embedded reference counts;
    * The memory footprint of `intrusive_ptr` is the same as the corresponding raw pointer;
    * `intrusive_ptr<T>` can be constructed from an arbitrary raw pointer of type `T*`.
  * intrusive_ref_counter
  * local_shared_ptr, make_local_shared
  * Generic Pointer Casts
    * https://www.boost.org/doc/libs/1_80_0/libs/smart_ptr/test/pointer_cast_test.cpp
  * pointer_to_other

  ```c++
  #include <boost/pointer_to_other.hpp>
  
  template <class VoidPtr>
  class memory_allocator
  {
      // Predefine a memory_block
  
      struct block;
  
      // Define a pointer to a memory_block from a void pointer
      // If VoidPtr is void *, block_ptr_t is block*
      // If VoidPtr is smart_ptr<void>, block_ptr_t is smart_ptr<block>
  
      typedef typename boost::pointer_to_other
          <VoidPtr, block>::type block_ptr_t;
  
      struct block
      {
          std::size_t size;
          block_ptr_t next_block;
      };
  
      block_ptr_t free_blocks;
  };
  ```

  * atomic_shared_ptr

  * [`std::owner_before`](https://en.cppreference.com/w/cpp/memory/shared_ptr/owner_before) 便于智能指针做key

    ```c++
    std::unordered_set< boost::shared_ptr<void>,
      boost::owner_hash< boost::shared_ptr<void> >,
      boost::owner_equal_to< boost::shared_ptr<void> > > set;
    ```

* [Appendix A: Smart Pointer Programming Techniques](https://www.boost.org/doc/libs/1_80_0/libs/smart_ptr/doc/html/smart_ptr.html#techniques)

  * Using incomplete classes for implementation hiding

    * This technique relies on `shared_ptr`’s ability to execute a custom deleter, eliminating the explicit call to `fclose`, and on the fact that `shared_ptr<X>` can be copied and destroyed when `X` is incomplete.

  * The "Pimpl" idiom

  * Using abstract classes for implementation hiding

  * Preventing `delete px.get()`   ---> use a private/protected deleter

  * Encapsulating allocation details, wrapping factory functions

    * ```c++
      shared_ptr<X> createX()
      {
          shared_ptr<X> px(CreateX(), DestroyX);
          return px;
      }
      ```

  * Using a shared_ptr to hold a pointer to a statically allocated object

    * ```c++
      struct null_deleter
      {
          void operator()(void const *) const
          {
          }
      };
      
      static X x;
      
      shared_ptr<X> createX()
      {
          shared_ptr<X> px(&x, null_deleter());
          return px;
      }
      ```

  * Using a shared_ptr to hold a pointer to a COM Object

  * Using a shared_ptr to hold a pointer to an object with an embedded reference count

  * Using a shared_ptr to hold another shared ownership smart pointer

  * Obtaining a shared_ptr from a raw pointer

  * Obtaining a shared_ptr (weak_ptr) to this in a constructor

    * Depending on context, if the inner `shared_ptr this_` doesn’t need to keep the object alive, use a `null_deleter`
    * If `X` is supposed to always live on the heap, and be managed by a `shared_ptr`, use a static factory function

  * Obtaining a shared_ptr to this

    * solution 1: keep a weak pointer to `this` as a member in `impl`
    * solution 2: `enable_shared_from_this()`

  * Using shared_ptr as a smart counted handle

    * ```c++
      typedef shared_ptr<void> handle;
      
      handle createProcess()
      {
          shared_ptr<void> pv(CreateProcess(), CloseHandle);
          return pv;
      }
      ```

  * **Using shared_ptr to execute code on block exit**

    * ```c++
      // executing f(p)
      shared_ptr<void> guard(p, f);
      // executing f(x,y)
      shared_ptr<void> guard(static_cast<void*>(0), bind(f, x, y));
      ```

  * Associating arbitrary data with heterogeneous shared_ptr instances

    * ```c++
      typedef int Data;
      
      std::map<shared_ptr<void>, Data> userData;
      // or std::map<weak_ptr<void>, Data> userData; to not affect the lifetime
      
      shared_ptr<X> px(new X);
      shared_ptr<int> pi(new int(3));
      
      userData[px] = 42;
      userData[pi] = 91;
      ```

  * Using shared_ptr as a CopyConstructible mutex lock

    * ```c++
      class mutex
      {
      public:
          void lock();
          void unlock();
      };
      
      shared_ptr<mutex> lock(mutex & m)
      {
          m.lock();
          return shared_ptr<mutex>(&m, mem_fn(&mutex::unlock));
      }
      ```

    * ```c++
      class shared_lock
      {
      private:
          shared_ptr<void> pv;
      public:
          template<class Mutex> explicit shared_lock(Mutex & m): pv((m.lock(), &m), mem_fn(&Mutex::unlock)) {}
      };
      ...
      shared_lock lock(m);
      ```

  * Using shared_ptr to wrap member function calls

    * ```c++
      template<class T> class pointer
      {
      private:
          T * p_;
      
      public:
          explicit pointer(T * p): p_(p)
          {
          }
      
          shared_ptr<T> operator->() const
          {
              p_->prefix();
              return shared_ptr<T>(p_, mem_fn(&T::suffix));
          }
      };
      
      class X
      {
      private:
          void prefix();
          void suffix();
          friend class pointer<X>;
      
      public:
          void f();
          void g();
      };
      
      int main()
      {
          X x;
      
          pointer<X> px(&x);
      
          px->f();
          px->g();
      }
      ```

  * Delayed deallocation

    * ```c++
      vector< shared_ptr<void> > free_list;
      
      class Y
      {
          shared_ptr<X> px;
      
      public:
      
          void f()
          {
              free_list.push_back(px);
              px.reset();
          }
      };
      
      // periodically invoke free_list.clear() when convenient
      ```

  * Weak pointers to objects not managed by a shared_ptr

    * Make the object hold a `shared_ptr` to itself, using a `null_deleter`

#### boost 的使用

```shell
apt-cache search boost
sudo apt-get install libboost-all-dev

"-lpthread",
"-lboost_thread",
"-lboost_system"
```

### illumos

https://illumos.org/

[LISA11 - Fork Yeah! The Rise and Development of illumos](https://www.youtube.com/watch?v=-zRN7XLCRhc)

* uts/common/os/fio
  * http://home.mit.bme.hu/~meszaros/edu/oprendszerek/segedlet/unix/4_fajlrendszerek/solaris_internals_ch14_file_system_framework.pdf
  * [flist_grow](https://github.com/illumos/illumos-gate/blob/master/usr/src/uts/common/os/fio.c#L338): use memory retiring to implement per-chain hash-table locks，避免hashtable查找时加全局锁来获取当前hashtable的指针
    * fi_list是volatile，并且要加memory barrier
    * fi_lock protects fi_list and fi_nfiles
    * ufp->uf_lock 只在发现 size 变大时锁住（调用 UF_ENTER）
    * flist_grow() must acquire all such locks -- fi_lock and every fd's uf_lock -- to install a new file list
* uts/common/sys/user.h
  * [UF_ENTER](https://github.com/illumos/illumos-gate/blob/master/usr/src/uts/common/sys/user.h#L176)


* uts/common/os/kmem.c
  
  
  * https://www.cs.dartmouth.edu/~sergey/cs108/2015/solaris_kernel_memory.pdf
	
	* For example, the Solaris kernel memory allocator has per-CPU caches of memory buffers. When a CPU exhausts its per-CPU caches, it must obtain a new series of buffers from a global pool. Instead of simply acquiring a lock in this case, the code [*attempts* to acquire the lock](https://github.com/illumos/illumos-gate/blob/master/usr/src/uts/common/os/kmem.c#L2090), incrementing a counter when this fails (and then acquiring the lock through the blocking entry point). If the counter reaches a predefined threshold, the size of the per-CPU caches is increased, thereby dynamically reducing contention.
	  * `int kmem_depot_contention = 3;`
	
	  * `if (cp->cache_chunksize < cp->cache_magtype->mt_maxbuf && (int)(cp->cache_depot_contention - cp->cache_depot_contention_prev) > kmem_depot_contention) {need_magazine_resize = 1;}`
	
	    
	
### tinyflow

#### abstract

#### examples

* softmax

  * ```python
    def softmax(x):
        x = x - np.max(x, axis=1, keepdims=True)
        x = np.exp(x)
        x = x / np.sum(x, axis=1, keepdims=True)
        return x
    ```

  * 



### tensorflow

#### abstract

[一篇很好的回答，梳理tf的核心框架思路](https://www.zhihu.com/question/51216952/answer/124708405) MUSTDO

#### Ops

* dropout
  * keras/layers/core.py: `Class Dropout(Layer)` -> `nn.dropout`
  * ops/nn_ops.py: _dropout()
    * 注意是在training时进行scale，推理时忽略



### TODO

这有一系列关于hash_map精巧设计的文章 https://preshing.com/20160222/a-resizable-concurrent-map/

配合经典的tbb实现阅读：https://oneapi-src.github.io/oneTBB/main/tbb_userguide/concurrent_hash_map.html

https://github.com/oneapi-src/oneTBB/blob/master/include/oneapi/tbb/concurrent_hash_map.h



读ulib的solaris实现的avl树

https://github.com/huangrt01/ulib

perf/avl

Second (and perhaps counterintuitively), one can achieve concurrency and composability by having no locks whatsoever. In this case, there must be no global subsystem state—subsystem state must be captured in per-instance state, and it must be up to consumers of the subsystem to assure that they do not access their instance in parallel. By leaving locking up to the client of the subsystem, the subsystem itself can be used concurrently by different subsystems and in different contexts

* A concrete example of this is the AVL tree implementation used extensively in the Solaris kernel. As with any balanced binary tree, the implementation is sufficiently complex to merit componentization, but by not having any global state, the implementation may be used concurrently by disjoint subsystems—the only constraint is that manipulation of a single AVL tree instance must be serialized.
