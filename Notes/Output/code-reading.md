
[toc]

### Code Reading Tools

* 代码行数统计工具 [cloc](https://github.com/AlDanial/cloc)

* 树形文件结构

```shell
tree --charset=ascii -I "bin|unitTest" -P "*.[ch]|*.[ch]pp"
tree -d
```

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

#### cpptree & calltree

* [C++阅码神器cpptree.pl和calltree.pl的使用 - satanson的文章 - 知乎](https://zhuanlan.zhihu.com/p/339910341)
  * https://github.com/satanson/cpp_etudes/blob/master/calltree.pl
  * 代码阅读的切入点不一，应该视系统的开发动机和workload而定. 比如：
    * 严格存储系统: 搜fsync, fdatasync, sync_file_range, wal, prepare, commit；
    * 基于volcano的执行引擎：搜ExecNode， processor， fragment；
    * JIT的源码：搜optimize， phi， compile， interrupt， jit, vm, virtualmachine, x86, x86_64；
    * SQL分析器：搜parse， rewrite， plan，analyze, ast;
    * 网络库： 搜epoll，reuse_port, listen, accept, write, read, off/in-eventloop.
    * 推荐系统：搜fetch、inference
  

```shell
cpptree '\w+' '' 1 3
cpptree '(?i)\w+' '(?i)poll' 1 4

calltree '(?i)(dispatch|submit|queue|put|push)' '' 1 1 1
```

* [如何查看大型工程的源代码？- satanson的回答 - 知乎](https://www.zhihu.com/question/21499539/answer/1958299570)
  * 几条关键原则
    * 方法很重要, 但是看大型项目的代码, 态度+意愿+决心更加重要. 学习非线性, 突破在刹那. 学习方法都是ok的，学习结果的差异有赖于非理性因素. 学习本身和实践不同，是一个很唯心的过程. 
    * 把代码手动编译, 手动部署起来，跑起来.  手动编译可以帮助后面添加日志打印观察输出；手动部署而非官方提供的一键部署, 这样做的好处是，可以感知系统架构, 感知每个组件角色和职责, 感知彼此之间的交互关系.  
    * 找一个经典的workload让系统执行. 比如tpc-h的Q5用来研究OLAP系统的代码就是一个很好例子. 在源码中添加日志打印, 查看线程, 用perf工具看栈，用`gdb -q -batch -ex "thread apply all bt"`看栈, 看栈的方法比较多.  看配置文件的生效方式用`strace -e 'open' -e 'openat'`; 在/proc/pid/下面多看看, 可以发现不少关键信息.  用tree命令看看系统的数据, 元数据, 日志目录的组织形式.  形成自己的使用偏好.
      * 看栈：我的【dotfiles】-aliases-pstack[full]，`sudo pstack[full] -p $pid`
    * 添加日志是一个屡试不爽的好方法，添加backtrace. C++项目的backtrace,  笔者曾经在看arangodb的时候，编写过，主要用backtrace, bactrace_symbols和demangle.
      * TODO：试用backtrace代码
  * 看代码myth
    * 学习代码的最好的方式，就是找一个磨刀石同学，拉到小黑板面前, 你讲他听，他攻你守
    * 看代码是一种认知行为. 什么是认知，记忆+识别+重现，就是你记住了一个东西，当完全一样或者相似物出现的时候，可以识别并且归纳为一类，你可以通过语言文字把这个东西转述给别人. 评价是高于认知的，实践是高于评价的. 因此还需要评价和实践夯实学习，还需要咀嚼反刍修正认知.
  * 看代码的方法学
    * demo系统、增量学习、实践试错、磨刀石



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



#### 代码结构

* `**` 表示用户不可见的内部类
* 对于使用 muduo 库而言，只需要掌握 5 个关键类:Buffer、 EventLoop、TcpConnection、TcpClient、TcpServer
  * 用户不可见类通过前向声明技术隐藏，比如 TcpClient.h 前向声明 Connector

```
// 主体
muduo 
\-- net
      |** Acceptor.{h,cc}                 接受器，用于服务端接受连接
      |-- Buffer.{h,cc}                   缓冲区，非阻塞IO必备
      |-- Callbacks.h
      |-- Channel.{h,cc}                  用于每个Socket连接的事件分发
      |-- CMakeLists.txt
      |** Connector.{h,cc}                连接器，用于客户端发起连接
			|-- Endian.h                        网络字节序与本机字节序的转换
      |-- EventLoop.{h,cc}                事件分发器
      |-- EventLoopThread.{h,cc}          新建一个专门用于EventLoop的线程
      |-- EventLoopThreadPool.{h,cc}      muduo默认多线程IO模型
      |-- InetAddress.{h,cc}              IP地址的简单封装
      |** Poller.{h,cc}                   IO multiplexing的基类接口
			|** poller                          IO multiplexing的实现
      |   |-- DefaultPoller.cc            根据环境变量MUDUO_USE_POLL选择后端
      |   |-- EPollPoller.{h,cc}          基于epoll(4)的IO multiplexing后端
      |   \-- PollPoller.{h,cc}           基于poll(2)的IO multiplexing后端
      |** Socket.{h,cc}                   封装Sockets描述符，负责关闭连接
      |** SocketsOps.{h,cc}               封装底层的Sockets API
      |-- TcpClient.{h,cc}                TCP客户端
      |-- TcpConnection.{h,cc}            最大的一个类，有300多行
      |-- TcpServer.{h,cc}                TCP服务端
			|-- tests                           简单测试
      |** Timer.{h,cc}                    以下几个文件与定时器回调相关
      |-- TimerId.h
      \** TimerQueue.{h,cc}

// 网络附属库，使用的时候需要链接相应的库，例如 -lmuduo_http、-lmuduo_inspect 等等
// HttpServer 和 Inspector 暴露出一个 http 界面，用于监控进程的状态，类似于 Java JMX(§9.5)

muduo 
`-- net
    |-- http         不打算做成通用的HTTP服务器，这只是简陋而不完整的HTTP协议实现
    |   |-- HttpContext.h
    |   |-- HttpRequest.h
    |   |-- HttpResponse.h
    |   |-- HttpServer.h
    |   `-- tests              示范如何在程序中嵌入HTTP服务器
    |-- inspect                基于HTTP协议的窥探器，用于报告进程的状态
    |   |-- Inspector.h
    |   |-- PerformanceInspector.h
    |   |-- ProcessInspector.h
    |   |-- SystemInspector.h
    |   `-- tests
    |-- protobuf
    |   |-- BufferStream.h
    |   `-- ProtobufCodecLite.h
    |-- protorpc               简单实现Google Protobuf RPC
    |   |-- google-inl.h
    |   |-- RpcChannel.h
    |   |-- RpcCodec.h
    |   `-- RpcServer.h
```

```shell
cpptree '(?i)\w+' '' 1 3

# 找epoll调用、EventLoop的创建
calltree '(?i)epoll' '' 1 1 3
calltree '(?i)(create|new|make)\w*(poller|eventloop)' '' 1 1 
cpptree '(?i)EventLoop' '' 1 4

# off-EventLoop处理
calltree '(?i)loop' '' 1 1 1
calltree '(?i)in.*loop' '' 1 1 1
calltree '(?i)(dispatch|submit|queue|put|push)' '' 1 1 1
calltree 'read' '' 1 1 2
calltree '(?i)handleRead' '' 1 1 2
calltree '(?i)handleRead' '' 0 1 1

ag 'TcpConnection::handleRead'
vim net/TcpConnection.cc +54

# 找callback
ag 'messageCallback_'
calltree '(?i)^set(\w+)callback' '' 1 1 1
# 找callback触发点
calltree '(?i)callback_' '' 1 1 2

# 是否支持多个Acceptor
calltree '(?i)reuse.*port' '' 1 1 3
```





#### net's interface

* Buffer 仿 Netty ChannelBuffer 的 buffer class，数据的读写通过 buffer 进行。 用户代码不需要调用 read(2)/write(2)，只需要处理收到的数据和准备好要发 送的数据(§ 7.4)。
  * findCRLF(): `const char Buffer::kCRLF[] = "\r\n";` , `std::search`

* InetAddress 封装 IPv4 地址(end point)
  * 注意，它不能解析域名，只认 IP 地址。因为直接用 gethostbyname(3) 解析域名会阻塞 IO 线程
  * static函数resolve
* EventLoop 事件循环(反应器 Reactor)，每个线程只能有一个 EventLoop 实体， 它负责 IO 和定时器事件的分派。
  * 用 eventfd(2) 来异步唤醒，这有别于传统的用一对 pipe(2) 的办法。
  * 用 TimerQueue 作为计时器管理，用 Poller 作为 IO multiplexing。
* EventLoopThread 启动一个线程，在其中运行 EventLoop::loop()。
  * getNextLoop: round-robin给connection分配loop

* TcpConnection 整个网络库的核心，封装一次 TCP 连接，注意它不能发起连接
  * `boost::any context_;`
  * `HttpContext* context = boost::any_cast<HttpContext>(conn->getMutableContext());`

* TcpClient 用于编写网络客户端，能发起连接，并且有重试功能
* TcpServer 用于编写网络服务器，接受客户的连接。

##### Note

* TcpConnection 的生命期依靠 `shared_ptr` 管理(即用户和库共同控制)。Buffer 的生命期由 TcpConnection 控制。其余类的生命期由用户控制
* Buffer 和 InetAddress 具有值语义，可以拷贝;其他 class 都是对象语义，不可以拷贝

#### net's internal

- Channel 是 selectable IO channel，负责注册与响应 IO 事件，注意它不拥有 file descriptor。它是 Acceptor、Connector、EventLoop、TimerQueue、TcpConnection 的成员，生命期由后者控制。
- Socket是一个RAIIhandle，封装一个filedescriptor，并在析构时关闭fd。它是 Acceptor、TcpConnection 的成员，生命期由后者控制。EventLoop、TimerQueue 也拥有 fd，但是不封装为 Socket class。
- SocketsOps 封装各种 Sockets 系统调用。
- Poller 是 PollPoller 和 EPollPoller 的基类，采用“电平触发”的语意。它是 EventLoop 的成员，生命期由后者控制。
  - 维护channels，注意hasChannel的条件，考虑了fd channel变化
  - interface: poll, updateChannel, removeChannel
    - poll: set_revents
      - poll和epoll的逻辑区别：
        - epoll的event里存了channel指针
        - poll用channel的idx维护其在pollfds中的index

  - Internal 
    - `Timestamp now(Timestamp::now()); ... return now;`
    - `update(): event.events = channel->events(); event.data.ptr = channel;` 

* PollPoller 和 EPollPoller 封装 poll(2) 和 epoll(4) 两种 IO multiplexing 后端。poll 的存在价值是便于调试，因为 poll(2) 调用是上下文无关的，用 strace(1) 很容易知道库的行为是否正确。

- Connector 用于发起 TCP 连接，它是 TcpClient 的成员，生命期由后者控制。
- Acceptor 用于接受 TCP 连接，它是 TcpServer 的成员，生命期由后者控制。
  - TcpServer::kReusePort 控制 Acceptor 的 reuse port 开启

- TimerQueue 用 timerfd 实现定时，这有别于传统的设置 poll/epoll_wait 的等待时长的办法。TimerQueue 用 std::map 来管理 Timer，常用操作的复杂度是 O(log N )，N 为定时器数目。它是 EventLoop 的成员，生命期由后者控制。
- EventLoopThreadPool 用于创建 IO 线程池，用于把 TcpConnection 分派到某个 EventLoop 线程上。它是 TcpServer 的成员，生命期由后者控制。

##### Note

* 关于callback
  * 从read事件触发，到一路回调下来，在回调里把消息给工作线程池处理. 工作线程池处理完成后, 把响应给回到EventLoop，然后发出去

#### 网络附属库

* net/http
  * `void defaultHttpCallback(const HttpRequest&, HttpResponse* resp)`

#### examples

```
examples
|-- ace
|   |-- logging
|   `-- ttcp
|-- asio            从 Boost.Asio 移植的例子
|   |-- chat          多人聊天的服务端和客户端，示范打包和拆包(codec) 
|   `-- tutorial    一系列 timers
|-- cdns            基于 c-ares 的异步 DNS 解析
|-- curl            基于 curl 的异步 HTTP 客户端
|-- fastcgi
|-- filetransfer    简单的文件传输，示范完整发送 TCP 数据
|   `-- loadtest
|-- hub             一个简单的 pub/sub/hub 服务，演示应用级的广播
|-- idleconnection  踢掉空闲连接
|-- maxconnection   控制最大连接数
|-- memcached
|   |-- client
|   `-- server
|-- multiplexer     1:n 串并转换服务
|-- netty           从 JBoss Netty 移植的例子
|   |-- discard        可用于测试带宽，服务器可多线程运行
|   |-- echo           可用于测试带宽，服务器可多线程运行
|   `-- uptime         带自动重连的 TCP 长连接客户端
|-- pingpong        pingpong 协议，用于测试消息吞吐量
|-- procmon
|-- protobuf        Google Protobuf 的网络传输示例
|   |-- codec       	自动反射消息类型的传输方案
|   |-- resolver			RPC 示例，实现 Sudoku 服务
|   |-- rpc
|   |-- rpcbalancer
|   `-- rpcbench
|-- roundtrip				测试两台机器的网络延时与时间差
|-- shorturl				简单的短址服务
|-- simple					5 个简单网络协议的实现
|   |-- allinone
|   |-- chargen				RFC 864，可测试带宽
|   |-- chargenclient
|   |-- daytime				RFC 867
|   |-- discard				RFC 863
|   |-- echo					RFC 862
|   |-- time					RFC 868
|   `-- timeclient
|-- socks4a					Socks4a 代理服务器，示范动态创建 TcpClient
|-- sudoku					数独求解器，示范 muduo 的多线程模型
|-- twisted					从 Python Twisted 移植的例子
|   `-- finger
|-- wordcount
`-- zeromq					从 ZeroMQ 移植的性能(消息延迟)测试

# http://github.com/chenshuo/muduo-udns:基于 UDNS 的异步 DNS 解析
# http://github.com/chenshuo/muduo-protorpc:新的 RPC 实现，自动管理对象生命期
```

* ping_pong：击鼓传花，书 6.5.2
  * `server.setThreadNum(threadCount);`
  * Client多线程：`threadPool_.getNextLoop()`
    * 每个Session配一个TcpClient，Session存Client指针方便回调
  * on_disconnect: `conn->getLoop()->queueInLoop(std::bind(&Client::quit, this));`
  * `bench.cc`: run_once, 然后调用channel的disableAll()和remove()方法
  * 为了运行测试
    * 调高每个进程能打开的文件数，比如设为 256000
    * `conn->setTcpNoDelay(true);`
  
* shorturl: SO_REUSEPORT

* simple/echo

  * EchoServer不需要继承任何类，在构造函数里给TcpServer注册几个回调函数

  * ```shell
    ./echoserver_unittest
    ./echoclient_unittest 0.0.0.0
    ```

* sudoku：详解muduo多线程模型

  * server_basic.cc: 方案5
  * server_threadpool.cc：方案8
  * server_multiloop.cc: 方案9
    * `EventLoopThreadPool`
    * 方案5 + `server_.setThreadNum(numThreads);` 
  * 方案11: 方案8 + `server_.setThreadNum(numThreads);` 

* twisted/finger: 若干个基础例子



#### logging
base文件夹中

* Mutex

  * ThreadAnnotation

* Logging: 日志前端

  * 每次LOG_INFO都是一个新的Logging对象，对象析构时finish()添加换行符，然后g_output、g_flush

  * enum最后加一个NUM_LOG_LEVEL，便于定义定长数组给编译器做优化

  * SourceFile：每行日志消息的源文件名部分采用了编译期计算来获得 basename，避免运行期 strrchr(3) 开销。这里利用了 gcc 的内置函数。

    * ```c++
      template<int N>
      SourceFile(const char (&arr)[N]) : data_(arr), size_(N-1) {}
      
      SourceFile(__FILE__)
      ```

  * helper class T: for known string length at compile time，定义了LogStream的<<
  * `Impl::formatTime()`: snprintf
    * 时间戳字符串中的日期和时间两部分是缓存的，一秒之内的多条日志只需重新格式化微秒部分，利用了`__thread`
  * Fatal log会core在logger析构里，有改进空间
  * 在线调整日志级别：gdb内 `call (void)'muduo::Logger::setLogLevel'(3)`

* LogFile: 日志后端

  * append, flush, RollFile

    * `append->append_unlocked->AppendFile::append,`
    * 3s flush一次，24h rollfile一次

  * ```c++
    char name[256] = { '\0' };
    strncpy(name, argv[0], sizeof name - 1);
    g_logFile.reset(new muduo::LogFile(::basename(name), 200*1000));
    ```

  * FileUtil::AppendFile

    * `::setbuffer(fp_, buffer_, sizeof buffer_)`
    * `::fwrite_unlocked`
    
  * FileUtil::ReadSmallFile, `::fstat` , `::read`

* LogStream

  * muduo 没有用标准库中的 iostream，这主要是出于性能原因(§ 11.6.6)

  * FixedBuffer

    * `typedef detail::FixedBuffer<detail::kSmallBuffer> Buffer;`
    * LogStream用4KB，AsyncLogging用4MB (至少1k条日志)

  * convert, formatSI, formatIEC

  * 支持 `LogStream& operator<<(LogStream& s, const Fmt& fmt);`

  * ```C
    // TODO: better itoa.
    #if defined(__clang__)
    #pragma clang diagnostic ignored "-Wtautological-compare"
    #else
    #pragma GCC diagnostic ignored "-Wtype-limits"
    #endif
    ```

* AsyncLogging: 将日志数据从多个前端高效地传输到后端

  * 后台线程，用latch确保启动
  * [double buffering技术](https://en.wikipedia.org/wiki/Multiple_buffering)
    * `currentBuffer_`满了再notify后端做IO
    * 默认后端3s写一次，非常规的 condition variable 用法，没有使用 while 循环，而且等待时间有上限
    * `buffers_`和`BuffersToWrite`指针交换，持有锁的时间很短
    * `nextBuffer_`和`currentBuffer_`的创建都由后台线程做
    * 考虑Page_fault，后端尽量将最早的Buffer归还给前端

  * 拥塞控制
    * 如果前端很忙，也可能给`currentBuffer_`分配内存；拥塞控制由后端进行，保证后台线程循环速度，间接减轻前端压力
    * 处理日志堆积的方法很简单:直接丢掉多余的日志 buffer，以腾出内存

  * 优化相关：
    * 前后端copy日志比传递日志指针要快，不用每次内存分配
    * 锁：改进方向可以像 Java 的 ConcurrentHashMap 那样用多个桶子(bucket)，前端写日志的时候再按线程 id 哈希到 不同的 bucket 中，以减少 contention
    * 拥塞控制：`nextBuffer_` 替换为`emptyBuffers_`
    * Make the logging thread a lower priority so it won't starve the main application thread.
    * 另可参考 http://highscalability.com/log-everything-all-time

* tests

  * LogFile_test：使用示例
  * Logging_test.cc：测试性能

* 优化相关

  * 日志消息的前 4 个字段是定长的，因此可以避免在运行期求字符串长度，`Logging::operator<<(LogStream& s, T v)`
    * 编译器认识 memcpy() 函数，对于定长的内存复制， 会在编译期把它 inline 展开为高效的目标代码

  * 线程 id 预先格式化为字符串，`CurrentThread::tidString()`

#### Scripts

```shell
# 依赖，参考.travis.yml
sudo apt-get install g++ cmake make 
sudo apt-get install libboost-dev libboost-test-dev libboost-program-options-dev libboost-system-dev
sudo apt-get install libcurl4-openssl-dev libc-ares-dev
sudo apt-get install protobuf-compiler libprotobuf-dev libprotoc-dev libgoogle-perftools-dev


mkdir build
cd build
BUILD_TYPE=debug ../build.sh -j4
# 编译 muduo 库和它自带的例子，生成的可执行文件和静态库文件，分别位于 ../build/release-cpp11/{bin,lib}

../build.sh install
# 以上命令将 muduo 头文件和库文件安装到 ../build/release-install-cpp11/{include,lib}， 以便 muduo-protorpc 和 muduo-udns 等库使用
```

```shell
./release-cpp11/bin/inspector_test
curl localhost:12345

# finger
./bin/twisted_finger07

telnet localhost 1079
# ctrl + ] 呼出telnet命令行
```



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



#### python

* echo_*.py: muduo书 chpt6 6.6

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

https://github.com/huangrt01/tinyflow

* [MXNet专栏 | 陈天奇：NNVM打造模块化深度学习系统](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650719529&idx=3&sn=6992a6067c79349583762cb28eecda89)
* [Build your own TensorFlow with NNVM and Torch](https://www.r-bloggers.com/2016/09/build-your-own-tensorflow-with-nnvm-and-torch/)

* [评价 nnvm 和 tf 的文章](https://www.zhihu.com/question/51216952/answer/124708405)
  * 正方（chentianqi）：直接讨论一下设计，目前TF采取了单一的动态执行模式，使得本身执行特别依赖于动态内存分配以及threading。而这并非是大部分场景下的最优方案。大部分场景下基于对于有限的图进行的静态分配，可以更大的缓解这个问题，实际情况如MX本身的[内存损耗](https://www.zhihu.com/search?q=内存损耗&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A124740328})可以做的更好。为什么目前TF不会出现多种执行模式呢，是因为TF本身Op的接口还是过于一般地针对的动态，而如果要更好的优化需要更细化的Op接口（分开内存分配和计算的部分），这就考虑到一个Op甚至可能有多种接口的可能性。
  * 反方（wangpengfei）：monolithic 的框架重构相对方便，会有更加旺盛的生命力。而 NNVM 的理想，恐怕跟现实还是有一定差距的。目前更有价值的，我觉得并不在图表示层，而是各种 Operator 的 kernels. 每个设备的 kernel 都需要专业人员定制，工作量大，难度高。cudnn 解决了 CUDA 设备上的大部分问题，但仍然有很多 Operator 需要自己实现。lowbit 目前也并没有特别可用的实现。如果能有一个统一的库，定义每个 Operator 在各种设备上的最优运行代码，应该对社区更有帮助。
  * 补充（lv-yafei）：在MxNet中，对于变长lstm和attention等网络，图的构建和销毁开销还是比较大的，虽然nnvm优化了建图的时间，但是还是无法做到可以被忽略不计，nnvm以后是否会提供类似于tensorflow的动态流图的构建。对于NLP等任务，动态流图可能无法做到显存最优，但是却可以避免反复构建图的开销。
    * Response(chentianqi)：未来考虑子图结构组合吧，这样子图可以避免反复拷贝构建



#### 代码结构

#### examples/autodiff-graph-executor-with-tvm

* ```python
  def softmax(x):
      x = x - np.max(x, axis=1, keepdims=True) # 防溢出
      x = np.exp(x)
      x = x / np.sum(x, axis=1, keepdims=True)
      return x
    
  def sum_node_list(node_list):
      """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
      from operator import add
      from functools import reduce
      return reduce(add, node_list)
  ```

* Autodiff
  * ad.gradients() 是在构图，倒序遍历topology
    * node.op.gradient(self, node, output_grad): op级别的后向，输出一个新op
  * Executor.run() 是在运行图，顺序遍历topology
    * node.op.compute(self, node, input_vals): op级别的前向，输出value
  * test: 支持grad_of_grad
  
* Graph_executor_with_tvm

```shell
export PYTHONPATH="${PYTHONPATH}:/path/to/python"
pip3 install --upgrade pip
pip3 install apache-tvm

nosetests -v tests/test_tvm_op.py --nocapture [--match=test_softmax_cross_entropy]

# see cmd options with 
# python tests/mnist_dlsys.py -h

# run logistic regression on numpy
python tests/mnist_dlsys.py -l -m logreg
# run MLP on numpy
python tests/mnist_dlsys.py -l -m mlp
```

### tvm

![tvm-arch](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/code-reading/tvm-arch.png)

* 参考资料：[从零开始学习深度学习编译器](http://giantpandacv.com/project/%E9%83%A8%E7%BD%B2%E4%BC%98%E5%8C%96/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%BC%96%E8%AF%91%E5%99%A8/TVM%20%E5%AD%A6%E4%B9%A0%E6%8C%87%E5%8D%97/)
  * 这个资料好全，深坑。。。 列为TODO吧

* 内存管理

  * executor结构简单，Grab saved optimization plan from graph

    * ```c++
      // apps/bundle_deploy.c
      TVM_DLL void tvm_runtime_get_output(void* executor, int32_t index, DLTensor* tensor) {
        TVMGraphExecutor* graph_executor = (TVMGraphExecutor*)executor;
        TVMGraphExecutor_GetOutput(graph_executor, index, tensor);
      }
      
      // src/runtime/ctr/graph_executor.c
      struct TVMGraphExecutor
      int TVMGraphExecutor_SetupStorage()
      ```

    * `attrs->storage_id` 
    * ndarray的view接口

  * 内存管理算法：`src/relay/backend/graph_plan_memory.cc`

  * graph_executor_codegen 调用 GraphPlanMemory

    * 核心是对图按toposort遍历，visit op，给每块内存分配storage id。visit时调用StorageAllocator (只维护每个storage的元信息，不实际管理内存)，先 CreateToken 再CheckForRelease

    * class StorageAllocaBaseVisitor : public transform::DeviceAwareExprVisitor

      * `DeviceAwareVisitExpr_` 输入是 FunctionNode，不处理 sub functions、primitive functions
      * `std::unordered_map<const ExprNode*, std::vector<StorageToken*>> token_map_;`
      * GetToken 方法
      * `virtual void CreateTokenOnDevice(const ExprNode* op, const VirtualDevice& virtual_device, bool can_realloc) = 0;`

    * class StorageAllocaInit : protected StorageAllocaBaseVisitor

      * `DeviceAwareVisitExpr_`: create token for the call node, and for each input, visit argument token.

    * class StorageAllocator : public StorageAllocaBaseVisitor

      * `prototype_ = StorageAllocaInit(&arena_).GetInitTokenMap(func);`
      * `StorageInfo(std::vector<int64_t> storage_ids, std::vector<VirtualDevice> virtual_devices, std::vector<int64_t> storage_sizes_in_bytes);`
      * DeviceAwareVisitExpr_(CallNode) final

    * ```c++
      class StorageAllocator {
        // allocator
        support::Arena arena_;
        /*! \brief internal prototype token map */
        std::unordered_map<const ExprNode*, std::vector<StorageToken*>> prototype_;
        /*! \brief token allocator for optimizing 1d and 2d token alloc requests */
        TokenAllocator allocator_;
      }
      
      class TokenAllocator2D {
      	std::unordered_map<int64_t, MemBlock> blocks_;
        std::unordered_set<int64_t> free_list_;
      }
      
      class TokenAllocator1D {
        // scale used for rough match
        const size_t match_range_{16};
        // free list of storage entry
        std::multimap<size_t, StorageToken*> free_;
        // all the storage resources available
        std::vector<StorageToken*> data_;
      }
      ```

* 内存管理相关RFC

  * https://discuss.tvm.apache.org/t/rfc-unified-static-memory-planning/10099
  * [issue: symbolic shape runtime](https://github.com/apache/tvm/issues/2451)
  * [rfc: relay dynamic runtime](https://github.com/apache/tvm/issues/2810), converged in VM design

* tests
  * tests/python/relay/test_backend_graph_executor.py: test_plan_2d_memory()
* tvm and tf: [TVMDSOOp RFC](https://discuss.tvm.apache.org/t/add-the-document-for-tvmdsoop/6622), [PR](https://github.com/apache/tvm/pull/4459/files)
  * src/contrib/tf_op/tvm_dso_op_kernels.cc
    * 如果不align到64，需要 `EnsureAlignment` 分配内存然后 `input.CopyFromOrigin()` 做memcpy



#### tvm turorial

##### How To Guides

* [Work With Tensor Expression and Schedules](https://tvm.apache.org/docs/how_to/work_with_schedules/index.html)

  * [reduction](https://tvm.apache.org/docs/how_to/work_with_schedules/reduction.html)

  * cross thread reduction

  * `Reductions are only allowed at the top level of compute. Please create another tensor for further composition.`

  * ```python
    n = te.var("n")
    m = te.var("m")
    A = te.placeholder((n, m), name="A")
    k = te.reduce_axis((0, m), "k")
    B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name="B")
    
    s = te.create_schedule(B.op)
    ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
    
    # Reduction Factoring and Parallelization
    BF = s.rfactor(B, ki)
    
    xo, xi = s[B].split(B.op.axis[0], factor=32)
    s[B].bind(xo, te.thread_axis("blockIdx.x"))
    s[B].bind(xi, te.thread_axis("threadIdx.x"))
    print(tvm.lower(s, [A, B], simple_mode=True))
    ```

* [Optimize Tensor Operators](https://tvm.apache.org/docs/how_to/optimize_operators/index.html)
  * [optimize conv2d](https://tvm.apache.org/docs/how_to/optimize_operators/opt_conv_cuda.html)
  * [optimize using TensorCores](https://tvm.apache.org/docs/how_to/optimize_operators/opt_conv_tensorcore.html)

##### User Tutorial

* Working with Operators Using Tensor Expression

  * We can do more specializations. For example, we can write `n = tvm.runtime.convert(1024)` instead of `n = te.var("n")`, in the computation declaration. The generated function will only take vectors with length 1024.

  * Example 2: Manually Optimizing Matrix Multiplication with TExample 2: Manually Optimizing Matrix Multiplication with TE

    * ```python
      bn = 32
      
      # Blocking by loop tiling
      xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
      (k,) = s[C].op.reduce_axis
      ko, ki = s[C].split(k, factor=4)
      
      # Hoist reduction domain outside the blocking loop
      s[C].reorder(xo, yo, ko, ki, xi, yi)
      ```

    * Make Iterations Row-friendly: `s[C].reorder(xo, yo, ko, xi, ki, yi)`

      * `ki` 是reduce axis的切片，对每列`xi`的切片运算是行级别的

    * Array Packing

      * By reordering a `[16][16]` array to a `[16/4][16][4]` array the access pattern of B will be sequential when grabbing the corresponding value from the packed array.
      * ![array-packing](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/code-reading/array-packing.png)

* [Optimizing Operators with Schedule Templates and AutoTVM](https://tvm.apache.org/docs/tutorial/autotvm_matmul_x86.html#install-dependencies)

* testing
  * `np.testing.assert_allclose()`



### 



### tensorflow

见【tensorflow】笔记



### TODO

这有一系列关于hash_map精巧设计的文章 https://preshing.com/20160222/a-resizable-concurrent-map/

配合经典的tbb实现阅读：https://oneapi-src.github.io/oneTBB/main/tbb_userguide/concurrent_hash_map.html

https://github.com/oneapi-src/oneTBB/blob/master/include/oneapi/tbb/concurrent_hash_map.h







读ulib的solaris实现的avl树

https://github.com/huangrt01/ulib

perf/avl

Second (and perhaps counterintuitively), one can achieve concurrency and composability by having no locks whatsoever. In this case, there must be no global subsystem state—subsystem state must be captured in per-instance state, and it must be up to consumers of the subsystem to assure that they do not access their instance in parallel. By leaving locking up to the client of the subsystem, the subsystem itself can be used concurrently by different subsystems and in different contexts

* A concrete example of this is the AVL tree implementation used extensively in the Solaris kernel. As with any balanced binary tree, the implementation is sufficiently complex to merit componentization, but by not having any global state, the implementation may be used concurrently by disjoint subsystems—the only constraint is that manipulation of a single AVL tree instance must be serialized.
