[toc]
### CS144-Lab-Computer-Networking

**写在前面**

在历史的伟力面前，个人的命运是不可捉摸的。学生生涯结束地比想象中快，下个月就要正式入职字节跳动了。回顾本科期间，做过的大作业不少，却大多是期中对着一页不明就里的薄纸发呆，期末临近deadline，东抄抄西补补，勉强弄个不忍直视的半成品，没有时间也没有能力完成一次高质量的大作业。打算在入职前至少做一个Lab，之所以选择stanford的CS144，一方面是因为这门课质量很高，b站有配套的视频，Lab也在这两年做了大的改进，改为了一个优雅的TCP实现，全部资料都开源在[课程网站](https://cs144.github.io/)上，适合自学。另一方面，我本科期间没有学过计算机网络，这次补课也有一举两得的意味。

做下来感觉不错，课程的老师和助教很用心：说明文档十页长，FAQ覆盖了作业中会遇到的方方面面的问题；大作业的单元测试的代码量是模块代码的几倍，倾注了助教的心血。如果没有这些模块架构和单元测试，对于初学者来说是很难完成即使是初级的TCP协议栈编写，充分利用这些资源，站在巨人的肩膀上，能力能得到更好的锻炼。

以下是一些课程资源：

* 我的Lab仓库：https://github.com/huangrt01/sponge-CS144-Lab
* [CS144课程网站（including pdf, project and FAQs）](https://cs144.github.io/)
* [大课视频](https://www.bilibili.com/video/BV1wt41167iN?from=search&seid=12807244912122184980)

- [x] Lab 0: networking warmup

- [x] Lab 1: stitching substrings into a byte stream

- [x] Lab 2: the TCP receiver

- [x] Lab 3: the TCP sender

- [x] Lab 4: the TCP connection
- [ ] Lab 5: the network interface
- [ ] Lab 6: the IP router  

#### Lab0: networking warmup
##### 1.配环境
设虚拟机，[实验指导书](https://stanford.edu/class/cs144/vm_howto/vm-howto-image.html#connect) ，可参考[我的Shell笔记](https://github.com/huangrt01/Markdown-Transformer-and-Uploader/blob/master/Notes/Output/Shell-MIT-6-NULL.md)，迁移[dotfiles](https://github.com/huangrt01/dotfiles)

```shell
sudo apt-get update
...
cd sponge/build
rm CMakeCache.txt
CLANG_TIDY=clang-tidy-6.0 CXX=clang++-6.0 cmake .. -DCMAKE_BUILD_TYPE=Debug
```




##### 2.Networking by Hand
2.1 Fetch a Web page

```shell
telnet cs144.keithw.org http
GET /hello HTTP/1.1 # path part,第三个slash后面的部分
Host: cs144.keithw.org # host part,`https://`和第三个slash之间的部分
```
* 返回的有[ETag](https://www.cnblogs.com/happy4java/p/11206015.html), 减少服务器带宽压力

```
HTTP/1.1 200 OK
Date: Sat, 23 May 2020 12:00:46 GMT
Server: Apache
X-You-Said-Your-SunetID-Was: huangrt01
X-Your-Code-Is: 582393
Content-length: 113
Vary: Accept-Encoding
Content-Type: text/plain

Hello! You told us that your SUNet ID was "huangrt01". Please see the HTTP headers (above) for your secret code.
```

```shell
netcat -v -l -p 9090
telnet localhost 9090
```

##### 3.Writing a network program using an OS stream socket
OS stream socket： ability to create areliable bidirectional in-order byte stream between two programs
* turn “best-effort datagrams” (the abstraction the Internet provides) into“reliable byte streams” (the abstraction that applications usually want)

3.1 Build

3.2 Modern C++: mostly safe but still fast and low-level

* 读文档：https://cs144.github.io/doc/lab0/inherits.html
  * a Socket is a type of FileDescriptor, and a TCPSocket is a type of Socket.

```c++
//! \name
//! An FDWrapper cannot be copied or moved
//!@{
FDWrapper(const FDWrapper &other) = delete;
FDWrapper &operator=(const FDWrapper &other) = delete;
FDWrapper(FDWrapper &&other) = delete;
FDWrapper &operator=(FDWrapper &&other) = delete;
//!@}
```

3.4 `webget()`
* SHUT_RD/WR/RDWR，先用SHUT_WR关闭写，避免服务器等待
```c++
void get_URL(const string &host, const string &path) {
    TCPSocket sock{};
    sock.connect(Address(host,"http"));
    string input("GET "+path+" HTTP/1.1\r\nHost: "+host+"\r\n\r\n");
    sock.write(input);
    // cout<<input;
    // If you don’t shut down your outgoing byte stream,
    // the server will wait around for a while for you to send
    // additional requests and won’t end its outgoing byte stream either.
    sock.shutdown(SHUT_WR);
    while(!sock.eof())
        cout<<sock.read();  
    sock.close();
}
```

3.5 An in-memory reliable byte stream
* 数据结构deque，注意eof的判断条件即可

#### lab1: stitching substrings into a byte stream
##### 3.Putting substrings in sequence
* assemble数据时，为了简化代码流程，先将其和可能的字段合并，再判断是否可以write，因此需要设计一个merge函数
* 注意`end_input()`的判断条件
* 用set保存(index, data)数据，可以用`lower_bound`查找
* 细节：`push_substring`的bytes接收范围图
<img src="Computer-Networking-Lab-CS144-Stanford/reassembler.png" alt="reassembler" style="zoom:100%;" />

#### lab2: the TCP receiver
##### 3.1 Sequence Numbers

<img src="Computer-Networking-Lab-CS144-Stanford/001.jpg" alt="different index" style="zoom:100%;" />

* 利用头文件中的函数简化代码
* 计算出相对checkpoint的偏移量之后，再转化成离checkpoint最近的点，如果加多了就左移，注意返回值太小无法左移的情形。

```c++
uint64_t unwrap(WrappingInt32 n, WrappingInt32 isn, uint64_t checkpoint) {
    uint32_t offset = n - wrap(checkpoint, isn);
    uint64_t ret = checkpoint + offset;
    // 取距离checkpoint最近的值，因此判断的情况是否左移ret
    //注意位置不够左移的情形！！！
    if (offset >= (1u << 31) && ret >= UINT32_LEN)
        ret -= (1ul << 32);
    return ret;
}
```
##### 3.2 window
* lower: ackno     
* higher~window size
* window size = capacity - ByteStream.buffer_size()

##### 3.3 TCP receiver的实现
1. receive segmentsfrom its peer
2. reassemble the ByteStream using your StreamReassembler, and calculate the 
3. acknowledgment number (ackno) 
4. and the window size.

* `_reassembler`忽视SYN，所以要手动对index减1、ackno()加1
* 非常规路线的处理：比如对于第二个SYN或者FIN信号，接收机选择忽视，具体见`bool TCPReceiver::segment_received(const TCPSegment &seg)`的实现

#### lab3: the TCP sender
##### 3.1 重传时机

sponge网络库的设计，TCP的测试中利用到状态判断，但具体到sender、receiver这几个类的设计时，对类的设计是面向对象，对类内部的函数是面向过程，而不像[Linux内核的tcp实现](https://github.com/torvalds/linux/blob/cb8e59cc87201af93dfbb6c3dccc8fcad72a09c2/net/ipv4/tcp.c)中有利用`goto`语句来模拟有限状态机。因此，在实现这个Lab的函数的时候，依然要以面向过程的思路，理解sender和receiver在不同的情景下会如何工作，使函数在内部看来是一个过程，外界测试时又能完美的体现状态变化。比如三次握手和四次挥手就是一个很好的例子
* sender发送new segments(包含SYN/FIN)，用`_segments_out`这个queue跟踪，影响它的因素是ackno
* 重传条件是"outstanding for too long", 受tick影响，tick仅由外部的类调用，sender内部不调用任何时间相关的函数
* retransmission timeout(RTO)，具体实现是RFC6298的简化版
  * 重传连续的之后double ；收到ackno后重置到`_initial_RTO`
  * 可参考[RFC 6298](https://datatracker.ietf.org/doc/rfc6298/?include_text=1)第5小节实现_timer
* 注意读`/lib_sponge/tcp_helper/tcp_state.cc`帮助理解状态变化



<img src="Computer-Networking-Lab-CS144-Stanford/receiver.jpg" alt="receiver" style="zoom:100%;" />



<img src="Computer-Networking-Lab-CS144-Stanford/sender.jpg" alt="sender" style="zoom:100%;" />

#### lab4: the summit (TCP in full)

这次Lab是把之前的receiver和sender封装成TCPConnection类，用来进行真实世界的通信，下图有助于直观理解结构。

<img src="Computer-Networking-Lab-CS144-Stanford/dataflow.jpg" alt="TCP dataflow" style="zoom:100%;" />

<img src="Computer-Networking-Lab-CS144-Stanford/header.jpg" alt="TCP header" style="zoom:80%;" />



实验的测试文件一如既往的重要，命名规则如下：

* “c” means your code is the client (peer that sends the first syn)
* “s” means your code is the server.  
* “u” means it is testing TCP-over-UDP
* “i” is testing TCP-over-IP(TCP/IP). 
* “n” means it is trying to interoperate with Linux’s TCP implementation
* “S” means your code is sending data
* “R” means your code is receiving data
* “D” means data is being sent in bothdirections
* lowercase “l” means there is packet loss on the receiving (incoming segment) direction
* uppercase “L” means there is packet loss on the sending (outgoing segment) direction.

**实现时的重要细节：** 

1.需要单独讨论重传ACK的情形，在我的实现中我写了一个`send_ack_back()`函数

**In `TCPConnection::segment_received`, what are the three conditions in which the TCPConnection needs     to make sure that the segment receives at least one ACK segment in reply, and may need to force the TCPSender to spit out an empty segment to make this happen?**

* If the incoming segment occupies any sequence numbers       (`length_in_sequence_space() > 0`) 	    
* If the `TCPReceiver` thinks the segment is unacceptable (`TCPReceiver::segment_received()` returns `false`) 	    
* If the `TCPSender` thinks the ackno is invalid (`TCPSender::ack_received()` returns `false`)   

2.处理RST
* 如果收到RST，需要给sender和receiver的stream用set_error()，不需要回传
* 发送RST的情形
  * 错误的connect，connect()函数中  
  * unclean shutdown，析构函数中
  * 连续重传超次数 `_sender.consecutive_retransmissions() > TCPConfig::MAX_RETX_ATTEMPTS`

3.判断终结条件
* 具体实现：见代码以及实验指导书的第5节。
* 背后的理念：Because of the [Two Generals Problem](https://en.wikipedia.org/wiki/Two_Generals'_Problem), it’s impossible to guarantee that both peers can achieve a clean shutdown

**Debug**

1.最后和linux系统真实地进行通信，总有10个tests过不了，打算先研究这一个测试样例: `../txrx.sh -isDnd 128K -w 8K -l 0.1`

2.Github上找到一个顺利过关的[印度大哥](https://github.com/gcidart/cs144)，他给了一些[debug建议](https://github.com/gcidart/cs144/issues/1)

3.改了一些tcp_connection和tcp_sender的细节，benchmark速度提升

4.debug无果，决定替换印度大哥模块逐一排查，发现是tcp_receiver出了问题，

4.最终的bug非常坑，即使是助教写的测试样例也无法照顾到，只有当和linux的tcp进行真实丢包通信时才会出现，具体是在以下这一行tcp滑窗控制: 

`bool inbound =  (seq_start>=win_start&& seq_start<=win_end) ||  (payload_end>=win_start && seq_end<=win_end);`

receive的条件是数据片段和receiver的窗有重合，这个实现看似很简单，只需要写出数据的begin和end，窗的begin和end，然后做判断。具体来说，判断数据的begin是否在窗里，再判断数据的end是否在窗里即可。但在TCPReceiver的具体实现中，在判断数据的end是否在窗里时，需要用两个普通的end：代码中的`payload_end`是原先的`seq_end`经过处理得到的，需要把syn和fin占的位置排除掉，这样处理是因为receiver内部的`_reassembler`类只处理data，不处理syn和fin。

综合来说，需要对基础的窗的判断条件做修改，具体实现中不是一个uniform的形式，只有这样才能过最后几个tests。



#### 工程细节

* [注意迭代器的使用](https://www.cnblogs.com/blueoverflow/p/4923523.html)
  * `container.erase(iter++)`, 同时完成删除和迭代
  * 如果iterator重复erase，可能导致seg fault
* 单元测试
  * 用generate生成随机数据
```c++
auto rd = get_random_generator();
const size_t size = 1024;
string d(size, 0);
generate(d.begin(), d.end(), [&] { return rd(); });
```

* 类cmp函数的定义，适用 `lower_bound()`方法
  * 两个**const**都不能掉！
```c++
class typeUnassembled {
  public:
    size_t index;
    std::string data;
    typeUnassembled(size_t _index, std::string _data) : index(_index), data(_data) {}
    bool operator<(const typeUnassembled &t1) const { return index < t1.index; }
};
```
* `urg = static_cast<bool>(fl_b & 0b0010'0000); // binary literals and ' digit separator since C++14!!!`

