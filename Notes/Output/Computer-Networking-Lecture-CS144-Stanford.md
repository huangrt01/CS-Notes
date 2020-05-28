### Stanford CS144
* [CS144视频（b站）](https://www.bilibili.com/video/BV1wt41167iN?from=search&seid=12807244912122184980)
* [CS144课程网站（包括Pdf、Lab）](https://cs144.github.io/)
* [我的CS144 Lab分析]()(https://github.com/huangrt01/CS-Notes/blob/master/Notes/Output/Computer-Networking-Lab-CS144-Stanford.md)

##### 1-0 The Internet and IP Introduction
* internet layer: Internet Protocol, IP address, packet's path
* 彩蛋：世一大惺惺相惜
<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/master/Notes/Computer-Networking-Lecture-CS144-Stanford/005.jpg" alt="Stanford-THU" style="zoom:60%;" />

* 用`ping`和`traceroute`看IP地址; 光纤2/3光速，8637km - RTT=86ms
##### 1-1 A day in the life of an application
* Networked Applications: connectivity, bidirectional and reliable data stream
* Byte Stream Model: A - Internet - B, server和A、B均可中断连接
* World Wide Web(HTTP: HyperText Transfer Protocol)
  * request: GET, PUT, DELETE, INFO, 400(bad request) 
  * GET - response(200, OK)  , 200代表有效
  * document-centric: "GET/HTTP/1.1", "HTTP/1.1 200 OK \<contents of the index.html\>"
* BitTorrent: peer-to-peer model
  * breaks files into "pieces" and the clients join and leave "swarms" of clients
  * 先下载torrent file -- tracker存储lists of other clients
  * dynamically exchange data
* Skype: proprietary system, a mixed system
  * two clients： A -- (Internet + Rendezvous server) -- NAT -- B
  
  * NAT(Network Address Translator): 连接的单向性，使得A只能通过Rendezvous server询问B是否直连A =>reverse connection
  
  * Rendezvous server
  
  * 如果模式是A -- NAT-- (Internet + Rendezvous server) -- NAT -- B，Skype用Relay来间接传递信息
  

<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/master/Notes/Computer-Networking-Lecture-CS144-Stanford/003.jpg" alt="Lego TCP/IP" style="zoom:60%;" />

##### 1-2 The four layer Internet model
<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/master/Notes/Computer-Networking-Lecture-CS144-Stanford/001.jpg" alt="4-layer" style="zoom:60%;" />

* 4 layer: 利于reuse
* Internet: end-hosts, links and routers
  * Link Layer: 利用link在end host和router或router和router之间传输数据, hop-by-hop逐跳转发; e.g. Ethernet and WiFi
  * Network Layer: datagrams, Packet: (Data, Header(from, to))
    * packets可能失去/损坏/复制，no guarantees
    * must use the IP
    * may be out of order
  * Transport Layer: TCP(Transmission Control Protocol)负责上述Network的局限性，controls congestion
    * sequence number -> 保序
    * ACK(acknowledgement of receipt)，如果发信人没收到就resend
    * 比如视频传输不需要TCP，可以用UDP(User Datagram Protocol),不保证传输
* Application Layer
  
* two extra things
  * IP is the "thin waist"   ,这一层的选择最少
  * the 7-layer OSI Model

<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/master/Notes/Computer-Networking-Lecture-CS144-Stanford/002.jpg" alt="7-layer" style="zoom:60%;" />

##### 1-3 The IP Service
* Link Frame (IP Datagram(IP Data(Data, Hdr), IP Hdr), Link Hdr )
* The IP Service Model的特点
  * Datagram: (Data, IP SA, IP DA)，每个router有forwarding table，类比为postal service中的letter
  * Unreliable: 失去/损坏/复制，保证只在必要的时候不可靠（比如queue congestion）
  * **Best-effort** attempt
  * Connectionless : no per-flow state, mis-sequenced
* IP设计简单的原因
  * minimal, faster, streamlined
  * end-to-end(在end points implement features)
  * build a variety of reliable/unreliable services on top
  * works over any link layer

* the IP Service Model
  1. tries to prevent packets looping forever (实现：在每个datagram的header加hop-count field: time to live TTL field, 比如从128开始decrement)
  2. will fragment packets if they're too long (e.g. Ethernet, 1500bytes)
  3. header checksum：增强可靠性
  4. allows for new versions of IP
  5. allows for new options to be added to header (由router处理新特性，慎重使用)

##### 1-4 A Day in the Life of a Packet
* 3-way handshake
  1. client: SYN 
  2. server: SYN/ACK
  3. client: ACK
* IP packets
  * IP address + TCP port(web server通常是80)
  * hops, Routers: wireless access point (WiFi的第一次hop)
  * forwarding table
  * default router

* wireshark: 
  * [谈谈Linux中的TCP重传抓包分析](https://segmentfault.com/a/1190000019734707)

##### 1-5 Principle: Packet switching principle
* packet: self-contained
* packet switching: independently for each arriving packet, pick its outgoing link. If the link is free, send it. Else hold the packet for later.
* source packet: (Data, (dest, C, B, A))  发展成只存destination，每个switch有table
* two consequences
  * simple packet forwarding: No per-flow state required，state不需要store/add/remove
  * efficient sharing of links: busty data traffic; statistical multiplexing => 对packet一视同仁，可共享links

##### 1-6 Principle: Layering

* 一种设计理念，layers are functional components, they communicate sequentially 
* edit -> compile -> link -> execute
  
  * compiler: self-contained, e.g. lexical analysis, parsing the code, preprocessing declarations, code generation and optimization
* 有时需要break layering
  * 比如Linux内核的部分代码C语言直接用汇编 => code不再layer-independent
  * a continual tension to improve the Internet by making cross-layer optimizations and the resulting loss of flexibility. e.g. NATs=>很难加其它类型的传输层
  * epoll这个接口是linux独有的，FreeBSD里是kqueue
  * UDP header的checksum计算用到IP header
  
* layering的原因：1.modularity 2.well defined service 3.reuse 4.separation of concerns 5.continuous improvement 6.p2p communications

##### 1-7 Principle: Encapsulation

* TCP segment is the **payload** of the IP packet. IP packet encapsulates the TCP segment.

* 一层层，套footer和header

    * 两种写法，底层的写法(switch design)header在右边，software的写法(protocol)header在左边（IETF）
    * VPN: (Eth, (IP, (TCP, (TLS, IP Packet))))，外层的TCP指向VPN gateway

##### 1-8 Byte Order
* 2^32 ~ 4GB ~  0x0100000000
* 1024=0x0400	大端：0x04 0x00；小端: 0x00 0x04. 
* Little endian: x86, big endian: ARM, network byte order
* e.g. `uint16_t http_port=80; if(packet->port==http_port){...}` IPv4的packet_length注意大小端
* 函数：`htons(),ntohs(),htonl(),ntohl()`
  * host/network, short/long
  * `#include<arpa/inet.h>`

##### 1-9 IPv4 addresses

goal:
* stitch many different networks together
* need network-independent, unique address

IPv4:
* layer 3 address
* 4 octets  a.b.c.d
* 子网掩码netmask: 255.128.0.0 前9位，1越少网络越大，same network不需要路由，直接link即可

<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/master/Notes/Computer-Networking-Lecture-CS144-Stanford/004.jpg" alt="IPv4 Datagram" style="zoom:60%;" />

IPv4 Datagram
* Total Packet Length: 大端，最多65535bytes, 1400 -> 0x0578
* Protocol ID: 6->TCP

Address Structure
* network+host
* class A,B,C: 0,7+24; 10, 14+16; 110, 21+8

Classless Inter-Domain Routing(CIDR，无类别域间路由)
* address block is a pair: address, count
* counts是2的次方? 表示netmask长度
* e.g. Stanford 5/16 blocks `5*2^(32-16)`
* 前缀聚合，防止路由表爆炸
* IANA(Internet Assigned Numbers Authority): give /8s to RIRs

##### 1-10 Longest Prefix Match(LPM)
* forwarding table: CIDR entries
  * LPM的前提是必须先match，再看prefix
  * default: 0.0.0.0/0

##### 1-11 Address Resolution Protocol(ARP)
* IP address(host) -> link address(Ethernet card, 48bits)
* Addressing Problem: 一个host对应多个IP地址，不容易对应
  * 解决方案：gateway两侧ip地址不同，link address确定card，network address确定host
  * 这有点历史遗留问题，ip和link address的机制没有完全地分离开，decoupled logically but coupled in practice
  * 对于A，ip的目标是B，link的目标是gateway

* ARP，地址解析协议：由IP得到MAC地址 => 进一步可得到gateway address
  * 是一种request-reply protocol
  * nodes cache mappings, cache entries expire
  * 节点request a link layer broadcast address，然后收到回复，回复的packet有redundant data，看到它的节点都能生成mapping
  * reply：原则上unicast，只回传给发送者=>实际实现时更常见broadcast
  * No "sharing" of state: bad state will die eventually
  * MacOS中保留20min
  * gratuitous request: 要求不存在的mapping，推销自己

<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/master/Notes/Computer-Networking-Lecture-CS144-Stanford/007.jpg" alt="ARP" style="zoom:60%;" />

e.g. 
* hardware:1(Ethernet)
* protocol: 0x0800(IP)
* hardware length:6 (48 bit Ethernet)
* protocol length:4(32 bit IP)
* opcode: 1(request) /2(reply)
* Destination: broadcast (ff:ff:ff:ff:ff:ff)

##### 1-12 recap

##### 1-13 SIP, Jon Peterson Interview
the intersection between technology and public policy
* IETF ( The Internet Engineering Task Force)
* ICANN（The Internet Corporation for Assigned Names and Numbers）

SIP（Session Initiation Protocol，会话初始协议）
* end-to-end的设计
* soft switching: 将呼叫控制功能从传输层分离
* PSTN ( Public Switched Telephone Network ) -> VOIP(Voice over Internet Protocol): telephony replacement

SIP的应用场景
* Skype内部协议转换成SIP
* VOIP, FiOS( a telecom service offered over fiber-optic lines)

现代技术
* SDN (Software Defined Network)
* I2RS(interface to the routing system)
* CDN(Content Delivery Network): 1.express coverage areas 2.advertise services that they provide, in order to allow collaboration or peering among CDNs => optimal selections of CDNs
* 识别robo calling

##### 2-0 Transport (intro)

* 关注TCP的correctness
* detect errors的三个算法：checksums, cyclic redundancy checks, message authentication codes
* TCP(Transmission Control Protocol)、UDP(User Datagram Protocol)、ICMP(Internet Control Message Protocol)

##### 2-1 The TCP Service Model
**The TCP Service Model**

* reliable, end-to-end, bi-directional,in-sequence, bytestream service
* Peer TCP layers communicate: connection
* congestion control

**过程**：三次握手和四次挥手

* 3-way handshake
  1. client: SYN, 送base number to identify bytes
  2. server: SYN/ACK, 也送base number
  3. client: ACK
* 传送TCP segment，最小可以1byte，比如在ssh session打字
* connection teardown
  1. client: FIN
  2. server: (Data +) ACK
  3. server: FIN
  4. client: ACK

**Techniques to manufacture reliability**

* Remedies
  * Sequence numbers: detect missing data
  * Acknowledgments: correct delivery
    * Acknowledgment (from receiver to sender)  
    * Timer and timeout (at sender)
    * Retransmission (by sender)
  * Checksums/MACs: detect corrupted data
    * Header checksum (IP)
    * Data checksum (UDP)
	* Window-based Flow-control: prevents overrunning receiver
  * FEC
  * Retransmission
  * Heartbeats
* Correlated failure
* TCP/DNS
* Paradox of airplanes



**The TCP Segment Format**
<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/master/Notes/Computer-Networking-Lecture-CS144-Stanford/008.jpg" alt="TCP header" style="zoom:60%;" />

* [IANA port number](https://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.xhtml): ssh 22, smtp 23, web 80
* source port: 初始化用不同的port避免冲突
* PSH flag: push，比如键盘敲击
* HLEN和(TCP options)联系

<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/master/Notes/Computer-Networking-Lecture-CS144-Stanford/009.jpg" alt="TCP uniqueness" style="zoom:60%;" />
* 五个部分，104bit
* 唯一性
  * 要求source port initiator每次increment:64k new connections
  * TCP picks ISN to avoid overlap with previous connection with same ID, 多一个域，增加随机性

##### 2-2 UDP service model
不需要可靠性：app自己控制重传，比如早期版本的NFS(network file system)

<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/master/Notes/Computer-Networking-Lecture-CS144-Stanford/010.jpg" alt="UDP header" style="zoom:60%;" />
* Checksum对于IPv4可选，可以为全0
* checksum用了IP header，违背layering principle，因为能detect错传
* UDP header有length字段，而TCP没有，因为TCP对空间要求高，用隐含的方式计算length
* port demultiplexing, connectionless, unreliable

应用
* DNS: domain name system，因为request全在单个datagram里
* DHCP: Dynamic Host Configuration Protocol
  * new host在join网络时得到IP
  * 连WiFi
* 对重传、拥塞控制、in-sequence delivery有special needs的应用，比如音频，但现在UDP不像以前用的那么多，因为很多是http，基于TCP。

##### 2-3 The Internet Control Message Protocol (ICMP) Service Model
* 用于report errors and diagnoise problems about network layer
* 网络层work的三个因素：IP、Routing Tables、ICMP

<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/master/Notes/Computer-Networking-Lecture-CS144-Stanford/011.jpg" alt="ICMP" style="zoom:40%;" />

* Message的意义见RFC 792
* 应用于ping：先发送8 0( echo request)，再送回0 0(echo reply)
* 应用于traceroute: 
  * 核心思想：连续发送TTL从1开始递增的UDP，期待回复的11 0(TTL expires)
  * 由于路由选择问题，traceroute 无法保证每次到同一个主机经过的路由都是相同的。
  * traceroute 发送的 UDP 数据报端口号是大于 30000 的。如果目的主机没有任何程序使用该端口，主机会产生一个3 3(端口不可达)ICMP报文给源主机。

##### 2-4 End-to-End Principle
**Why Doesn't the Network Help?**
* e.g.：压缩数据、Reformat/translate/improve requests、serve cached data、add security、migrate connections across the network
* end-to-end principle: function的正确完整实现只依赖于通信系统的end points

end-to-end check 
* e.g. File Transfer: link layer的error detection只检测transmission错误，不检测error storage
* e.g. TCP小概率会出错（stack）、BitTorrent
* wireless link相比wire link功能复杂，可靠性低，所以在link layer重传，可提升TCP性能
* RFC1958: "strong" end to end: 不推荐在middle实现任何功能，比如在link layer重传，假定了reliabilty的提升值得latency的牺牲

##### 2-5 Error Detection: 3 schemes: 3 schemes

* detect errors的三个算法：checksums, CRC(cyclic redundancy checks), MAC(message authentication codes)
  * append: ethernet CRC, TLS MAC
  * prepend: IP checksum

* Checksum (IP, TCP)
  * not very robust, 只能检1位错
  * fast and cheap even in software
  * IP, UDP, TCP use one's complement算法：16-bit word packet求和，进位加到底部，再取反码（特例：0xffff -> 0xffff，因为在TCP，checksum field为0意味着没有checksum）
* CRC: computes remainder of a polynomial (Ethernet)，见[通信与网络笔记](https://github.com/huangrt01/CS-Notes/blob/master/Notes/%E9%80%9A%E4%BF%A1%E4%B8%8E%E7%BD%91%E7%BB%9C.md)
  * 虽然more expensive，但支持硬件计算
  * 可对抗2 bits error、奇数error、小于c bits的突发错(burst)
  * 可incrementally计算
  * e.g. USB(CRC-16):  <img src="https://www.zhihu.com/equation?tex=%5Cbf%7BM%7D%20%3D%200x8005%20%3D%20x%5E%7B16%7D%2Bx%5E%7B15%7D%2Bx%5E2%2B1" alt="\bf{M} = 0x8005 = x^{16}+x^{15}+x^2+1" class="ee_img tr_noresize" eeimg="1"> ，对于generator需要给左边pad 1
* MAC: message authentication code: cryptographic transformation of data(TLS)
  * robust to malicious modifications, but not errors
  * 检错能力有局限，受随机性影响，不如CRC，no error detection guarantee
  *  <img src="https://www.zhihu.com/equation?tex=c%3DMAC%28M%2Cs%29" alt="c=MAC(M,s)" class="ee_img tr_noresize" eeimg="1"> ，M + c意味着对方有secret或者replay
  * 对于replay，`ctr++`, 具体见[我的密码学笔记](https://github.com/huangrt01/CS-Notes/blob/master/Notes/Output/Cryptography%20I%2C%20Stanford%20University%2C%20Coursera.md)的TLS部分
  
##### 2-6 Finite State Machines
<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/master/Notes/Computer-Networking-Lecture-CS144-Stanford/012.jpg" alt="HTTP Request" style="zoom:40%;" />

<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/master/Notes/Computer-Networking-Lecture-CS144-Stanford/013.jpg" alt="TCP Connection" style="zoom:100%;" />

* 非常规路线的处理：比如对于第二个SYN或者FIN信号，接收机选择忽视，具体见`bool TCPReceiver::segment_received(const TCPSegment &seg)`的实现

##### 2-7 Flow Control I: Stop-and-Wait
* 核心是receiver给sender反馈，让sender不要送太多packets
* 基本方法
  * stop and wait
  * sliding window

**stop and wait**
* flight中最多一个packet
* 针对ACK Delay（收到ACK的时间刚好在timeout之后）的情形，会有duplicates
  * 解决方案：用一个1-bit counter提供信息
  * assumptions：1）网络不产生重复packets；2）不delay multiple timeouts

<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/master/Notes/Computer-Networking-Lecture-CS144-Stanford/014.jpg" alt="stop-and-wait" style="zoom:100%;" />

##### 2-7 Flow Control II: Sliding Window
* Stop-and-Wait的性能：RTT=50ms, Bottleneck=10Mbps, Ethernet packet length=12Kb => 性能(2%)远远不到瓶颈
* Sliding Window计算Window size填满性能

**Sliding Window Sender**

* Every segment has a sequence number (SeqNo)
* Maintain 3 variables
  * Send window size(SWS)
  * Last acknowledgment(LAR)
  * Last segment sent(LSS)
* Maintain invariant:  <img src="https://www.zhihu.com/equation?tex=%28LSS%20-%20LAR%29%20%5Cleq%20SWS" alt="(LSS - LAR) \leq SWS" class="ee_img tr_noresize" eeimg="1"> 
* Advance LAR on new acknowledgement 
* Buffer up to SWS segments

**Sliding Window Receiver**
* Maintain 3 variables
  * Receive window size(RWS)
  * Last acceptable segment(LAS)
  * Last segment received(LSR)
* Maintain invariant:  <img src="https://www.zhihu.com/equation?tex=%28LAS%20-%20LSR%29%20%5Cleq%20RWS" alt="(LAS - LSR) \leq RWS" class="ee_img tr_noresize" eeimg="1"> 
* 如果收到的packet比LAS小，则发送ack
  * 发送cumulative acks: 收到1, 2, 3, 5，发送3
  * TCP acks are next expected data，因此要加一，上个例子改为4，初值为0

**RWS, SWS, and Sequence Space**
*  <img src="https://www.zhihu.com/equation?tex=RWS%20%5Cgeq%201%2C%20SWS%20%5Cgeq%201%2C%20RWS%20%5Cleq%20SWS" alt="RWS \geq 1, SWS \geq 1, RWS \leq SWS" class="ee_img tr_noresize" eeimg="1"> 
* if  <img src="https://www.zhihu.com/equation?tex=RWS%20%3D%201" alt="RWS = 1" class="ee_img tr_noresize" eeimg="1"> , "go back N" protocol ,need SWS+1 sequence numbers (需要多重传)
* if  <img src="https://www.zhihu.com/equation?tex=RWS%20%3D%20SWS" alt="RWS = SWS" class="ee_img tr_noresize" eeimg="1"> , need 2SWS sequence numbers
* 通常需要 <img src="https://www.zhihu.com/equation?tex=RWS%2BSWS" alt="RWS+SWS" class="ee_img tr_noresize" eeimg="1">  sequence numbers：考虑临界情况，RWS最左侧的ACK没有成功发送，重传后收到了RWS最右侧的ACK

**TCP Flow Control**

* Receiver advertises RWS using window field
* Sender can only send data up to LAR+SWS


















##### potpourri
* RFC 792: ICMP Message
* [RFC 1958](https://datatracker.ietf.org/doc/rfc1958/?include_text=1):Architectural Principles of the Internet
* [RFC 2606](https://datatracker.ietf.org/doc/rfc2606/): localhost
* [RFC 6335](https://tools.ietf.org/html/rfc6335): port number









