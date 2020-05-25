### Stanford CS144
* [CS144视频（b站）](https://www.bilibili.com/video/BV1wt41167iN?from=search&seid=12807244912122184980)
* [CS144课程网站（包括pdf、project）](https://github.com/CS144)

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
  * Best-effort attempt
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
* 有时需要break layering: 比如Linux内核的部分代码C语言直接用汇编 => code不再layer-independent
  
  * a continual tension to improve the Internet by making cross-layer optimizations and the resulting loss of flexibility. e.g. NATs=>很难加其它类型的传输层
  * epoll这个接口是linux独有的，FreeBSD里是kqueue
  
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
  * reply：原则上unicast，只回传给发送者=>实际更常见broadcast
  * No "sharing" of state: bad state will die eventually
  * MacOS保留20min
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









##### potpourri
* [RFC 2606](https://datatracker.ietf.org/doc/rfc2606/): localhost









