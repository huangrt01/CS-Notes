[toc]
### CS144-Lab-Computer-Networking
* 我的Lab仓库：https://github.com/huangrt01/sponge-CS144-Lab
* [CS144课程网站（包括pdf、project）](https://cs144.github.io/)
* [CS144: Lab FAQs](https://cs144.github.io/lab_faq.html)
* [Sponge: Class Hierarchy]()


- [x] Lab 0: networking warmup

- [x] Lab 1: stitching substrings into a byte stream

- [ ] Lab 2: the TCP receiver

- [ ] Lab 3: the TCP sender

- [ ] Lab 4: the TCP connection
- [ ] Lab 5: the network interface
- [ ] Lab 6: the IP router  

#### Lab结构
* In Lab 1, you’ll implement astream reassembler—a module that stitches small piecesof the byte stream (known as substrings, or segments) back into a contiguous stream of bytes in the correct sequence.
* In Lab 2, you’ll implement the part of TCP that handles the inbound byte-stream:  the **TCPReceiver**.  This involves thinking about how TCP will represent each byte’s place in the stream—known as a “sequence number.”  The **TCPReceiver** is responsible for telling the sender (a) how much of the inbound byte stream it’s been able to assemble successfully (this is called “acknowledgment”) and (b) how many more bytes the sender is allowed to send right now (“flow control”).
* In Lab 3, you’ll implement the part of TCP that handles the outbound byte-stream:  the **TCPSender**.  How should the sender react when it suspects that a segment it transmitted was lost along the way and never made it to the receiver?  When should it try again and re-transmit a lost segment?
* In Lab 4,  you’ll combine your work from the previous to labs to create a working TCP implementation:  a **TCPConnection** that contains a **TCPSender** and **TCPReceiver**.You’ll use this to talk to real servers around the world.

#### Lab0: networking warmup
##### 1.配环境
设虚拟机，[实验指导书](https://stanford.edu/class/cs144/vm_howto/vm-howto-image.html#connect) ，可参考[Shell笔记](https://github.com/huangrt01/Markdown-Transformer-and-Uploader/blob/master/Notes/Output/Shell-MIT-6-NULL.md)，迁移dotfiles

```shell
sudo apt-get update
sudo apt install zsh
zsh --version

# log out and login back
echo $SHELL
$SHELL --version

sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

source ~/.zshrc

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

#### lab1:stitching substrings into a byte stream
##### 3.Putting substrings in sequence
* 用set保存(index, data)数据，便于查找，可以用`lower_bound`查找，代码视线中省略了，采用的顺序遍历
* `push_substring`的bytes接收范围图
  * <img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/master/Notes/Computer-Networking-Lab-CS144-Stanford/reassembler.png" alt="reassembler" style="zoom:100%;" />


#### 工程细节
* [注意迭代器的使用](https://www.cnblogs.com/blueoverflow/p/4923523.html)
  * `container.erase(iter++)`, 同时完成删除和迭代
* 如果iterator重复erase，可能在初始化string时发生未知的seg fault
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
