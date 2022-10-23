muduo 是一个基于非阻塞 IO 和事件驱动的现代 C++ 网络库，原生支持 one loop per thread 这种 IO 模型。muduo 适合开发 Linux 下的面向业务的多线程服务 端网络应用程序

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

* Note:
  * 临界区在 Windows 上是 struct CRITICAL_SECTION，是可重入的;在 Linux 下是 pthread_mutex_t，默认是不可重入的
  * 在 Java 中，一个 reference 只要不为 null，它一定指向有效的对象
  * 如果这几种智能指针是对象 x 的数据成员，而它的模板参数 T 是个 incomplete 类型，那么 x 的析构函数不能是默认的或内联的，必须在 .cpp 文件里边 显式定义，否则会有编译错或运行错(原因见 §10.3.2)
  * 智能指针参考boost库学习：https://www.boost.org/doc/libs/1_80_0/libs/smart_ptr/doc/html/smart_ptr.html#introduction
    * [handle-body idiom](https://www.cs.vu.nl/~eliens/online/tutorials/objects/patterns/handle.html) 可以 [用 scoped_ptr 或 shared_ptr 来做](https://www.boost.org/doc/libs/1_80_0/libs/smart_ptr/example/scoped_ptr_example.hpp)
    * shared_ptr
      * 有一些atomic相关的member function
      * 为什么统一shared_ptr的实现：否则a reference counted pointer (used by library A) cannot share ownership with a linked pointer (used by library B.)
  * [function/bind的救赎（上） by 孟岩](https://blog.csdn.net/myan/article/details/5928531)
    * 过程范式、函数范式、对象范式
    * 对象范式的两个基本观念：
      - 程序是由**对象**组成的；
      - 对象之间互相**发送消息**，协作完成任务；
      - 请注意，这两个观念与后来我们熟知的面向对象三要素“封装、继承、多态”根本不在一个层面上，倒是与再后来的“组件、接口”神合
    * C++的静态消息机制：不易开发windows这种动态消息场景；“面向类的设计”过于抽象
    * Java和.NET中分别对C++最大的问题——缺少对象级别的delegate机制做出了自己的回应



#### chpt 2 线程同步精要

