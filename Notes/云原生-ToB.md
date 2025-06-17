## 云原生 & ToB

[toc]

### Intro

* ToB的挑战
  * 试错成本太高。你不知道你做的东西是否有市场，跟To C不同，To C可以通过许多实际数据和指标了解应用的现状和前景。但做To B，特别是基础设施，需要花几个月甚至几年开发一个产品，之后再去市场验证，周期非常长，你很难快速试错。尤其在AI基础设施领域，用户需求的变化很大，不确定性高，试错成本也高。
* 国内ToB的挑战
  * 头部客户的用量实在是太大了，后面的腰部客户加起来还不如一个头部客户的量大。所以对云厂商来说，做一个产品把所有的腰部客户都服务好，也不如把一个头部客户服务好收益更大。

* 十万台是一个企业架设服务器或者使用云服务的一个成本临界点



### Literature Review

* 早在1956年，ChristopherStrachey提出了虚拟化的概念，这是云计算一开始的理论基础。
* 在2006年8月，云计算概念第一次在互联网一个大型会议提出，由此掀开了被称为“互联网的第三次革命”，于是各大科技巨头奋起，想要占领这块具有巨大市场潜力的“新大陆”。
* 亚马逊公司是现在全世界云服务市场份额最大的公司，亚马逊公司出于其网购占有率全世界第一的情况下，就有类似“黑色星期五”的疯狂购物节，此时平时分散时间购物的人们会集中在极短的时间内在亚马逊购物，由此亚马逊必须有足够的储存空间和运算能力来处理这种“突发情况”，在平时，通过把这些多余的储存空间和算力“出租”出去，就形成了一个云服务商的一个大优势。

### 各种概念

#### CNCF (Cloud Native Computing Foundation) 的定义

* 概念：在公有云、私有云和混合云等新型动态环境中，构建和运行可弹性扩展的应用。
* 技术：包括容器、服务网格、微服务、不可变基础设施和声明式 API。
* 效率
  * 基础设施标准化
  * 业务框架抽象化（无状态服务、离线训练任务）
  * 规范流程自动化
  * 交付形态一致化
* 成本
  * 秒级弹性
  * 按需分配
* 未来：containerd + kata + 裸金属



#### APM (Application Performance Management)

* [回到网易后开源 APM 技术选型与实战 - 李乐](https://www.infoq.cn/article/apm-pinpoint-practice)
  * 思想源自 Dapper
  * Pinpoint 实战
* [国内首个开源全链路压测平台 Takin](https://cloud.tencent.com/developer/article/1852614)
* [Oracle APM](https://www.oracle.com/manageability/application-performance-monitoring/)

##### 《Studying the Effectiveness of Application Performance Management (APM) Tools for Detecting Performance Regressions for Web Applications: An Experience Report》

APM tools处理regression test的缺点：1）mining approaches少；2）定位问题慢，需要manual work；3）难拓展

detect performance anomalies:

* Baseline-based Approach
* Threshold-based Approach
  * Percentage deviation threshold
  * Standard deviation threshold
  * Fixed threshold

Case Study Setup:

* simple mining approaches 是否能满足基础的监测功能
* injected regressions
  * Excessive Memory Usage Regressions: 1) unreleased resources and 2) inefficient use of streams.
  * High CPU Utilization Regressions
  * Inefficient Database Use Regressions: 1) Excessive Data and 2) One-by-one Processing

6.Discussion

6.1 More Mining Approaches are Needed for Reducing Manual Effort

6.2 APM Tools should Provide Better Data Aggregation and Summarization





#### "十万台的魔咒"

* [The Cost of Cloud, a Trillion Dollar Paradox](https://a16z.com/2021/05/27/cost-of-cloud-paradox-market-cap-cloud-lifecycle-scale-growth-repatriation-optimization/)



#### 云的应用

* 云+体验：内容存储分发加工推荐创作
  * 内容获取
  * 内容加工
  * 内容分发
  * 内容消费
  * 数据分析
* 云+数据
  * 数据驱动业务优化->分析评估->业务过程数字化->数据生产沉淀
  * 要解决的：高密度计算问题、数仓问题
  * 底层是湖仓一体
* 云+智能
  * “我的长沙”APP：“融媒体+城市服务”融合平台

![image-20211202153930004](./%E4%BA%91%E5%8E%9F%E7%94%9F-ToB/rec-value.png)





#### Borg, Omega, and Kubernetes

https://skyscribe.github.io/post/2019/07/21/from-borg-to-kubernetes/

https://www.wired.com/2015/09/google-2-billion-lines-codeand-one-place/

Borg was built to manage both long-running services and batch jobs

Omega was driven by a desire to improve the software engineering of the Borg ecosystem.

Kubernetes is accessed exclusively through a domain-specific REST API that applies higher-level versioning, validation, semantics, and policy, in support of a more diverse array of clients.

CONTAINERS

两个重要概念：the runtime isolation and the image

* user-facing jobs reserve more resources
* containers cannot prevent interference in resources that the operating-system kernel doesn’t manage, such as level 3 processor caches and memory bandwidth
* containers need to be supported by an additional security layer (such as virtual machines) to protect against the kinds of malicious actors found in the cloud
* a modern container includes an image

APPLICATION-ORIENTED INFRASTRUCTURE

* This shift of management APIs from machine-oriented to application oriented dramatically improves application deployment and introspection.
* 容器不止利于production，同样有益于development
* The key to making this abstraction work is having a
  hermetic container image that can encapsulate almost all of an application’s dependencies into a package that can be deployed into the container.
* containers's generic APIs
  *  the health check uses a user-specified HTTP endpoint or exec command that runs inside the container.)

every Kubernetes object has three basic fields in its description: ObjectMetadata, Specification (or Spec), and Status.

* 基于对容器“观测”的设计，让k8s对controller依赖小
* The design of Kubernetes as a combination of microservices and small control loops is an example of control through choreography—achieving a desired emergent behavior by combining the effects of separate, autonomous entities that collaborate. This is a conscious design choice in contrast to a centralized orchestration system, which may be easier to construct at first but tends to become brittle and rigid over time, especially in the presence of unanticipated errors or state changes.
* 声明式API

THINGS TO AVOID

* Don’t make the container system manage port numbers
* Don’t just number containers: give them labels
* Be careful with ownership
* Don't expose raw state

SOME OPEN, HARD PROBLEMS

* Configuration
  * configuration需要，但考虑到separation of computation and data，它的复杂度应该有限度
* Dependency management

### docker

* [docker container run](https://phoenixnap.com/kb/docker-run-command-with-examples)
  * [docker build and run](https://www.freecodecamp.org/news/docker-easy-as-build-run-done-e174cc452599/)
  * [network模式](https://loocode.com/post/docker-network-ru-men-yong-fa)
    * host/none/bridge
    * `docker inspect bridge`
    * `-P`选项Docker会把Dockerfile中的通过EXPOSE指令或`--expose`选项暴露的端口随机映射到临时端口
    * [Docker容器访问宿主机网络](https://jingsam.github.io/2018/10/16/host-in-docker.html)
      * `ip addr show docker0`
      * `host.docker.internal`

#### cgroup

##### cgroup基础

[一篇搞懂容器技术的基石： cgroup](https://segmentfault.com/a/1190000040980305)

* cgroup 的主要作用：管理资源的分配、限制；
* namespace 的主要作用：封装抽象，限制，隔离，使命名空间内的进程看起来拥有他们自己的全局资源；
* Chroot的安全性问题：sudo下，程序在当前目录和系统原本的根目录下可进行切换
* Linux VServer 应用程序：针对 "chroot-again" 类型的攻击没有很好的进行安全保护
* cgroup 主要限制的资源是：
  - CPU
  - 内存
  - 网络
  - 磁盘 I/O
* cgroup 主要有两个组成部分：
  - core - 负责分层组织过程；
  - controller - 通常负责沿层次结构分配特定类型的系统资源。每个 cgroup 都有一个 `cgroup.controllers` 文件，其中列出了所有可供 cgroup 启用的控制器。当在 `cgroup.subtree_control` 中指定多个控制器时，要么全部成功，要么全部失败。在同一个控制器上指定多项操作，那么只有最后一个生效。每个 cgroup 的控制器销毁是异步的，在引用时同样也有着延迟引用的问题；
* cgroup核心文件
  * cgroup.type
  * cgroup.procs
  * cgroup.controllers
  * cgroup.subtree_control
  * cgroup.events
  * cgroup.threads
  * ...

* 跨 cgroup 迁移进程是一项代价昂贵的操作并且有状态的资源限制（例如，内存）不会动态的应用于迁移。因此，经常跨 cgroup 迁移进程只是作为一种手段。不鼓励直接应用不同的资源限制。
  * 当一个进程 fork 出一个子进程时，该进程就诞生在其父亲进程所属的 cgroup 中。

* cgroups
  * 当明确提到多个单独的控制组时，才使用复数形式 “cgroups” 
  * cgroups 形成了树状结构。（一个给定的 cgroup 可能有多个子 cgroup 形成一棵树结构体）每个非根 cgroup 都有一个 `cgroup.events` 文件，其中包含 `populated` 字段指示 cgroup 的子层次结构是否具有实时进程。所有非根的 `cgroup.subtree_control` 文件，只能包含在父级中启用的控制器。
  * **子节点 cgroup 与父节点 cgroup 是否会存在内部进程竞争的情况呢**？
    * 当然不会。cgroup v2 中，设定了非根 cgroup 只能在没有任何进程时才能将域资源分发给子节点的 cgroup。简而言之，只有不包含任何进程的 cgroup 才能在其 `cgroup.subtree_control` 文件中启用域控制器，这就保证了，进程总在叶子节点上。

* 委派和迁移
  * 跨 cgroup 迁移，从委派中，我们可以很明确的得知跨 cgroup 迁移对于普通用户来讲，是有限制条件的。即，是否对目前 cgroup 的 “cgroup.procs” 文件具有写访问权限以及是否对源 cgroup 和目标 cgroup 的**共同祖先的 “cgroup.procs” 文件具有写访问权限**。

* cgroups 的资源分配模型：
  - 权重 - (例如，cpu.weight) 所有权重都在 [1, 10000] 范围内，默认值为 100。按照权重比率来分配资源。
  - 限制 - [0, max] 范围内，默认为“max”，即 noop（例如，io.max）。限制可以被过度使用（子节点限制的总和可能超过父节点可用的资源量）。
  - 保护 - [0, max] 范围内，默认为 0，即 noop（例如，io.low）。保护可以是硬保证或尽力而为的软边界，保护也可能被过度使用。
  - 分配 - [0, max] 范围内，默认为 0，即没有资源。分配不能被过度使用（子节点分配的总和不能超过父节点可用的资源量）。


```shell
➜  ~ docker run --rm -d  --cpus=2 --memory=2g --name=2c2g redis:alpine 
e420a97835d9692df5b90b47e7951bc3fad48269eb2c8b1fa782527e0ae91c8e
➜  ~ cat /sys/fs/cgroup/system.slice/docker-`docker ps -lq --no-trunc`.scope/cpu.max
200000 100000
➜  ~ cat /sys/fs/cgroup/system.slice/docker-`docker ps -lq --no-trunc`.scope/memory.max
2147483648
➜  ~ 
➜  ~ docker run --rm -d  --cpus=0.5 --memory=0.5g --name=0.5c0.5g redis:alpine
8b82790fe0da9d00ab07aac7d6e4ef2f5871d5f3d7d06a5cdb56daaf9f5bc48e
➜  ~ cat /sys/fs/cgroup/system.slice/docker-`docker ps -lq --no-trunc`.scope/cpu.max       
50000 100000
➜  ~ cat /sys/fs/cgroup/system.slice/docker-`docker ps -lq --no-trunc`.scope/memory.max
536870912
```

##### cgroup 资源限制

* https://docs.kernel.org/scheduler/sched-bwc.html

  * cpu.cfs_period_us：设置了每个周期的时长（默认100000us

  * cpu.cfs_quota_us设置了每个周期最多可以用的cpu数量（例如200000us）则表示在每个周期内可以使用两个cpu

  * 在k8s层面：cpu.cfs_quota_us/cpu.cfs_period_us == pod的cpu limit（也就是grafana面板里面 CPU Core (limit) 这个指标）

![1280X1280 (1)](./%E4%BA%91%E5%8E%9F%E7%94%9F-ToB/cpu-stat.png)

##### [namespace技术](https://segmentfault.com/a/1190000041096866?utm_source=sf-similar-article)

* cgroup namespaces

  * Cgroup namespace 是进程的 cgroups 的虚拟化视图，通过 `/proc/[pid]/cgroup` 和 `/proc/[pid]/mountinfo` 展示。
  * 使用 cgroup namespace 需要内核开启 `CONFIG_CGROUPS` 选项。可通过以下方式验证：`grep CONFIG_CGROUPS /boot/config-$(uname -r) --> CONFIG_CGROUPS=y`

* cgroup namespace 提供了一系列的隔离支持：

  - 防止信息泄漏（容器不应该看到容器外的任何信息）。
  - 简化了容器迁移。
  - 限制容器进程资源，因为它会把 cgroup 文件系统进行挂载，使得容器进程无法获取上层的访问权限。

* 一些类型：cgroup、ipc、network、mount

* init 进程

  * 我们都知道在 Linux 系统中有一个进程比较特殊，所谓的 init 进程，也就是 PID 为 1 的进程。
  * 前面我们已经说了每个 PID namespace 中进程号都是从 1 开始的，那么它有什么特点呢？首先，PID namespace 中的 1 号进程是所有孤立进程的父进程。其次，如果这个进程被终止，内核将调用 `SIGKILL` 发出终止此 namespace 中的所有进程的信号。 **这部分内容与 Kubernetes 中应用的优雅关闭/平滑升级等都有一定的联系。**
  
  * 最后，从 Linux v3.4 内核版本开始，如果在一个 PID namespace 中发生 `reboot()` 的系统调用，则 PID namespace 中的 init 进程会立即退出。**这算是一个比较特殊的技巧，可用于处理高负载机器上容器退出的问题。**
  
* setns(2) 调度：进程只能从父 PID namespace 调度到 子 PID namespace 中

* time namespaces

* user namespaces

* UTS namespaces

* Namespaces API

  * clone(2)

  * unshare(2)

  * setns(2)

* 关键目录

  * /proc/[pid]/ns/

  * /proc/sys/user 目录下的文件记录了各 namespace 的相关限制。当达到限制，相关调用会报错 error ENOSPC 

#### docker.file

* ENV PATH=/usr/lib/bin:${PATH}
  * ENV ABC abc

* [Docker磁盘空间不足如何解决](https://blog.csdn.net/weixin_39505820/article/details/121850984)
  * /var/lib/docker
  * /etc/docker/daemon.json

#### docker-compose

```
docker-compose up -d
```





#### [Docker and the PID 1 zombie reaping problem](https://blog.phusion.nl/2015/01/20/docker-and-the-pid-1-zombie-reaping-problem/)

* Docker does not run processes under a special init process that properly reaps child processes, so that it is possible for the container to end up with zombie processes that cause all sorts of trouble. 

* Docker also does not do anything with syslog so that it's possible for important messages to get silently swallowed, etcetera.

* 问题原理：

  * Unix Process的概念：each process has a parent except for the top-most process.

    * init process, started by the kernel when you boot your system
    * this init process is responsible for starting the rest of the system, such as starting the SSH daemon, starting the Docker daemon, starting Apache/Nginx, starting your GUI desktop environment, etc. Each of them may in turn spawn further child processes.

  * Process terminiate时发生什么

    * "defunct process", also known as a "zombie process".
      * processes that have terminated but have not (yet) been waited for by their parent processes.
    * Unix is designed in such a way that parent processes must explicitly "wait" for child process termination, in order to collect its exit status
    * "reaping": 直到parent process系统调用wait_pid，zombie process才消失
    * e.g.  if bash terminates then the operating system will send a SIGCHLD signal to sshd to wake it up. Sshd notices this and reaps the child process.

  * init process -- PID 1

    * 动机：
      * orphaned process的含义是，Parent process终止，没有wait子进程即结束
      * 需要一个进程来 "adopt" orphaned child processes
    * **the operating system expects the init process to reap adopted children too**.
    * Pretty much all daemon software [expect that daemonized child processes are adopted and reaped by init](http://stackoverflow.com/questions/881388/what-is-the-reason-for-performing-a-double-fork-when-creating-a-daemon).

  * Why zombie processes are harmful

    * 占用内核资源：consume a slot in the kernel process table, and if this table fills, it will not be possible to create further processes

  * 和docker的联系：docker container不是一个完整的init system，需要用户自己确保zombie process被正确reap

  * 解决思路：利用bash的fully init system的能力

    * ```
      方案一：CMD ["/bin/bash", "-c", "set -e && /path-to-your-app"]
      ```

    * 问题：无法传递SIGTERM信号，只能传递SIGKILL signal. SIGKILL cannot be trapped, so there is no way for processes to terminate cleanly

      * Suppose that the app you're running is busy writing a file; the file could get corrupted if the app is terminated uncleanly in the middle of a write.
      * 为什么需要考虑SIGTERM的传递？ 因为docker stop是发送这个信号

    * 方案二：

      * ```
        #!/bin/bash
        function cleanup()
        {
            local pids=`jobs -p`
            if [[ "$pids" != "" ]]; then
                kill $pids >/dev/null 2>/dev/null
            fi
        }
        
        trap cleanup EXIT
        /path-to-your-app
        ```

    * 问题：the init process must also *wait* for child processes to terminate, before terminating itself

    * 最终实现：https://github.com/phusion/baseimage-docker/blob/rel-0.9.16/image/bin/my_init

* 最佳实践：https://forums.docker.com/t/what-the-latest-with-the-zombie-process-reaping-problem/50758/2
  * Docker-compose默认开启 init: true





### Kubernetes

![kubernetes](./%E4%BA%91%E5%8E%9F%E7%94%9F-ToB/kubernetes.png)

**Kubernetes 架构核心组件**

* ETCD: 存储 K8S 对象
* API Server：API 入口
* Scheduler：任务和资源调度中心
* Controller Manager：对象控制中心
  * 不断驱动对象向用户指定的Spec进行变更
  * e.g. 无状态服务的升级、扩容、回滚
* Kubelet：计算节点 
  * 单机层面的agent，负责容器的生命周期管理、单机维度的资源管控
  * 会和其它可拓展插件 (Networking, Container Runtime) 协同



**一切皆对象 (Object)**

给定对象预期状态 (Spec)，系统不断自驱运行直到最终状态 (Status) 符合 Spec（声明式架构）

**基本对象** 

* Pod：最小调度单元，可包含多个 container
* Deployment：无状态服务的抽象
* Node：机器资源抽象
* CRD：自定义业务形态扩展



有状态服务

* 多分片：给每个pod分配shard id
* 持续存储：本地盘/远程盘



#### k8s with MLSys

* Persia: https://github.com/PersiaML/tutorials/blob/main/src/kubernetes-integration/index.md
  * [nats operator](https://github.com/nats-io/nats-operator)、mounting volumes、image、configure resources/envs

#### kubectl

```shell
 kubectl get pods
 kubectl get pods -l psm=xxx
 kubectl exec -it xxxxx (-n $namespace) bash
 kubectl cp $container:$path $local_path
 kubectl describe pod $pod_name
 
 kubectl logs -f $pod_name -n $namespace
```

##### 查看pod所在CPU

```shell
kubectl exec -it $pod_name bash
cd $(echo /sys/fs/cgroup/cpu/$(cat /proc/self/cgroup |grep cpu,cpuacct|awk -F':' '{print $NF}'))
cat cpu.cfs_period_us 
cat cpu.cfs_quota_us 
cat cpu.stat
```

### 镜像

#### [火山引擎基于 Dragonfly 加速实践](https://mp.weixin.qq.com/s/NTgMiUZYLaLs_LmlbWZ3pA)

* 术语
  * **ECS**：是一种由CPU、内存、云盘组成的资源集合，每一种资源都会逻辑对应到数据中心的计算硬件实体。
  * **CR**：火山引擎镜像仓库服务。
  * **VKE**：火山引擎通过深度融合新一代云原生技术，提供以容器为核心的高性能 Kubernetes 容器集群管理服务，助力用户快速构建容器化应用。
  * **VCI**：火山一种 Serverless 和容器化的计算服务。当前 VCI 可无缝集成容器服务 VKE，提供 Kubernetes 编排能力。使用 VCI，可以专注于构建应用本身，而无需购买和管理底层云服务器等基础设施，并仅为容器实际运行消耗的资源付费。VCI 还支持秒级启动、高并发创建、沙箱容器安全隔离等能力。
* 千台集群，秒级启动
* 细节
  * 基于机器学习的多场景自适应智能 P2P 节点调度, 为当前下载节点选择最优父节点。

#### docker镜像大小

* [如何给 Docker 镜像瘦身？](https://www.infoq.cn/article/tbiwieu87e*wkunvjdwm)
  * 中间层: docker image history my_image:my_tag





### 云厂商相关

#### 机器规格定价

* aws

https://aws.amazon.com/cn/ec2/instance-types/

#### 定制内核功能

* aliyun
  * https://help.aliyun.com/zh/alinux/product-overview/alibaba-cloud-linux-overview?spm=a2c4g.11186623.0.0.6ada55623xzX9G



### k8s 与性能

#### 主机超售

* k8s/docker容器技术的原理是，通过namespace限制进程可见性，通过cgroup限制资源quota，同一个主机上的所有pod共享同一个内核；除此之外，在主机的视角里，pod内的进程和普通进程并没有多大差别。 由于内核是共享的，也就是所有pod内的进程运行在同一个操作系统中，因此相对于虚拟机而言，一个pod更容易受同一个主机上的其他pod影响 --- 正如同一个主机上，一个进程会受另一个进程的影响一样。
  * 例如：主机上所有进程共享TLB和CPU 缓存，和其他进程的激烈竞争，会导致TLB和CPU缓存命中率降低；所有进程都受同一个内核的调度，和其他进程的激烈竞争，会提高进程被非自愿抢占的频率，增加进程在CPU上排队的延迟。

* 衡量主机超售对业务pod影响的性能指标和衡量CPU是否成为了性能瓶颈的指标相同：

  - 主机CPU频率

  - IPC

  - CPU队列长度

  - task调度延迟

  - task非自愿抢占

* 最简单的确认方式：控制有/无其它租户，做控制变量，测量e2e指标和微架构指标

### 运维监控

#### 网络

* PingMesh https://cloud.tencent.com/developer/article/1780947

#### 通用

* [Grafana：SpaceX 的数据监测利器，云原生领域的 Tableau](https://mp.weixin.qq.com/s/zgd8KjpGoqwPGC6b1I9owg)
  * 本质是提升数据可观测性（Data Observability），打破数据边界，提供一个“统一的视窗”，实现对数据的全览和实时监控
  * 也有观点认为，可视化的重要性远大于指标、日志和链路追踪
  * 推动“数据民主化”



### AI-ToB

#### Intro

![image-20250616165500735](./%E4%BA%91%E5%8E%9F%E7%94%9F-ToB/image-20250616165500735.png)

#### 火山AI Force大会

* https://www.volcengine.com/event/force-2506
  * 刊例价 70元/1000万 token
  * 16.4万亿 token 等于 1.15亿元

![image-20250616164911048](./%E4%BA%91%E5%8E%9F%E7%94%9F-ToB/image-20250616164911048.png)

![image-20250616165347724](./%E4%BA%91%E5%8E%9F%E7%94%9F-ToB/image-20250616165347724.png)

### ToB

#### Intro

* [一篇分析 ToB 与 ToC 技术上区别的文章](https://zhuanlan.zhihu.com/p/341358485)
  * tob更注重数据设计、数据一致性
  * toc：分布式解决方案、花式存储、异步化
* 标准化 v.s. 服务类产品
  * 产品线相对可控，单个产品硬件成本低、可以开渠道做销售。需要的营销方向的人手相对少。渠道要分润，所以毛利低，项目周期相对短一些。
    * 渠道不是越多越好：市场中渠道太多，区域市场价格竞争激烈，渠道毛利就会降低，渠道伙伴不赚钱，就没有积极性。
  * 服务类：做项目服务业务，产品方向多，难以标准化，单个产品硬件成本高，需要很多人手。需要自己做项目交付，所以毛利高项目周期长。

#### ToB 阶段

* PMF Product Market Fit / 产品符合市场匹配

  * B端产品（硬件或软件）达到PMF的一种判断标准可能是获得20-30个企业级付费

  * 验证点：

    * 核心卖点
      * 如何打动客户
    * 报价清单
      * 客户愿意为哪些关键特性付费

  * 各环节状态

    * 销售
    * 报价
    * 案例
    * 优劣势认知
    * 产品销售模式（免费、捆绑、付费）

  * 新进入者定价：

    * B端：行业领导者的70%
    * C端：行业领导者的50%

  * 周期：

    * 12-18个月

  * 其它：

    * 小微客户SMB的付费不能作为PMF的一个验证标准
    * 企业级：1000人以上的企业和组织

    

#### 火山引擎业务

* [一篇从火山引擎出发的ToB市场分析文章](https://zhuanlan.zhihu.com/p/519653936)
  * 在音视频基础编辑、智能编辑、CV、创意商城四个方面都拥有突出优势
  * 巨头选择了不同类型差异化的道路，比如阿里云的大而全；比如腾讯云是以微信小程序搭建作为突破口，形成自己的体系优势；比如说百度智能云的AI能力；比如说华为云的政企服务等等。
  * 火山引擎公布的十三家核心客户中，抖音，头条和西瓜三个是字节自家的应用，其他包括：中信银行，美的，得到，虎扑，悟空租车，彩云科技，故事，不背，网眼，和STARLINK

##### [V-moment特别版](https://www.volcengine.com/docs/search?q=%E7%89%B9%E5%88%AB%E7%89%88)

##### [第一期|云上视界新看法](https://www.volcengine.com/docs/6703/101619) 视频云

* 定义：制作、存储、处理、分发、分析、审核、检索、推荐、理解等
  * 完整的市场包括：公有云、私有云、混合云基础设施、视频内容分发网络、面向视频应用场景的云解决方案
* 场景：远程特点
  * 点播：编解码、存储分发
  * 直播：基于直播，有上行加速、WebRTC方案
  * 音视频通信：增加了时延、稳定性要求
* 市场：70亿美元量级
  * 未来五年增长方向：传统行业的视频云需求
  * 技术演进方向：高清、交互、with AI
  * 医疗行业的碎片化，让视频云应用不易
* 对内到对外角色的转化
  * 专精 -> 通用：更清晰地划分业务层次，需要做原子化拆分、设计更精细的API
  * 客户考核发生变化：以前是结果导向，现在需要考虑最佳实践方案和行业通用方案对齐
  * 灵活性：更多的适配工作
  * 团队人员思维转变：由专精到更注重全面性、服务性

##### [特别版第一期|效率竞争时代，如何构建敏捷的数字化生产力](https://www.volcengine.com/docs/6703/130251)

* 不确定的外部环境：阵地在，敌人没了
* 数字化的意义，提升企业的“变”的能力：1. 对外业务效果；2. 对内管理效率
* 深势科技联合创始人兼首席科学家张林峰
  * AI for science，分子动力学、蛋白结构预测
  * 痛点是规模化，上云是被推着走
  * 技术方案的复杂度和业务规模挂钩 --> 上成熟的云，能减少踩坑，帮助在业务的一些关键点获取突破
* 火山引擎的效率优势
  * 软硬件技术创新，运营技术（大规模并池）
  * 敏捷性，云原生全栈能力
    * 核心云原生基础
    * 融合化：云原生+ML；云原生+数仓
    * 敏捷的概念是相对的，行业内部比较的（我的理解，云/平台等，让行业内卷。。。）
  * 行业抽象：金融等
* 云原生改造能力：利用云上最新的技术特性，进行业务实践改造，帮助满足增长需求
  * 容器、微服务、存算分离、低代码、serverless
  * 改造期间需要保证原有架构的运行，因此需要长期规划
    * 无状态服务容器化 -> 有状态服务 -> 多云适配 -> 微服务治理
    * 敏捷问题 -> 成本问题 -> 治理问题
  * e.g. 弹性能力解决业务扩缩容问题
  * e.g. 本土到全球化发布的改造问题
  * e.g. 创新业务，利用特色能力
* 面向未来的数字化合作模式：深势 with VolEngine
  * 突破舒适区，互相学习提升认知
  * 深势的痛点是idea的快速落地，因此给创新的土壤以更好的生态
  * 聪明的客户，从“自发”到“自觉”
*  火山引擎的业务优势：服务“聪明的客户”，提供合适的特性，帮助客户“主动进化”，和客户共同成长
  * 多云时代，云和业务发展的适应性，比如Paas接入多种IaaS的能力、cdn和rpc等能力
  * 内容和创意上的特性：数分、Machine Learning
  * 如何和业务共同成长，并发挥自己的差异化优势
    * 产品设计中的同理心，在内部有演进，对外服务的时候同样能有演进

##### [第二期|火山引擎支招应对云原生实践难点](https://www.volcengine.com/docs/6703/130703)

* 国内云原生的发展阶段：强调企业的整体应用面向云去设计
  * “开发+设计+运维”的理念
  * 以15-20年为周期，目前处于1/4~1/3的阶段，成长期

* 行业视角：**老问题老办法，新问题新办法**
  * 传统行业：金融电信政府交通，有自己的IT系统，单体架构可能经过了大量的迭代，比较成熟了
  * 传统行业中的新领域：自动驾驶、仿真
  * e.g. 晶泰科技，AI药物研发
  * e.g. 华联证券海豚股票APP
    * z世代，内容视频化、功能智能化、体验游戏化、平台社交化、推送个性化
    * 需求：内容连续性+弹性+内容服务
* 为什么海量算力问题是难题？
  * 算力哪里来
  * 如何管理海量算力
    * 潮汐问题

  * 如何提高算力的使用效率
    * 弹性调度：资源分配调度器、二进制分发（影响任务的拉起时间）
      * 用VKE解决二进制分发问题

    * GPU集群调度：网络要求高，云上RDMA
      * 192节点 4*200G RDMA集群

  * 如何判断是否充分发挥了算力价值

* 底层高效稳定，上层灵活多变
  * 如何实现上层对底层的infra awareness：容器平台、k8s调度平台
    * k8s知道底层设施状态
    * node discovery function

  * 软硬一体、异构计算，让算力更高效
    * 针对workload优化：intel crypto-ni、avx512、取多列数据

* 关于高可用
  * 分两个层面，一是基础设施层面，二是架构设计层面

  * 全生命周期管理：软件开发运维测试发布
    * design for failure: k8s的设计理念
    * 兼容性的设计，比如容器内静态IP的能力，满足传统应用的发现、调度、特定运维需求

  * 多机房容灾

* 关于弹性
  * 单体应用的互动性设计的bottleneck是瞬息万变的，不容易做弹性
  * 微服务：解耦复杂问题
* 云原生的发展
  * 容器 -> k8s -> 函数计算、无服务计算、云原生引擎（WebAssembly, Dapr）
  * 调度引擎
  * 从应用的角度：云原生在往上走（往企业产生价值的部分走）
  * AIOps
  * 软硬一体
  * 2025年中国数字化转型的费用规模将达到5000亿元


##### [特别版第二期|“内容为王”的时代，体验创新如何助力业务增长](https://www.volcengine.com/docs/6703/130862)

* 泰康保险
  * 移动互联网时代，先是保险上手机端，然后提升内容能力与用户互动
  * 信息源：官宣类、营销类
    * 火山引擎提供的内容加工能力，能快速结合，官方消息+企业自己的理解，进行二次创作
    * 释放剪映的PaaS能力
  * 好的商业模式 + 好的大规模科技能力
    * 取长补短，最重要的是自己的产品
  * 是否要从非数字原生企业改造为数字原生企业？
    * 现有的：金融属性 -> 保险+医养康宁服务（大健康生态企业）
    * 当战略布局变化后，利用数字化进行一系列的重组重构，形成最有效率的商业模式
    * 数字原生不是0和1，而是一条线
* 云上增长三要素：体验创新+数据驱动+敏捷迭代（泰康视角）
  * 体验创新：字节提供数据驱动下体验迭代的技术体系，文字、图片、视频、直播、元宇宙
  * 数据驱动：非数字原生企业的玩法完全不一样，存在线下特征
    * 线下数据加点：公司/经纪人/客户
    * 关键点在数据采集端，如何做好“数字孪生”
    * 从O2O到OMO 结合线上与线下 [ref1](https://www.sohu.com/a/609824185_121124322), [ref2](https://blog.shopline.hk/what-is-omo/)
    * 具体措施：
      * 作业在手机端完成（前提是合规）
      * abtest的应用
      * 三大在线：客户在线（3kw）、代理人在线、员工在线
    * 获客：从公域（微信系、字节系）到私域
      * 能否打通公域和私域？
        * 公私域的概念是相对的
        * 私域侧重服务
        * 数据效率：联邦学习、隐私计算等技术
  * 敏捷迭代
* 字节角度
  * 和“趋势型企业”合作共创，思考多方的优劣势
  * tob称竞品、toc称友商，经常有多个巨头服务同一个客户
  * 既解决大规模的问题，又解决小而美的问题
* 泰康视角点评各大云服务，很圆滑。。
  * 字节：内容
  * 腾讯：社交
  * 阿里：金融、电商
  * 百度：AI

##### [特别版第三期|数字经济时代，如何让数据真正驱动业务增长](https://www.volcengine.com/docs/6703/158929)

* 人物
  * 伊利集团数字化中心总经理尚直虎
  * 财经作家、890新商学以及蓝狮子出版创始人吴晓波
* 云服务对制造业企业的重要性
  * 云作为底座：IaaS
  * 云作为平台：PaaS（对外、对内）、Saas（运营、营销）
  * 云厂商的生态相比技术对实体企业更有意义
  * 互联网公司已经和实体经济耦合在一起
* 伊利
  * 大营销+大工业
  * 转型过程：从单点到多点到系统聚合，2019年启动
    * 核心：拥抱云、持续沉淀高价值数据资产
    * 各种数字化技术：会员系统、供应链系统、制造智能化、消费者触达能力、产品多元化
    * VR探索工厂设备检测
  * 用了一两年做数字化转型，先规划数据架构，再建设数据治理体系，再沉淀出“普惠高效智能”
  * 如何面对有的中层干部用不好数字技术的问题
    * 答的角度挺有意思，说从普惠的角度，要降低数字技术的使用门槛
  * 与字节的合作：短期任务驱动、长期产品战略驱动，POC->Project->Product->Platform
    * 抖音上面食品最大的品牌，兴趣电商
    * AR足球营销
    * 数据：从EIP系统到数据中台，人们总是短期高估长期低估技术价值
    * “须尽欢”冰淇淋，找合适的KOL，数据驱动从30多个创意实现2、3个创意方向
* 德国经验：先做云仓储，数字化工厂；   中国习惯是先解决终端问题
  * 区别的原因在于增长率
  * 生产端与营销端
* 不同体量的公司，提供的ToB能力有什么区别？
  * 小公司：对营销更关注，标准化的方案（比如直播间怎么做装修）
* 如何利用数据：一是“数据飞轮”，分四步；二是企业文化
  * 业务数字化
    * 背景：数字化转型，整个ToB市场有30%的客户白花钱了
    * 平台提供解决方案还是任务给客户？
      * 取决于平台自己是否Know-how，是否能起到coach的角色
    * 通用工具：数字埋点埋在哪是行业Know-how，埋点更精准和数据治理是平台能力
    * 做的好坏，老板和中层干部很重要
  * 数字协同
    * 打通不同部门，涉及组织变化
    * 数据权限问题（腰部中层能很好地利用数据）
    * 具体技术：流式计算、实时数仓、分库分表，ByteHouse
  * 数字创新
  * 数据评估
    * abtest
    * 积累数据
  * 再回到业务数字化
  * 关于企业文化
    * 业务部门和数据部门如何协同
    * 数据部门分为数据系统和数据业务BP，数据BP提升业务团队和数据团队的配合能力
* [字节跳动杨震原：抖音电商是如何实现数据驱动的？](https://zhuanlan.zhihu.com/p/546045914)
  * 实时数仓：实时大屏、实时分析、实时预警、实时营销
    * 数据一致性问题
  * 数据BP的标准“0987”：1）支持业务敏捷迭代；2）通过量化指标度量数据中台的工作
    * “0”：第一维度，关注稳定性指标，指数据中台产生数据要稳定，做到故障数SLA故障清零；
    * 数据丢了、数据错了（埋点错了、顺序错了）
    * “9”：第二维度，关注需求满足度，业务需求满足率要达到90%；
    * “8”：第三维度，关注数据中台数仓完善度，分析师查询覆盖率达到80%；
    * 数据覆盖问题，比如提前join之类的，利用ML、软硬加速等
    * “7“：第四维度，关注用户满意度，我们用NPS指标来看服务满意度，向业务同学收集调查问卷，目标NPS是70%。
      * [关于客户体验你必须知道的3大指标：CSAT/NPS/CES](https://zhuanlan.zhihu.com/p/30268198)
  * 数据治理
    * 分布式治理：要有治理委员会去制定各种标准，这些标准也都是从业务上传，在每个业务中也会有专人负责治理工作，让治理工作自下而上产生出来。
    * 经验复用
    * 经验沉淀到工具（DataLeap）

![img](./%E4%BA%91%E5%8E%9F%E7%94%9F-ToB/bytedance-data-platform.png)

* 如何将抖音的能力融入火山中
  * 火山做的包括公域和私域的一些事情，比如素材层面的剪映、投放层面通过火山引擎获得更细的用户画像颗粒度、直播间的装修、达人的匹配
  * [火山“万有计划”](https://www.volcengine.com/docs/6359/129308)
    * 抖音集团 资源、商机、品牌、产品、技术、政策与运营 等七大价值

##### TODO [2023第一期|如何通过大模型应用，创新用户体验](https://www.volcengine.com/docs/6703/1113728)





#### 火山引擎技术

* [使用 Rclone 访问 TOS](https://www.volcengine.com/docs/6349/81434)
  * [rclone官网](https://rclone.org/docs/)
    * copy
      * `--s3-no-head-object`
      * `rclone copy testdir volces-tos:bucket-demo`  将文件夹内所有文件放到tos
    * ls, lsd, lsl
      * `rclone $远程连接:$一级桶`
    * `rclone sync testdir volces-tos:bucket-demo/testdir`
    * size
    * check
      * `--size-only`：只比较文件大小。
      * `--download`：下载远程文件并对比。
  * `~/.config/rclone/rclone.conf`

```shell
curl -O https://downloads.rclone.org/rclone-current-linux-amd64.zip
unzip rclone-current-linux-amd64.zip
cd rclone-*-linux-amd64

sudo cp rclone /usr/bin/
sudo chown root:root /usr/bin/rclone
sudo chmod 755 /usr/bin/rclone

sudo mkdir -p /usr/local/share/man/man1
sudo cp rclone.1 /usr/local/share/man/man1/
sudo mandb

rclone ls remote:path # lists a remote
rclone copy /local/path remote:path # copies /local/path to the remote
rclone sync -i /local/path remote:path # syncs /local/path to the remote
```

```
[tos]
type = s3
provider = Other
access_key_id = AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
secret_access_key = AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
region = cn-beijing
endpoint = BBBBBBBBBBBBBBBBBBBB
force_path_style = false
disable_http2 = true
no_check_bucket = true

rclone lsd tos:
touch README.txt
rclone copy README.txt tos:ABC --s3-no-head-object
```



#### 再就业小酒馆 Blog

http://xhslink.com/byZMGl

![image-20221205185439684](./%E4%BA%91%E5%8E%9F%E7%94%9F-ToB/tob-stack.png)

* toB软件市场的本质和趋势
  * IaaS (55%)：稳定成熟，多云融合趋势，中国电子云
  * Paas (30%)
    * 数据中台：数据的采、存、算、管、用
      * 反观国内，更多关注是数据应用价值，对应的是数据资产管理类产品或数据应用，实施部署起来更多是项目，核心竞争力是从客户学习到横向经验。
    * 智能中台：项目居多，业务形式的有AIGC
    * 连接集成平台：对标国外的MuleSoft，多指数据和应用层面的集成
      * 新型路径钉钉、飞书，点集成
    * 低代码开发平台：主要核心是流程引擎和表单引擎，会结合AI，也有人把RPA往这里面放，解决的就是定制化的问题，以及不同角色信息传导的差异问题。
  * SaaS：渲染、飞书、推荐
    * role-based
    * industry-based: 闷声发小财
* 中美企业软件差异
  * 行业维度
    * 美国是消费型社会，中国仍处于生产型社会
  * 客户维度
    * 行业整体预算大头来自国企，因多种因素更愿意买硬件
  * 人才维度
    * AI、电商、支付领域是领先者
    * 产品营销类人才较为匮乏
  * 公司维度
    * 组织支持：中国更重视销售关系，大包大揽
  * 国内做sass还是难做，saas分为行业的和通用的，创业公司能起来的以行业为主，通用的基本是大厂在做生态，高频打低频
  * [中国真的适合sass吗？](https://www.zhihu.com/question/420454515/answer/2825655168)
    * saas是为了降本增效，估值高的公司真的很多需要saas吗？（医生、律师），受众主要还是中小企业
    * 中国的中小企业：一是没有，二是代表不了先进生产力的方向，三是降本增效不容易带来优势，四是就算有竞争优势，也不好淘汰老旧企业
    * 适合国情的SaaS：大型外包，ToG
* 被宠坏的软件公司要向硬件和服务公司学
  * 要注重产品PFM（product market fit）
  * 提升每个项目毛利，保证净利的正值
  * 经济学指标
    * 扣非净利润：扣除非经常损益后的净利润
    * 毛利率=[(销售收入-销售成本)/销售收入]*100%
    * `净利率=净利润/销售收入*100%=(销售收入-销售成本-各项期间费用-税金)/ 销售收入*100%`
    * 人效比=营业收入/人员总数

![image-20221205211302189](./%E4%BA%91%E5%8E%9F%E7%94%9F-ToB/tob-data.png)



#### 双因素增长模型 ToB

TODO 胡贤彬的blog https://www.jianshu.com/u/350f609099e2

* 案例：阿里云HTTPDNS，0.5人运维，年几千万营收
  * 解决域名解析跳转的问题
  * 规模效应：新品收入 * 活跃用户 * 转化率
* 增长困境
  * 持续增长，理想化是天级别
  * [双因素增长模型](https://www.jianshu.com/p/842640e8a05c)：企业规模价值 = 单客价值 * 客户数量 / 营销费用^2
    * 单客价值：续约、增购
    * 客户数量：客群规模（留存）、拉新
  * 挑战1：国内私有云一次性买断的模式，大客户可持续价值低
    * “维保” 5%~15%
    * 大客户占 90% 以上营收，中长尾三年消失 90%
  * 挑战2：碎片化市场，制约“客群快速增长”
    * 研发定制化（行业/单客）：大厂人力成本高
  * 理想是标准化，否则重服务，大客户项目制（卖产品+定制化）
* 对策
  * 从“卖软件”到“卖服务”
    * 引导公有云、引导专属云（如金融云）、引导私部订阅、优先泛互行业、非泛互非核心业务上云
      * [金融云 by aliyun](https://help.aliyun.com/document_detail/29851.html)
        * 金融云是服务于银行、证券、保险、基金等金融机构的行业云，采用独立的机房集群提供满足一行两会监管要求的云产品，并为金融客户提供更加专业周到的服务。
        * 案例：众安保险
      * 泛互行业：媒体、游戏、工具等
        * 没有等保的准入要求
        * KA大部分用公有云、SMB几乎全部用公有云
        * 泛互决策上云的一个原因是上云很快，几天内就能跑起来（上云速度是一个衡量软件敏捷性的重要指标）
    * 可持续、短期营收规模小
  * 聚焦优质行业
    * 蚂蚁移动开发平台 mPaaS，先专注做金融
  * 行业解决方案“被集成”
    * 做多行业通用的PaaS
    * 行业伙伴：完成行业属性的最后一公里
    * e.g. 微软Azure，70%以上服务由“伙伴”提供
* 云上解决方案：销售驱动 & 产品驱动
  * SLG (Sales-Led-Growth)：大客户销售解决方案，做KA大单（公有云KA本质上是SLG）
    * call high
    * 技术标：差异化能力
    * 商务标：控标项是利润
    * 积累：客户关系（私域客户规模）、增购提高利润
    * Tips:
      * 倾斜泛互行业，主要使用公有云
      * 有合规要求的行业，非核心业务引导上云
      * 行业云
    * e.g. 腾讯服务拼多多
  * PLG (Product-Led-Growth)：SMB
    * 市场线索、销售线索、产品线索
    * PLG的思路，教育市场，从toc到tob，从个人到SMB到KA，先聚焦再泛化
    * e.g. Slack，个人/团队办公提效产品，特点是个人有决策权，形成用户习惯，能口碑传播



#### 一些产品

* [Palantir](https://www.palantir.com/)

* [C3.ai](https://c3.ai/#null)

### ML ToB

>  [Z Potentials｜高策，27岁离开腾讯字节创业，如何5个月获300万下载和品类第一的弯道超车？](https://mp.weixin.qq.com/s/S9wnlCt89cSn_7N4QCPOeQ)

* 回顾历史，所有关注workflow、分布式、并行的公司发展得都没有那么好，而Weights&Biases可能是上一代AI里面发展最好的公司，我觉得最大的原因就是关注到了一个别人没有关注到的点。它从易用性、从小团队出发，跑出了类似于PLG的感觉，大家现在一想到experiement checking都会想到Weights&Biases。
* Cases
  * 还有一个例子是HashiCorp。它就是靠着自己前瞻性的想法，把多云这件事情做得特别早，在只有一个云的时候就开始做了。在2012年它就做了非常多的多云工具，比如terraform，没有任何竞争。那个时候我会觉得这东西真是没用，但是现在大家一想到多云肯定都会先想到它，同时它的易用性做的真的太好了。
  * envd深度参考了HashiCorp的第一个开源的项目，Vagrant。它产生的背景是在没有docker的时候让用户通过声明式的方式写一个类似docker file的描述文件，构建一个基于虚拟机的开发环境出来。虽然比较重而且要花五六分钟，但是在当时是一个非常革命性的东西。当时配置开发环境是非常复杂的，没有docker也没有环境隔离。有了Vagrant之后，他们可以在虚拟机里做，并且可以只描述想要的东西而不关心实现和安装。
* serverless inference
  * 大家都发现这是一个很好的市场。因为它是全托管的，类似于公有云，技术优化可以降低成本，而且通过降低成本产生的利润完全是自己保留的，商业模式非常类似于Snowflake。
  * 相应地，竞争激烈、替换成本低
* LLM给ML ToB带来的变化
  * 在上一代AI中，通常是一个团队负责整个流程。团队中有数据科学家和机器学习工程师，数据科学家负责训练模型，机器学习工程师负责部署模型，但整体是一个团队。最大的问题在于，真正产生商业价值的是数据科学家，也就是负责训练的人。因为只有通过训练才能获得一个好的模型，这些模型才能产生业务价值。因此，训练人员在团队中有最大的发言权，并决定团队需要什么样的工具。
    * 这些训练人员在训练时往往是单机操作、针对具体问题建模，需要深厚的经验和特定领域知识，如神经网络结构设计和数据预处理等。很多时候，他们并不是计算机科学出身，而是专门学机器学习或数学的，他们是真正产生价值的人。
  * 现在使用模型训练的人，更多是架构师+算法
  * 现在使用推理的受众，和以前有很大不同，更广泛
    * 他们不再需要传统意义上的高易用性工具，而是需要一些新的工具，比如AI Gateway，它强调的是上线后和开发运维（DevOps）环节的应用，而不是开发过程中的效率提升。

#### 火山引擎

* [火山引擎 ML 平台 PR 文章](https://hub.baai.ac.cn/view/19131)
  * 性能优化、资源调度、开发机、硬件矩阵等优势

#### 推荐系统

BytePlus: https://www.byteplus.com/en/product/recommend

#### 其它

* [一篇 AIoT ToB 的文章，讲人脸识别打卡上班系统](https://mp.weixin.qq.com/s/wSl8KOp48ntDjggo0FLQxA)
  * 核心的问题是许多项目都是不同的场景和方向，后台产研资源无法统筹规划，所以很难走上规模化的道路。
  * 营销中用了“减法”思路。先把业务方向聚敛到**一个核心场景**——刷脸通行。**四个方案场景**——企业刷脸+迎宾、写字楼刷脸+访客、写字楼刷脸+电梯控制、企业刷脸+考勤
    * 确定刷脸通行场景是因为我司的技术能力强，在不同光线条件下的精度是行业顶尖的。这部分算法的研发和数据资源很充足，不需要负担额外的成本。
    * 选择企业和写字楼是因为这类客户是高端客户，价格敏感度低。
  * 选择做标准产品。同时把方案和项目开放给渠道伙伴。
  * 砍渠道：对每家渠道都做了能力评定和准入要求，提供商机，做技术支持和辅导
    * 评估回购渠道自己开发的解决方案，在渠道平台共享销售

* Google Pixel 的一些功能
  * More people than ever rely on their phone cameras to record their daily lives and for artistic expression. The clever application of ML to computational photography has continued to advance the capabilities of phone cameras, making them easier to use, improving performance, and resulting in higher-quality images. Advances, such as [improved HDR+](https://ai.googleblog.com/2021/04/hdr-with-bracketing-on-pixel-phones.html), the ability to [take pictures in very low light](https://ai.googleblog.com/2019/11/astrophotography-with-night-sight-on.html), better handling of [portraits](https://ai.googleblog.com/2018/11/learning-to-predict-depth-on-pixel-3.html), and efforts to make cameras more inclusive [so they work for all skin tones](https://store.google.com/intl/en/discover/realtone/), yield better photos that are more true to the photographer’s vision and to their subjects. Such photos can be further improved using the powerful ML-based tools now available in Google Photos, like [cinematic photos](https://ai.googleblog.com/2021/02/the-technology-behind-cinematic-photos.html), [noise and blur reduction](https://ai.googleblog.com/2021/06/take-all-your-pictures-to-cleaners-with.html), and the [Magic Eraser](https://blog.google/products/photos/magic-eraser/).
  * In addition to using their phones for creative expression, many people rely on them to help communicate with others across languages and modalities in real-time using [Live Translate](https://blog.google/products/pixel/meet-pixel-6-pixel-6-pro/) in messaging apps and [Live Caption](https://support.google.com/accessibility/android/answer/9350862?hl=en#) for [phone calls](https://blog.google/outreach-initiatives/accessibility/live-relay-phone-calls-io/). Speech recognition accuracy has continued to make substantial improvements thanks to techniques like [self-supervised learning](https://arxiv.org/abs/2010.10504) and [noisy student training](https://arxiv.org/abs/2005.09629), with marked improvements for accented speech, [noisy conditions or environments with overlapping speech](https://ai.googleblog.com/2020/11/improving-on-device-speech-recognition.html), and across many languages. Building on advances in text-to-speech synthesis, people can listen to web pages and articles using our [Read Aloud](https://youtu.be/psEX5jPkYiw) technology on a [growing number](https://blog.google/intl/en-in/company-news/outreach-initiatives/partnering-jio-help-bring-promise-internet-connectivity-and-affordability-everyone/) of [platforms](https://blog.google/products/chromebooks/accessibility-features/), making information more available across barriers of modality and languages. Live speech translations in the [Google Translate](https://blog.google/products/translate/one-billion-installs/) app have become significantly better by [stabilizing the translations](https://ai.googleblog.com/2021/01/stabilizing-live-speech-translation-in.html) that are generated on-the-fly, and high quality, robust and responsible [direct speech-to-speech translation](https://ai.googleblog.com/2021/09/high-quality-robust-and-responsible.html) provides a much better user experience in communicating with people speaking a different language. New work on combining ML with traditional codec approaches in the [Lyra speech codec](https://ai.googleblog.com/2021/02/lyra-new-very-low-bitrate-codec-for.html) and the more general [SoundStream audio codec](https://ai.googleblog.com/2021/08/soundstream-end-to-end-neural-audio.html) enables higher fidelity speech, music, and other sounds to be communicated reliably at much lower bitrate.
  * Everyday interactions are becoming much more natural with features like [automatic call screening](https://blog.google/products/pixel/phone-app-updates/) and ML agents that will [wait on hold for you](https://blog.google/products/pixel/phone-app-updates/), thanks to [advances in Duplex](https://blog.google/technology/ai/duplex-helpful-updates/). Even short tasks that users may perform frequently have been improved with tools such as [Smart Text Selection](https://ai.googleblog.com/2021/11/predicting-text-selections-with.html), which automatically selects entities like phone numbers or addresses for easy copy and pasting, and [grammar correction as you type](https://ai.googleblog.com/2021/10/grammar-correction-as-you-type-on-pixel.html) on Pixel 6 phones. In addition, [Screen Attention](https://support.google.com/pixelphone/answer/6111557?hl=en) prevents the phone screen from dimming when you are looking at it and [improvements in gaze recognition](https://ai.googleblog.com/2021/05/accelerating-eye-movement-research-for.html) are opening up new use cases for accessibility and for [improved wellness and health](https://www.nature.com/articles/s41746-021-00415-6). ML is also enabling new methods for ensuring the safety of people and communities. For example, [Suspicious Message Alerts](https://support.google.com/messages/answer/11231641?hl=en) warn against possible phishing attacks and [Safer Routing](https://blog.google/products/maps/google-maps-101-ai-power-new-features-io-2021/) detects hard-braking events to suggest alternate routes.
