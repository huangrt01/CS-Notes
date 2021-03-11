### Distributed Systems

**[l01-Introduction and MapReduce](https://github.com/huangrt01/CS-Notes/blob/master/Notes/Distributed-Systems/l01-Introduction%20and%20MapReduce)**

#### 《Apache Hadoop YARN: Yet Another Resource Negotiator》

[YARN官网](https://hadoop.apache.org/docs/stable/hadoop-yarn/hadoop-yarn-site/YARN.html)

[YARN 在字节跳动的优化与实践](https://mp.weixin.qq.com/s/9A0z0S9IthG6j8pZe6gCnw)

1.Abstract

解决两大问题：1）依赖特定编程模型 2）中心式管理难以scale

设计核心是 separating resource management functions from the programming model，更彻底地将 platform和framework抽象开

2.History

R1,R2: hadoop community

R3: HoD

R4,R5: HoD -> shared clusters

R6,R7: shared clusters

R8,R9,R10: Apache MapReduce的局限性 -> YARN

3.Architecture

RM是中心，也是连接AM和NM的桥梁

RM还能找AM要回资源

NM: container launch context (CLC)的概念，启动、监控、local service、auxiliary service

AM: late binding, the process spawned is not bound to the request, but to the lease.

fault tolerance: RM对AM有责任、AM责任最重

* YARN only secures its deployment
* RM单点故障，Work is in progress to add sufficient protocol sup-port for AMs to survive RM restart.
* 抽象分离的代价：更多fault tolerance工作交给了frameworks来设计

4.YARN in the real-world

compute resource，分布式计算资源池，跑spark、flink任务等

集群一般会配两个RM、成千上万个NM；多个RM之间用zk选主，同一时间只有一个在工作

e.g. 500000 jobs daily

一些concern：

* HDFS NameNode becomes the bottleneck
* YARN have been observed on a few large clusters with massive amounts of small applications, but recent improvements in heartbeat handling have mitigated some of these issues.


One of the most important architectural differences that partially explains these improvements is the removal of the static split between map and reduce slots.

6.Related Work

和MesOs等类似架构对比，YARN特点是multiple application frameworks、per-job intra-framework scheduler、资源管控、lazy binding等

和基于虚机的任务对比，YARN更适合运行时间短、切换更频繁的任务









理想调度算法

ideal scheduler: Multiple Knapsack Problem(MKP) (NP-C Problem)

经典生产调度算法1 - 吞吐型

* [YARN原生调度策略](https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/FairScheduler.html)

* with Quota/Priorities/DRF

  

经典生产调度算法1 - 全局视角
* 每次分配按规则对节点进行sort
* 问题1:吞吐效率问题？(scalable)
* 缓存节点排序结果(多次调度一次排序/ 抽样排序)
* 并发多个 {调度队列, 节点分区}（做hash、吞吐提升明显）
* 问题2:同一request内多个节点间的约束很难表达
  * 比如gang性、打散、集中、碎片最少等

[The evolution of cluster scheduler architectures](https://www.cl.cam.ac.uk/research/srg/netos/camsas/blog/2016-03-09-scheduler-architectures.html)

调度策略：队列

* 单集群内的物理节点被划分为多个 label
* Label 的资源总量在逻辑上被划分为多个队列
* 队列配置min/max值（允许一定的超用、max对用户隐藏）

调度策略：DRF

* 队列和应用都同时声明不止一个维度的资源 (eg. CPU, MEM)
* 在进行排序时, 如果只考虑一个维度, 可能导致其它维度碎片严重
* DRF(Dominant Resource Fairness) 将各个维度上已经使用的资源占其维度上的资源总量的百分比中最大的值作为其主资源的代表, 各个队列、应用直接对主资源代表进行比较, 进而得到排序

=> req定制Node的弱约束、req内强/弱约束



Quota管理

* 固定分组 -> 固定分组+池化quota -> 完全quota

* 可抢占任务 & 不可抢占任务
  * 可抢占任务不占用 Quota，填满空闲资源
  * 不可抢占任务占用 Quota，保证 Quota 内的任务所需资源始终可以得到满足

* Quota化带来资源碎片问题



Fragmentation

* 碎片率取决于 max_task_size/node_size，这也是为什么大家追求集群node数量的原因（YARN最多单集群5w nodes）

* 资源碎片可能导致紧急任务的资源不够，这是池化的一个副作用
* 策略：
  * Binpack 算法尽量堆叠任务，优先将一台机器资源占满 => 高优任务堆叠 + 低优任务逆向堆叠
  * 小分组问题（资源本身就少）：整机资源预留，以保证整机资源高可用；预留时可调度低优任务。



资源隔离：物理隔离

方案一：将物理机分别划入不同的集群 

* 场景：
  * 离线&流式机器隔离 
  * 单集群过大拆分
* 优势：完整独立的故障域 
* 缺点：运维成本，需要搭建独立的RM及配置的监控报警

方案二：将物理机划入同个集群内的不同Label中 

* 场景：
  * 核心作业队列隔离
  * 跨POD物理机隔离
* 优势：共用相同RM
* 缺点：共享RM全局配置, 如单心跳分配container数量



资源隔离：共享物理机隔离

* AMD机器跨numa能力弱，因此划分subnuma

![isolate-resource](Distributed-Systems/isolate-resource.png)





#### 《Mesos: A Platform for Fine-Grained Resource Sharing in the Data Center》

1.Introduction

across framework的shared cluster

two-level scheduling mechanism called resource offers. 

* 不同框架接入Mesos很方便，只需要迁移自己的scheduler

和YARN做对比：

* YARN的AM的“影响调度”的能力比Mesos弱，并不具备完全的"two-level scheduling"能力
* Mesos is an offer-based resource manager, whereas YARN has a request-based approach 感觉YARN在框架间对资源的分配会更灵活一些

3.Architecture

rejection mechanism

fault tolerance

* designed the master to be soft state, so that a new master can completely reconstruct its internal state from information held by the slaves and the framework schedulers.

4.Mesos Behavior

5.Implementation

hadoop port实现的local filesystem访问问题：

Hadoop normally writes map output files to the local filesystem, then serves these to reduce tasks using an HTTP server included in the Task-Tracker. However, the TaskTracker within Mesos runs as an executor, which may be terminated if it is not running tasks. This would make map output files unavailable to reduce tasks. We solved this problem by providing a shared file server on each node in the cluster to serve local files

6.Evaluation

overhead < 4%



Mesos 迁移 Yarn 的思路

* 阶段一：Proxy Solution: Big AM 模式，将 Yarn 调度提到一个 AM Proxy 来实现，好处是能复用原先的大部分逻辑，缺点是和 Yarn 生态有冲突，RM 的调度被“剥夺”，不利于后续资源池化推进
* 阶段二：Yarn 池化方案：AM 不再感知所有机器节点信息并独占机器，而是将资源分配交给 RM
* 阶段三：A Big AM -> An AM per Job





#### 《Design patterns for container-based distributed systems》

https://skyscribe.github.io/post/2019/07/21/from-borg-to-kubernetes/

https://kubernetes.io/

single-container patterns for container management, single-node patterns of closely cooperating containers, and multi-node patterns for distributed algorithms

容器是更通用、更贴近底层的分布式抽象手段，是代码开发到生产环境更便捷的通路

容器之于分布式就是对象之于oop，只有用容器化这一抽象把系统和应用分离开，才能基于此做很多更复杂的优化（混布提升利用率、错峰潮汐、更精细的物理资源划分）

3.Single-container management patterns

从upward和downward两方面思考

* upward
* downward
  * scale up
  * For example, Kubernetes uses a “grace-ful deletion” feature of Docker that warns a container, via the SIGTERM signal, that it is going to be termi-nated, an application-defined amount of time before it is sent the SIGKILL signal.

4.Single-node, multi-container application patterns

4.1 Sidecar pattern

可利用 [cgroups](https://www.kernel.org/doc/Documentation/cgroup-v1/cgroups.txt)

4.2 Ambassador pattern

4.3 Adapter pattern

5.Multi-node application patterns

5.1 Leader election pattern

5.2 Work queue pattern

5.3 Scatter/gather pattern

6.Related Work

* Service-oriented architectures (SOA)



#### 《Borg, Omega, and Kubernetes》

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
*  a modern container includes an image

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



#### Kubernetes

![kubernetes](Distributed-Systems/kubernetes.png)

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



#### 云原生

CNCF (Cloud Native Computing Foundation) 的定义
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



### Ray: 动态可编程分布式计算框架

1.Ray是一个动态可编程的分布式计算框架，支持分布式训练，主要体现在以下几方面：

丰富的本地训练场景（清晰指派local/remote的任务/角色，task无状态，actor有状态）

灵活训练规划pipeline（用户可以在Actor里自定义逻辑，包括循环、计时器等）

灵活数据源（流批处理融合，支持简单的数据处理chain）

One-off system：针对RL任务的特化，1）training, serving, simulation一体化，2）dynamic execution。推荐系统不一定需要，可能跟粗排联系更紧密 3）model serving的时候，更强调client and server colocate的情形，不适合我们的精排场景

2.目前工业界主要使用Ray的方式分两种，一是用Ray的上游生态lib，二是用Ray的底层能力深入自研框架

（1）Ray的上游生态：

- 强化学习库(RLLib)
- 超参调优库(Tune):  支持任意ML框架：PyTorch，XGBoost， MXNet， Keras，集成了很多优化器的库和算法， 通过TensorBoard做显示，可以和Ray Serve无缝结合
- Training with RaySGD: 优势在于能和其它Ray lib无缝结合，并且实现了分布式的dataset

（2）Ray的底层能力：大厂结合Ray自研框架的lecture在这里：[蚂蚁金服Ray Forward推广](https://tech.antfin.com/community/activities/698/review)

3.Ray对应于我们系统中的多个层次，它的底层能力对应于资源管理层REAM (包括Flink, Yarn等)，上游生态对应于我们的LagrangeX

4.最值得我们借鉴的地方有哪些？

底层：

- Ray实现了高效的系统间数据传输， "在底层hack了python的内存对象，和Redis内存共享，实现Numpy、Pandas等数据格式序列化/反序列化时候zero-copy，没有损耗"，是否可以引入这一思想减小我们model serving/training过程中序列化/反序列化的开销
- Ray的结构非常优雅，global scheduler调度全部任务，local scheduler调度同一Node内共享内存的worker，object store支持Node之间的通信。

易用性：

Ray是用户友好的分布式计算框架，具体体现在

1）轻量级的API，几乎不提高代码复杂性。只用在函数前加`@ray.remote`，就能通过remote调用，使task在其它ray集群执行。

2）支持ray dashboard，利于debugging and profiling



官方doc：https://docs.ray.io/en/latest/installation.html

开源史海钩沉系列 [1] Ray：分布式计算框架 - 高策的文章 - 知乎 https://zhuanlan.zhihu.com/p/104022670



**读论文**： 《Ray: A Distributed Framework for Emerging AI Applications》, OSDI 18

结合Bulk-synchronous parallel systems和Task-parallel systems这两类系统

无状态和有状态

- In contrast stateful computations are a good fit for implementing parameter servers, performing repeated computation on GPU-backed data, or running third-party simulators that do not expose their state.

Stateful edges

- embed actors in an otherwise stateless task graph
- enable us to maintain lineage.

![img](Distributed-Systems/ray-graph.png)

The system layer consists of three major components: a global control store, a distributed scheduler, and a distributed object store. 

![img](Distributed-Systems/ray-system.png)

- GCS让系统的任何成分都是stateless的 ，支持fault tolerance，makes it easy to scale the distributed object store and scheduler independently
- 调度的分层，有助于scaling，先由local scheduler调度，只有overload的情形，才会给global scheduler



**读源码：**https://www.qtmuniao.com/2019/07/28/ray-source-reading-1/

![img](Distributed-Systems/ray-state.png)

src/ray/common/task/scheduling_resources.cc

- ResourceSet
- ResourceIds
- ResourceIdSet
- SchedulingResources

src/ray/raylet/scheduling_queue.cc

- TaskQueue
- SchedulingQueue

src/ray/raylet/scheduling_policy.cc