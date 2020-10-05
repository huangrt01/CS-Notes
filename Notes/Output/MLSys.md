[toc]

plethora of ML frameworks：NCCL, Horovod, BytePS, Mesh-TensorFlow, Gpipe, Ray, HugeCTR, DALI

### TensorFlow Internals

#### chpt1: 介绍

概念：数据流图、DAG、本地设备集

DistBelief: 异步SGD（model replicas），主从结构

TensorFlow: 延迟计算、原子OP、抽象设备（CPU、GPU、ASIC）、抽象任务（基于任务的PS）


#### chpt2: 编程环境

https://www.tensorflow.org/install/source#ubuntu


#### TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems [2015]

node对应operation

edge对应tensor

* 也可是control dependencies，应用：控制算法、control the peak memory usage

operations and kernels

* attributes -> make operations polymorphic 
* kernel: a particular implementation of an operation that can be run on a particular type of device (e.g., CPU or GPU)
* 定义operations and kernels: registration mechanism

Sessions: 支持Extend和Run

Variables: a special kind of opera-tion that returns a handle to a persistent mutable tensor
that survives across executions of a graph. Handles to
these persistent mutable tensors can be passed to a handful of special operations, such as `Assign` and `AssignAdd` (equivalent to +=) that mutate the referenced tensor. 



##### 3.Implementation

subgraph ~ devices  <img src="https://www.zhihu.com/equation?tex=%5Cstackrel%7B%5Cbf%7B%E5%A4%9A%E5%AF%B9%E4%B8%80%7D%7D%7B%5Clongrightarrow%7D" alt="\stackrel{\bf{多对一}}{\longrightarrow}" class="ee_img tr_noresize" eeimg="1"> workers  <img src="https://www.zhihu.com/equation?tex=%5Cstackrel%7B%5Cbf%7B%7D%7D%7B%5Clongrightarrow%7D" alt="\stackrel{\bf{}}{\longrightarrow}" class="ee_img tr_noresize" eeimg="1">  master  <img src="https://www.zhihu.com/equation?tex=%5Cstackrel%7B%5Cbf%7Bsession%7D%7D%7B%5Clongleftarrow%7D" alt="\stackrel{\bf{session}}{\longleftarrow}" class="ee_img tr_noresize" eeimg="1">  client 

device信息: the job of "the task of worker" or "localhost"

* `/job:localhost/device:cpu:0` or `/job:worker/task:17/device:gpu:3`

tensors存在backing store buffers

* Single-Device Execution: 每个node存未执行的依赖数，降为0进入ready queue
* Multi-Device Execution
  * Node Placement: cost model估算node在特定device上执行用时，simulated execution, 贪心算法
  * Cross-Device Communication: Send and Receive Nodes, 给特定tensor、特定device限制下的所有users只分配一次空间; scheduling下放给节点执行，而非master

分布式实现：per subgraph per device, TCP or RDMA

* device层面自然地形成CPU和GPU的并行

**4.Extensions**

4.1 Gradient Computation

* extending the TensorFlow graph，使heuristics可能break down
* improvements to memory management, options include:
  * using more sophisticated heuristics to determine the order of graph execution
  * recomputing tensors instead of retaining them in memory
  * swapping out long-lived tensors from GPU memory to
    more plentiful host CPU memory.

4.2 Partial Execution

* First, the Run call accepts inputs, an optional mapping
  of `name:port` names to “fed” tensors values. Second,
  the Run call accepts output names, a list of output
  `name[:port]` specifications indicating which nodes
  should be executed, and, if the port portion is present in a
  name, that that particular output tensor value for the node
  should be returned to the client if the Run call completes
  successfully.
* 根据feed node和fetch node决定partial graph

4.3 Device Constraints

* 限制的范畴：1）GPU/CPU 2)task 3)colocate with some variables
* 实现利用并查集先分析colocation，再缩小devices范围，输入到placement algorithm's simulator

4.4 Control Flow

* The Switch
  and Merge operators allow us to skip the execution of
  an entire subgraph based on the value of a boolean ten-sor. The Enter, Leave, and NextIteration operators allow
  us to express iteration.

4.5 Input Operations

* input nodes，通过文件feed，client到worker需要一个额外的network hop

4.6 Queues

* 存下数据，意义是为了prefetch from disks，或者收集梯度进行更复杂的操作

* FIFO / Shuffling Queues

4.7 Containers

* Using containers, it is possible to share state even across
  completely disjoint computation graphs associated with
  different Sessions.

**5.Optimizations**

5.1 Common Subexpression Elimination

5.2 Controlling Data Communication and Memory Usage

* e.g. 分析critical path，用control edge来delay Receive Nodes

5.3 Asynchronous Kernels

5.4 Optimized Libraries for Kernel Implementations

5.5 Lossy Compression

* 参考"Hitchhiker"论文Arithmetic Precision这节

**7.Common Programming Idioms**

同步/异步SGD

**9.Tools**

9.1 TensorBoard

9.2 Performance Tracing: EEG

#### 附录A：代码阅读

* 发现领域模型
* 抛开细枝末节： `git checkout -b code-reading`
* 适可而止，BFS阅读

### Go+Torch

https://github.com/wangkuiyi/gotorch

Q: TensorFlow为什么需要引入图这个概念

A: 

1.backward自动求导，需要定义前向的数据结构

2.python执行速度慢，决定执行效率的是图的解释器。图是python代码的另一种表示形式，开始包括前向计算过程，通过调用TensorFlow API，加入其它op包括反向计算过程和模型更新过程。构造图本质上是在编译。

* [TFRT](https://github.com/tensorflow/runtime)

调用libtorch内部的native function类比tf的op，但native function是函数，而不是一个class，每一个function可以用HLO（一种古老的适用于数值计算的语言）写一遍。gotorch调libtorch调pytorch XLA里的HLO程序，翻译成特定设备优化的代码

* native function有YAML描述，可自动生成Go Wrapper
* torchscripts：用到的python语法的子集 => python高层api可翻译成torchscripts再翻译

如果 Go+Torch 在未来一年里孕育成熟，有望优化以下核心 应用场景:

1. 统一训练和预测系统(目前训练用 Python 写，预测用 C++)
2. 统一云和端系统(目前云上用 TensorFlow，端上比如 xNN 调用 TensorFlow Lite)
3. 统一训练和预测时的数据处理流程(目前需要用 Python和C++分别做两套，开销大，而且容易出错)
4. 统一搜索、推荐、广告、金融核心、移动智能和端智能、无人驾驶等多个领域的基础架构
5. 能支持新的机器学习模式——online learning、GAN、reinforcement learning、imitation learning等。

### OneFlow: 大规模分布式深度学习框架

数据并行：allreduce + PS

模型并行：参数如何划分？复杂的通信模式

![platforms](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/MLSys/platforms.jpg)

横向拓展：片间高速互联，e.g. TPU

纵向拓展：单个芯片从通用到专用



静态调度与流式执行系统![layers](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/MLSys/layers.jpg)



OneFlow架构

* actor及流水线
  * 内存槽，用类似rust的ownership解决内存冲突问题，ownership随状态转移

![memory-pipeline](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/MLSys/memory-pipeline.jpg)

* node placement: consistent view
  * SBP, 在op层面实现数据和模型并行 
![SBP](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/MLSys/SBP.jpg)


### Ray: 动态可编程分布式计算框架

1.Ray是一个动态可编程的分布式计算框架，支持分布式训练，主要体现在以下几方面：

丰富的本地训练场景（清晰指派local/remote的任务/角色，task无状态，actor有状态）

灵活训练规划pipeline（用户可以在Actor里自定义逻辑，包括循环、计时器等）

灵活数据源（流批处理融合，支持简单的数据处理chain）

One-off system：针对RL任务的特化，1）training, serving, simulation一体化，2）dynamic execution。推荐系统不一定需要，可能跟粗排联系更紧密 3）model serving的时候，更强调client and server colocate的情形，不适合我们的精排场景

2.目前工业界主要使用Ray的方式分两种，一是用Ray的上游生态lib，二是用Ray的底层能力深入自研框架

（1）Ray的上游生态：

* 强化学习库(RLLib)

* 超参调优库(Tune):  支持任意ML框架：PyTorch，XGBoost， MXNet， Keras，集成了很多优化器的库和算法， 通过TensorBoard做显示，可以和Ray Serve无缝结合
* Training with RaySGD: 优势在于能和其它Ray lib无缝结合，并且实现了分布式的dataset

（2）Ray的底层能力：大厂结合Ray自研框架的lecture在这里：[蚂蚁金服Ray Forward推广](https://tech.antfin.com/community/activities/698/review)，还没来得及听

3.Ray对应于我们系统中的多个层次，它的底层能力对应于资源管理层REAM (包括Flink, Yarn等)，上游生态对应于我们的LagrangeX

4.最值得我们借鉴的地方有哪些？

底层：

* Ray实现了高效的系统间数据传输， "在底层hack了python的内存对象，和Redis内存共享，实现Numpy、Pandas等数据格式序列化/反序列化时候zero-copy，没有损耗"，是否可以引入这一思想减小我们model serving/training过程中序列化/反序列化的开销
* Ray的结构非常优雅，global scheduler调度全部任务，local scheduler调度同一Node内共享内存的worker，object store支持Node之间的通信。

易用性：

Ray是用户友好的分布式计算框架，具体体现在

1）轻量级的API，几乎不提高代码复杂性。只用在函数前加`@ray.remote`，就能通过remote调用，使task在其它ray集群执行。

2）支持ray dashboard，利于debugging and profiling



官方doc：https://docs.ray.io/en/latest/installation.html

开源史海钩沉系列 [1] Ray：分布式计算框架 - 高策的文章 - 知乎 https://zhuanlan.zhihu.com/p/104022670



**读论文**：
《Ray: A Distributed Framework for Emerging AI Applications》, OSDI 18

结合Bulk-synchronous parallel systems和Task-parallel systems这两类系统

无状态和有状态

* In contrast stateful computations are a good fit for implementing parameter servers, performing repeated computation on GPU-backed data, or running third-party simulators that do not expose their state.

Stateful edges

* embed actors in an otherwise stateless task graph
* enable us to maintain lineage.

<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/MLSys/ray-graph.png" alt="" style="zoom:70%;" />

The system layer consists of three major components: a global control store, a distributed scheduler, and a distributed object store. 

<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/MLSys/ray-system.png" alt="" style="zoom:50%;" />

* GCS让系统的任何成分都是stateless的 ，支持fault tolerance，makes it easy to scale the distributed object store and scheduler independently

* 调度的分层，有助于scaling，先由local scheduler调度，只有overload的情形，才会给global scheduler



**读源码：**https://www.qtmuniao.com/2019/07/28/ray-source-reading-1/

<img src="https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/MLSys/ray-state.png" alt="" style="zoom:50%;" />

src/ray/common/task/scheduling_resources.cc

* ResourceSet
* ResourceIds
* ResourceIdSet
* SchedulingResources

src/ray/raylet/scheduling_queue.cc

* TaskQueue
* SchedulingQueue

src/ray/raylet/scheduling_policy.cc

### 论文阅读

已读，待整理：

#### 《MLSys: The New Frontier of Machine Learning Systems》

#### 《Deep Neural Networks for Youtube Recommendations, RecSys 16》

#### 《Wide & Deep learning for Recommender Systems, RecSys 17》

#### 《A Hitchhiker's Guide On Distributed Training Of Deep Neural Networks, JPDC 18》

#### 《TFX: A TensorFlow-based production-scale machine learning platform》

#### 《TensorFlow: A system for large-scale machine learning, OSDI 16》

#### 《Clipper: A Low-Latency Online Prediction Serving System, NSDI 17》

low latencies, high throughputs, and improved accuracy

prediction cache, batching queue

##### Model abstraction layer

用object store存模型，减少初始化开销

prediction cache：本质上类似SHARED属性（同一batch内的某一特征用相同的预估结果）。两者的区别在于，前者的输入更简单，以模型和req id为标识，易于做cache操作；后者是feature层面，更精细。推荐系统入图的特征输入很难做到完全一致，因此做prediction cache操作难度较大。

batching：动态选batch size的方式
* additive-increase-multiplicative-decrease (AIMD) scheme 
* quantile regression
* delayed batching：按攒batch的timeout来delay，适合并行优化明显的模型

model container: 无状态服务
* Clipper performs adaptive batching independently for each replica

##### Model selection layer

动态调整选用模型的策略，推荐系统采用这类技术比CV/NLP难度更大

* Single Model Selection Policy
  * address the trade-off between exploring possible actions and exploiting the estimated best action. 
* Ensemble Model Selection Policies
  * Robust Prediction 
    * agreement衡量prediction confidence 
    * 有针对degraded模型的降级机制
  * Straggler Mitigation
* Contextualization: instantiate a unique model selection state for each user, context, or session.



#### 《Hidden Technical Debt in Machine Learning Systems, NIPS 15》

boundary erosion, entanglement, hidden feedback loops, undeclared consumers, data dependencies, configuration issues, changes in the external world, and a variety of system-level anti-patterns.

2. Complex Models Erode Boundaries
* Entanglement: 即使多模型/超参的配置独立，但效果互相影响
* Correction Cascade: 模型级联是hidden debt
* Undeclared Consumers: 需要SLA(service-level agreement)

3. Data Dependencies Cost More than Code Dependencies
* Underutilized dependencies: legacy/bundled/epsilon/correlated, use exhaustive leave-one-feature-out evaluations to detect

4. Feedback Loops
* direct: related to bandit algorithms, costly
* hidden: two independent systems may interact

5. ML-System Anti-Patterns
* Glue Code: hard to achieve a domain-specific goal
* Pipeline Jungle: 特征工程的意义所在，thinking holistically about data collection and feature ex traction
* Dead Experimental Codepaths
* Abstraction Debt
* Common Smells

6. Configuration Debts
* Feature A was incorrectly logged from 9/14 to 9/17
* Feature B is not available on data before 10/7
* The code used to compute feature C has to change for data before and after 11/1 because of changes to the logging format
* Feature D is not available in production, so a substitute features D′ and D′′ must be used when querying the model in a live setting
* If feature Z is used, then jobs for training must be given extra memory due to lookup tables or they will train inefficient
* Feature Q precludes the use of feature R because of latency constraints.

7. Dealing with Changes in the External World




#### 《Ad Click Prediction: a View from the Trenches, KDD 13》
* a high-dimensional visualization tool was used to allow researchers to quickly see effects across many dimensions and slicings
* enables data sources and features to be annotated. Automated checks can then be run to ensure that all dependencies have the appropriate annotations, and dependency trees can be fully resolved.

#### 《XDL: An industrial deep learning framework for high-dimensional sparse data, KDD 19》

MPI(All Reduce)和PS，两种分布式计算方向

Sparse + Dense

* SparseNet: Representa-tion learning which captures information from high-dimensional sparse input and embeds them into a low-dimensional space

* DenseNet: Function fitting which models the relationship between dense em- bedding representation and supervised label

In order to facilitate deployment on various computing platforms,
XDL can be scheduled by multiple resource management platform, like Yarn, and provides data I/O interfaces to various data storage systems, like HDFS and Kafka.



* I/O
  * Hierarchical sample compression: prefix tree

![prefix-tree](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/MLSys/prefix-tree.jpg)

* Workflow pipeline

  * I/O: read sample and group mini-batch -> prefetch (maybe cudaMemcpyAsync()) -> pull/forward/backward/push
  * SparseNet and DenseNet

* Optimization for Advanced Model Server

  * Network: [Seastar](https://github.com/scylladb/seastar) + zero-copy/CPU-binding

* Online Learning with XDL

  * Feature Entry Filter
  * Incremental Model Export
  * Feature Expire

#### 《Ethane: Taking control of the enterprise, SIGCOMM 2007》

make networks more manageable and more secure，一种思路是全方位的增加控制，相当于新增一层，只是hide了复杂度；于是提出ethane

ethane的思想：
* The network should be governed by policies declared over high-
level names
* Policy should determine the path that packets follow
* The network should enforce a strong binding between a packet
and its origin.

Ethane的优势：
* Security follows management.

* Incremental deployability.

* Significant deployment experience.
  
  
  
  
  