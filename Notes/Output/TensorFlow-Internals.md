## TensorFlow Internals

#### chpt1: 介绍

概念：数据流图、DAG、本地设备集

DistBelief: 异步SGD（model replicas），主从结构

TensorFlow: 延迟计算、原子OP、抽象设备（CPU、GPU、ASIC）、抽象任务（基于任务的PS）


#### chpt2: 编程环境

https://www.tensorflow.org/install/source#ubuntu





##### TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems [2015]

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





