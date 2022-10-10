## TensorFlow

[toc]

### Op 学习

#### 环境配置

* python2.7 + tf1.15.0 + cuda10.0
  * 如果想 python3.7 + tf1.15.3 + cuda10.1/11，可以使用 [nvidia-tensorflow](https://github.com/NVIDIA/tensorflow)
  * 常规操作：编译用 cuda10.1，运行用 cuda10.0

```shell
export PATH="/usr/local/cuda-10.0/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"
```

```python
import tensorflow as tf
tf.test.is_gpu_available()
```

* 查看机器配置

```shell
nvidia-smi
nvcc --version
```

* 关于 tf 的 ABI
  * 社区版的 tf 要用 abi=0 的 lib.so；自己编译的默认 abi=1，要用 abi=0 的 lib.so
* FAQ
  * unsupported GNU version! gcc versions later than 7 are not supported
    * cuda 版本要用 10.1 以上的
  * `ValueError: Multiple enum values: 3`

```shell
$ pip uninstall enum   # 如果报错则直接下一步
$ pip install enum34==1.1.10
```

#### Perf

##### 浮点运算量

* [Calculate FLOPs and Number of Model Parameters]([https://github.com/anakin1028/Tensorflow-Examples/blob/master/notebooks/basic/Calculate%20FLOPs%20and%20Number%20of%20Model%20Parameters.ipynb](https://github.com/anakin1028/Tensorflow-Examples/blob/master/notebooks/basic/Calculate FLOPs and Number of Model Parameters.ipynb))
  * `saver = tf.train.import_meta_graph('./tmp/model.ckpt.meta')`
  * `W = tf.Variable(tf.random_normal([20, 8, 1, 64]))`
    * python/ops/random_ops.py

```python
import tensorflow as tf
tf.reset_default_graph()

# case1 normal save and restore
# define simple graphs
a = tf.Variable([3.], dtype=tf.float32, name='a')
b = tf.placeholder(tf.float32, shape=(), name='input')
# In this grpah we have three FLOPs
c = tf.multiply(a, b, name='wawa')
d = tf.multiply(c, c, name='tata')
e = tf.multiply(d, d, name='haha')
# In tf1, we need to initialize variables manually
init = tf.global_variables_initializer()

# session will bind to the global default graph
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(d, feed_dict={b:2}))
    saver.save(sess, './tmp/model.ckpt')
# then under the directory ./tmp you will find the two files
# model.ckpt.meta : The definition of graph
# model.ckpt.data-00000-of-00001 : The data (the value for the nodes)

# here we want to analyze the floating point opearation numbers
tf.reset_default_graph()
saver = tf.train.import_meta_graph('./tmp/model.ckpt.meta')
# saver = tf.train.Saver()
with tf.Session() as sess:
    # The session is binding to the default global graph
    tf.profiler.profile(
        sess.graph,
        options=tf.profiler.ProfileOptionBuilder.float_operation())

with tf.Session() as sess:
    # The session is binding to the default global graph
    parameters = tf.profiler.profile(
        sess.graph,
        options=tf.profiler.ProfileOptionBuilder
        .trainable_variables_parameter())
    print ('total parameters: {}'.format(parameters.total_parameters))



tf.reset_default_graph()

# for simplicity we consider the first conv layer

# You can think X as 1 example, 32 timestamps, spectral components for 40 mel-bands, and one input channel
# And typically TF call this as NHWC format
X = tf.placeholder(tf.float32, [1, 32, 40, 1])
# H:20, W:8, Input Channel: 1, Output Channel 64
W = tf.Variable(tf.random_normal([20, 8, 1, 64]))
b = tf.Variable(tf.random_normal([64]))
conv1 = tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='VALID')
conv1 = tf.nn.bias_add(conv1, b)
conv1 = tf.nn.max_pool(conv1, ksize=[1, 1, 3, 1], strides=[1,1,1,1], padding='VALID')

# now we have defined our graph, we can calculate the FLOPs and number of
# parameters

with tf.Session() as sess:
    with tf.Session() as sess:
        # The session is binding to the default global graph
        tf.profiler.profile(
            sess.graph,
            options=tf.profiler.ProfileOptionBuilder.float_operation())
        parameters = tf.profiler.profile(sess.graph,
                                         options=tf.profiler.ProfileOptionBuilder
                                         .trainable_variables_parameter())
        print ('total parameters: {}'.format(parameters.total_parameters))
    

# observe the output of this cell: the counts of parameter is indeed 10.2K!
```



### TensorFlow Internals

#### chpt1: 介绍

概念：数据流图、DAG、本地设备集

DistBelief: 异步SGD（model replicas），主从结构

TensorFlow: 延迟计算、原子OP、抽象设备（CPU、GPU、ASIC）、抽象任务（基于任务的PS）


#### chpt2: 编程环境

https://www.tensorflow.org/install/source#ubuntu

#### 代码阅读

[tensorflow源码解析-阿里云文章](https://developer.aliyun.com/profile/x6ehajer74kvo/highScore_1?spm=a2c6h.13262185.profile.4.139d6b06roaEa7)

[tensorflow源码解析-cnblogs](https://www.cnblogs.com/jicanghai/default.html?page=3)

* 发现领域模型
* 抛开细枝末节： `git checkout -b code-reading`
* 适可而止，BFS阅读


##### 代码风格

* TF_GUARDED_BY、TF_EXCLUSIVE_LOCKS_REQUIRED
  * tsl/platform/thread_annotations.h
  * [现代C++开发之线程安全注解 by 陈硕](https://zhuanlan.zhihu.com/p/47837673)

##### core/platform/refcount.h

```c++
// Helper class to unref an object when out-of-scope.
class ScopedUnref {
 public:
  explicit ScopedUnref(const RefCounted* o) : obj_(o) {}
  ~ScopedUnref() {
    if (obj_) obj_->Unref();
  }

 private:
  const RefCounted* obj_;

  ScopedUnref(const ScopedUnref&) = delete;
  void operator=(const ScopedUnref&) = delete;
};
```



##### core/platform/default/logging.h

DCHECK_EQ





tensor

IsInitialized()

RefCountIsOne()

##### cc/saved_model/loader.cc

```c++
// We disallow calls to Session::Extend() on the returned session, so we can
// reduce memory consumption by not storing the original GraphDef.
rewritten_options.config.mutable_experimental()
  ->set_optimize_for_static_graph(true);
```



#### allocator

https://www.cnblogs.com/jicanghai/p/9535808.html

参考 cpu_allocator_impl.c 注册

`REGISTER_MEM_ALLOCATOR("DefaultCPUAllocator", 100, CPUAllocatorFactory);`

##### tsl/framework/allocator_registry.*

* AllocatorFactoryRegistry

```c++
// 宏的写法，用到了 __COUNTER__，保证register时命名不重复
// CPUAllocator的priority是100，可以设一个大于它的值覆盖默认项
#define REGISTER_MEM_ALLOCATOR(name, priority, factory)                     \
  REGISTER_MEM_ALLOCATOR_UNIQ_HELPER(__COUNTER__, __FILE__, __LINE__, name, \
                                     priority, factory)
```





#### 基本 API 用法

##### Import Graph

```python
graph = tf.Graph()
with graph.as_default():
  tf.import_graph_def(graph_def, name='')
  loss = graph.get_tensor_by_name(loss_name)
  label = graph.get_tensor_by_name(label_name)
  pred = graph.get_tensor_by_name(pred_name)
  _, auc = tf.metrics.auc(label > 0.5, pred)
	sess = tf.Session(graph=graph)
	sess.run(tf.local_variables_initializer())
```

##### option

```python
tf.config.threading.set_inter_op_parallelism_threads(args.tf_cpu_count)
tf.config.threading.set_intra_op_parallelism_threads(args.tf_cpu_count)


```

##### build tensor and run session

https://www.tensorflow.org/api_docs/python/tf/compat/v1/Session?hl=zh-cn#run

```python
tensor = tensor_pb2.TensorProto()
if isinstance(value, str):
  value = value.encode()
tensor.ParseFromString(value)
inputs[name] = tf.make_ndarray(tensor)
fetch1, fetch2, fetch3, fetch4 = sess.run(fetches, feed_dict=inputs)
proto_fetch1 = {}
for name, value in fetch1.items():
	proto_fetch1[name] = tf.make_tensor_proto(value).SerializeToString()
```



##### gradients

```python
# gradients 返回 grad_ys*dy/dx，ys和xs是tensor
with tf.name_scope(""):
        with tf.name_scope("Gradients/%s/" % scope_name):
            gs = tf.gradients(ys=ys, xs=xs, name=name, colocate_gradients_with_ops=colocate_gradients_with_ops, grad_ys=grad_ys)
```

[about grad_ys](https://stackoverflow.com/questions/50967885/tf-gradients-how-can-i-understand-grad-ys-and-use-it)



[Understanding variable_scope and name_scope in tensorflow and variable sharing](https://stackoverflow.com/questions/36237427/understanding-variable-scope-and-name-scope-in-tensorflow-and-variable-sharing)

```python
def forward(inputs):
    init = tf.random_normal_initializer()
    w = tf.get_variable("weights", shape=(3,2), initializer=init)
    return tf.matmul(w, inputs)

with tf.name_scope("group_1"):
    a = tf.placeholder(tf.float32, shape=(2, 3), name="a")
    b = tf.placeholder(tf.float32, shape=(2, 3), name="b")
    c = tf.placeholder(tf.float32, shape=(2, 3), name="c")
    with tf.variable_scope("foo", reuse=False):
        aa = forward(a)
    with tf.variable_scope("foo", reuse=True):
        bb = forward(b)
        cc = forward(c)

with tf.name_scope("group_2"):
    d = tf.placeholder(tf.float32, shape=(2, 3), name="d")
    with tf.variable_scope("foo", reuse=True):
        dd = forward(d)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print(bb.eval(feed_dict={b: np.array([[1,2,3],[4,5,6]])}))
    for var in tf.global_variables():
        print(var.name)
        print(var.eval())
```

```python
tf.split(
    value, num_or_size_splits, axis=0, num=None, name='split'
)
# num_or_size_splits，既可以传入"N"等分，也可以传入每份的 size list
```

**mask input**

[consumers](https://www.kite.com/python/docs/tensorflow.contrib.graph_editor.SubGraphView.consumers)

```python
zero_tensor = tf.zeros_like(
	slice_tensor, name=normalize_tensor_name + "_mask")
normalize_zero_tensor_name = zero_tensor.name.split(':')[0]
consumers = [con for con in tensor.consumers() 
             if con.name != normalize_zero_tensor_name]
consumers_indices = {}
for consumer in consumers:
	consumers_indices[consumer] = [i for i, t in enumerate(consumer.inputs) if t is tensor]
for consumer in consumers:
	for i in consumers_indices[consumer]:
		consumer._update_input(i, zero_tensor)
```





#### TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems [2015]

node对应operation

edge对应tensor

* 也可是control dependencies，应用：控制算法、control the peak memory usage

operations and kernels

* attributes -> make operations polymorphic 
* kernel: a particular implementation of an operation that can be run on a particular type of device (e.g., CPU or GPU)
* 定义operations and kernels: registration mechanism

Sessions: 支持Extend和Run

Variables: a special kind of operation that returns a handle to a persistent mutable tensor that survives across executions of a graph. Handles to these persistent mutable tensors can be passed to a handful of special operations, such as `Assign` and `AssignAdd` (equivalent to +=) that mutate the referenced tensor. 

##### 3.Implementation

subgraph ~ devices  <img src="https://www.zhihu.com/equation?tex=%5Cstackrel%7B%5Cbf%7B%E5%A4%9A%E5%AF%B9%E4%B8%80%7D%7D%7B%5Clongrightarrow%7D" alt="\stackrel{\bf{多对一}}{\longrightarrow}" class="ee_img tr_noresize" eeimg="1"> workers  <img src="https://www.zhihu.com/equation?tex=%5Cstackrel%7B%5Cbf%7B%7D%7D%7B%5Clongrightarrow%7D" alt="\stackrel{\bf{}}{\longrightarrow}" class="ee_img tr_noresize" eeimg="1">  master  <img src="https://www.zhihu.com/equation?tex=%5Cstackrel%7B%5Cbf%7Bsession%7D%7D%7B%5Clongleftarrow%7D" alt="\stackrel{\bf{session}}{\longleftarrow}" class="ee_img tr_noresize" eeimg="1">  client 

device信息: the job of "the task of worker" or "localhost"

* `/job:localhost/device:cpu:0` or `/job:worker/task:17/device:gpu:3`

Tensor backing store buffers are reference counted

Execution

* Single-Device Execution: 每个 node 存未执行的依赖数，降为0进入ready queue
* Multi-Device Execution
  * Node Placement: cost model 估算 node 在特定 device 上执行用时，simulated execution, 贪心算法
  * Cross-Device Communication: Send and Receive Nodes, 给特定tensor、特定device限制下的所有users只分配一次空间; scheduling下放给节点执行，而非master

分布式实现：per subgraph per device, TCP or RDMA

* device层面自然地达成 CPU & GPU 并行计算

**4.Extensions**

4.1 Gradient Computation

* 如果 extend the TensorFlow graph，自动地加入 gradient tensors，那么关于 tensor 使用位置/先后顺序 预测的 heuristic 可能break down，最先使用的 tensor 到最后依然需要使用
* improvements to memory management, options include
  * using more sophisticated heuristics to determine the order of graph execution
  * recomputing tensors instead of retaining them in memory
  * swapping out long-lived tensors from GPU memory to more plentiful host CPU memory.

4.2 Partial Execution

* First, the Run call accepts inputs, an optional mapping of `name:port` names to “fed” tensors values. Second, the Run call accepts output names, a list of output `name[:port]` specifications indicating which nodes should be executed, and, if the port portion is present in a name, that that particular output tensor value for the node should be returned to the client if the Run call completes successfully.
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

* e.g. 分析 critical path，用 control edge 来 delay Receive Nodes

5.3 Asynchronous Kernels

5.4 Optimized Libraries for Kernel Implementations

5.5 Lossy Compression

* 参考"Hitchhiker"论文Arithmetic Precision这节

**7.Common Programming Idioms**

同步/异步SGD

**9.Tools**

9.1 TensorBoard

9.2 Performance Tracing: EEG



#### TensorFlow: A system for large-scale machine learning [OSDI, 2016]

**Introduction**

* TensorFlow allows vertices to represent computations that own or update mutable state.
* synchronous replication

While MXNet partially fulfills our extensibility requirements, the parameter server is “privileged” code, which makes it difficult for researchers to customize the handling of large models

**3.TensorFlow execution model**

Dataflow with mutable state 是tf吸取PS架构的经验 

几种训练方式的讨论

* 同步：大并发、无gradient staleness、scalable
* 异步：资源利用率高 (maintain high throughput in the presence of
  stragglers)；可以只使用一部分worker的梯度做更新，虽然损失了信息，但减少了异步带来的冲突
* 半同步：dense同步、sparse异步



**4.3 Fault tolerance**

Having checkpoint and parameter management as programmable operations in the graph gives users the flexibility to implement schemes like these and others that we have not anticipated.



**4.4 Synchronous replica coordination**

synchronous with backup workers，和MapReduce的backup方案对比，更 proactive



原生tensorflow架构分析：

* 优点：
  * 无需开发PS
    * 实现需要额外存储变量的op在原生tf更为简单
    * 新optimizer的探索不需要单独部署PS

* 缺点：
  * distributed runtime有通信问题，每个slot产生一对send/recv op，对于大规模embedding的场景基本训不动模型



### TensorFlow Serving

#### 《TensorFlow-Serving: Flexible, High-Performance ML Serving》



load模型（尤其是对模型进行warmup）导致延迟spike的问题，确实不容易解决。特别复杂的模型warm up引起服务cpu抖动，可能是因为线程不够了

2.Library

2.1 Model Lifecycle Management

* Source, Source Adapters, Source Routers
* Canary and Rollback
* Aspired Versions Manager
  * RCU

2.2 Inference

* Inter-Request Batching（图内/外）

3.Canonical Binary and Hosted Service



[How Zendesk Serves TensorFlow Models in Production](https://medium.com/zendesk-engineering/how-zendesk-serves-tensorflow-models-in-production-751ee22f0f4b)

[美团：基于TensorFlow Serving的深度学习在线预估](https://tech.meituan.com/2018/10/11/tfserving-improve.html)