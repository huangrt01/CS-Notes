[toc]

### 使用相关

#### Tf编译构建

* build tf2.4 from source （debian 6.3.0失败）
  * https://www.tensorflow.org/install/source
  * 先用的编译op报错，可能和用的是gcc6而不是gcc8有关，也可能和内存有关
    * [ref](https://github.com/tensorflow/tensorflow/issues/349)
  * 转向下面的docker

```shell
virtualenv .myenv --python=python3
source .myenv/bin/activate
pip install pip numpy wheel packaging requests opt_einsum
pip install keras_preprocessing --no-deps

./configure
bazel build -c opt //tensorflow/tools/pip_package:build_pip_package --local_ram_resources 2048
```

* 用docker：https://www.tensorflow.org/install/source#docker_linux_builds
  * docker镜像：https://hub.docker.com/r/tensorflow/tensorflow/
  * 如果不安装git，会报错 “An error occurred during the fetch of repository 'io_bazel_rules_docker'”
    * [ref](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/issues/940)
  * 注意docker是python3.6的环境，pip install也需要在python3.6的venv下进行

```shell
docker pull tensorflow/tensorflow:2.4.0

docker container run -itd \
--name $(whoami)_tf_dev \
--network=host \
--mount type=bind,source= <img src="https://www.zhihu.com/equation?tex=%28pwd%29%2Ctarget%3D" alt="(pwd),target=" class="ee_img tr_noresize" eeimg="1"> (pwd) \
--mount type=bind,source= <img src="https://www.zhihu.com/equation?tex=HOME/.ssh%2Ctarget%3D" alt="HOME/.ssh,target=" class="ee_img tr_noresize" eeimg="1"> HOME/.ssh \
--mount type=bind,source= <img src="https://www.zhihu.com/equation?tex=HOME/.cache/bazel%2Ctarget%3D" alt="HOME/.cache/bazel,target=" class="ee_img tr_noresize" eeimg="1"> HOME/.cache/bazel \
    -e HOST_PERMS=" <img src="https://www.zhihu.com/equation?tex=%28id%20-u%29%3A" alt="(id -u):" class="ee_img tr_noresize" eeimg="1"> (id -g)" tensorflow/tensorflow:2.4.0 bash
    
docker exec -it $(whoami)_tf_dev /bin/bash
    
# 下载或挂载bazel-3.1.0

apt update
apt install git

bazel build --config=opt -c opt //tensorflow/tools/pip_package:build_pip_package

./bazel-bin/tensorflow/tools/pip_package/build_pip_package .
chown $HOST_PERMS tensorflow-version-tags.whl
sudo pip install tensorflow-version-tags.whl
```

* IDE配置
  * 参考【Compiling】-bazel-ide
  
* 编译优化
  * 有些需要手动开，包括xla这些

```
build --copt=-O3
build --copt=-mavx 
build --copt=-mavx2 
build --copt=-mfma 
build --copt=-msse4.1 
build --copt=-msse4.2
```

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
  * 社区版的 tf 要用 abi=0 的 lib.so；自己编译的默认 abi=1，要用 abi=1 的 lib.so
* FAQ
  * import tf时报错 tensorflow version `GLIBC_2.27' not found
    * `strings /lib/x86_64-linux-gnu/libm.so.6 |grep GLIBC` 查看系统支持的glibc版本
    * tf2必须要高版本的glibc，只能用docker跑了
  
  * unsupported GNU version! gcc versions later than 7 are not supported
    * cuda 版本要用 10.1 以上的
  * `ValueError: Multiple enum values: 3`
  

```shell
$ pip uninstall enum   # 如果报错则直接下一步
$ pip install enum34==1.1.10
```

#### 写Op的编译

* 利用Bazel + Tf source code
  * 先进入能编译tf的docker

```shell
cd tensorflow/core/user_ops


### BUILD File
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)
###

bazel build --config opt //tensorflow/core/user_ops:zero_out.so
cd ..
cp tensorflow/bazel-bin/tensorflow/core/user_ops/zero_out.so .
python

### python
import tensorflow as tf
zero_out_module = tf.load_op_library('./zero_out.so')
print(zero_out_module.zero_out([[1, 2], [3, 4]]).numpy())
###
```

```python
# zero_out_op_test.py
import tensorflow as tf

class ZeroOutTest(tf.test.TestCase):
  def testZeroOut(self):
    zero_out_module = tf.load_op_library('./zero_out.so')
    with self.cached_session():
      result = zero_out_module.zero_out([5, 4, 3, 2, 1])
      self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])

if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
```

* op编译的原理
  * A shared object which includes registration mechanisms for ops and kernels. Does not include the implementations of any ops or kernels. Instead, the library which loads libtensorflow_framework.so (e.g. `_pywrap_tensorflow_internal.so` for Python, `libtensorflow.so` for the C API) is responsible for registering ops with `libtensorflow_framework.so`. In addition to this core set of ops, user libraries which are loaded (via `TF_LoadLibrary`/`tf.load_op_library`) register their ops and kernels with this shared object directly.
  * For example, from Python `tf.load_op_library` loads a custom op library (via dlopen() on Linux), the library finds libtensorflow_framework.so (no filesystem search takes place, since libtensorflow_framework.so has already been loaded by pywrap_tensorflow) and registers its ops and kernels via REGISTER_OP and REGISTER_KERNEL_BUILDER (which use symbols from libtensorflow_framework.so), and pywrap_tensorflow can then use these ops. Since other languages use the same libtensorflow_framework.so, op libraries are language agnostic.
  * modular op registration support: framework_shared_object=true (set in the configure script unconditionally);
    * otherwise if it is false or undefined, the build is static and TensorFlow symbols (in Python only) are loaded into the global symbol table in order to support op registration. This means that projects building with Bazel and importing TensorFlow as a dependency will not depend on libtensorflow_framework.so unless they opt in.



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

### 代码阅读

```shell
git clone https://github.com/tensorflow/tensorflow.git
gco 582c8d # 2.4.0
cloc .
```

* 276w行代码，其中140w C++ code，68w python code



#### [gdb辅助读代码](https://jcf94.com/download/TensorFlow-SourceCode-Reading.pdf)

* dbg模式安装tf `bazel build --config=dbg //tensorflow/tools/pip_package:build_pip_package`

```shell
import tensorflow as tf
import os
os.getpid()
tf.compat.v1.disable_eager_execution()
c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
sess = tf.compat.v1.Session()
sess.run(c)
```

```
gdb -p 12345
b TF_NewBuffer
c
b tensorflow::(anonymous namespace)::ExecutorImpl::RunAsync
c
```



[tensorflow源码解析-阿里云文章](https://developer.aliyun.com/profile/x6ehajer74kvo/highScore_1?spm=a2c6h.13262185.profile.4.139d6b06roaEa7) TODO

[tensorflow源码解析-cnblogs](https://www.cnblogs.com/jicanghai/default.html?page=3) TODO

### python

* Python 定义和实现了 TensorFlow 的编程模型，并对外开放 API 供程序员使用
  * 62w行python

```
tensorflow/python
|-- autograph
|   |-- converters
|   |-- core
|   |-- g3doc
|   |-- impl
|   |-- lang
|   |-- operators
|   |-- pyct
|   `-- utils
|-- client
|-- compat
|-- compiler
|   |-- mlir
|   |-- tensorrt
|   `-- xla
|-- data
|   |-- benchmarks
|   |-- experimental
|   |-- kernel_tests
|   |-- ops
|   `-- util
|-- debug
|   |-- cli
|   |-- examples
|   |-- lib
|   `-- wrappers
|-- distribute
|   |-- cluster_resolver
|   |-- coordinator
|   |-- experimental
|   |-- integration_test
|   |-- parallel_device
|   `-- v1
|-- dlpack
|-- eager
|   |-- benchmarks
|   `-- memory_tests
|-- estimator
|   |-- canned
|   |-- export
|   `-- inputs
|-- feature_column
|   `-- testdata
|-- framework
|   |-- experimental
|   `-- testdata
|-- grappler
|-- integration_testing
|-- keras
|   |-- api
|   |-- applications
|   |-- benchmarks
|   |-- datasets
|   |-- distribute
|   |-- engine
|   |-- estimator
|   |-- feature_column
|   |-- initializers
|   |-- integration_test
|   |-- layers
|   |-- legacy_tf_layers
|   |-- mixed_precision
|   |-- optimizer_v2
|   |-- premade
|   |-- preprocessing
|   |-- protobuf
|   |-- saving
|   |-- tests
|   |-- type
|   |-- utils
|   `-- wrappers
|-- kernel_tests
|   |-- array_ops
|   |-- boosted_trees
|   |-- distributions
|   |-- linalg
|   |-- proto
|   |-- random
|   |-- signal
|   |-- testdata
|   `-- v1_compat_tests
|-- layers
|-- lib
|   |-- core
|   `-- io
|-- lite
|-- module
|-- ops
|   |-- distributions
|   |-- linalg
|   |-- losses
|   |-- numpy_ops
|   |-- parallel_for
|   |-- ragged
|   |-- signal
|   |-- structured
|   `-- v1_compat_tests
|-- platform
|-- profiler
|   |-- integration_test
|   `-- internal
|-- saved_model
|   `-- model_utils
|-- summary
|   `-- writer
|-- tf_program
|   `-- tests
|-- tools
|   `-- api
|-- tpu
|   |-- client
|   |-- experimental
|   |-- ops
|   `-- profiler
|-- training
|   |-- experimental
|   |-- saving
|   `-- tracking
|-- types
|-- user_ops
`-- util
    `-- protobuf
```

#### session.py

* pybind11桥接C++和python
  * pip安装后，进入 `.myenv/lib/python3.6/site-packages/tensorflow`
  * python/_pywrap_tensorflow_internal.so
    * -> pywrap_tf_session
  * python/pywrap_tensorflow.py: Python 通过 pybind11 来调用 C 和 C++ 的运行库
    * load _pywrap_tensorflow_internal.so
  * python/BUILD:  `pywrap_tensorflow_macro(name = "pywrap_tensorflow_internal"`
* pywrap_tf_session 怎么来的
  * client/pywrap_tf_session.py 将 _pywarp_tf_session.so 打包成 py
  * client/BUILD: 
    * `tf_python_pybind_extension(name = "_pywrap_tf_session",...`)
  * client/tf_session_wrapper.cc
    * `m.def("TF_NewBuffer", TF_NewBuffer, py::return_value_policy::reference);`
    * `m.def("_TF_NewSessionOptions", TF_NewSessionOptions, py::return_value_policy::reference, py::call_guard<py::gil_scoped_release>());`
    * ---> c/c_api.cc
* session.py
  * `class BaseSession(SessionInterface):` 
    * as_default: ops.default_session
  * Session基于BaseSession，增加了：
    * _default_graph_context_manager
    * _default_session_context_manager
  * run()函数
    * 有详细的session run注释
    * `_FetchHandler` 由fetches到targets（如果fetch的类型是op，则认为是target）
  * `SessionRef` blocks the return of Close() until all pending operations have been completed or cancelled and underlying session has been freed.
  * 调用 c/c_api.cc: 
    * Tf_NewSession
    * TF_SessionRun_wrapper -> TF_SessionRun_wrapper_helper
      * python obj和C++ obj之间的转换
      * 追踪到 tensorflow::DirectSession::Run()

```
# session.py
opts = tf_session.TF_NewSessionOptions(target=self._target, config=config)
    try:
      # pylint: disable=protected-access
      self._session = tf_session.TF_NewSessionRef(self._graph._c_graph, opts)
```



#### data

https://www.tensorflow.org/guide/data

* tf.data: A Machine Learning Data Processing Framework https://arxiv.org/pdf/2101.12127.pdf MUSTDO

  * `train_data = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)`

  * `image_batch, label_batch = next(iter(train_data))`

* ops/dataset_ops.py
  * DataSetV2 支持 shuffle、batch、repeat、map 等操作
    * shuffle(buffer_size)
    * batch(batch_size, drop_remainder=False)
    * map(parser_fn, num_parallel_calls=tf.data.AUTOTUNE)
    * prefetch(tf.data.experimental.AUTOTUNE)

```python
# 自定义dataset
from tensorflow.python.data.ops import dataset_ops

class MyDataset(dataset_ops.DatasetSource):

  def __init__(self, **kwargs):
    ...
    variant_tensor = my_datasource_ops.my_dataset(**kwargs)
    super(MyDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], self._dtype)
```

```python
# TFRecordDataset
def get_tfrecord_dataset(paths,
                         feature_description,
                         batch_size,
                         thread_num,
                         prefetch_buffer_size,
                         buffer_size):
    dataset = tf.data.TFRecordDataset(paths,
                                      buffer_size=buffer_size,
                                      num_parallel_reads=thread_num)
    dataset = dataset.batch(batch_size).map(
        partial(parse_example,
                feature_description=feature_description),
        num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
        prefetch_buffer_size)
    return dataset
```







#### distribute

* parameter_server_strategy_v2
  * https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/ParameterServerStrategy
  * _create_variable in round_robin fashion
    * Creates a `Variable` or a `ShardedVariable`

* parameter_server_strategy
  * `device_setter.replica_device_setter`: place variables on ps jobs in a round-robin fashion.
  * _make_dataset_iterator
  * _make_input_fn_iterator
* embedding_ops
  * 根据id查询sharded_variable，在dim1支持pooling操作，id的含义是tensor list的index
  * 调用了 Unique C++ Op：`ids, idx = array_ops.unique(ids)`
  * segment_ids: [0, 0, 1, 1, 2, 3]
* shared_variable
  * Partitioner
    * FixedShardsPartitioner
    * MinSizePartitioner 和 MaxSizePartitioner，分配尽量多或尽量少的shards
  * 定义了一个mixin类，ShardedVariableMixin
    * https://en.wikipedia.org/wiki/Mixin
    * Variables should not be shared between different `ShardedVariableMixin` objects.
    * We create an uninitialized saving_variable with the full shape, which can be later captured in signatures so that the signatures can treat this ShardedVariable as one single variable.
    * 支持了 assign、assign_add、assign_sub方法
  * `class ShardedVariable(ShardedVariableMixin, composite_tensor.CompositeTensor):`
    * one should generally not assume it has the same number of shards across save and load.

```python
class ShardedVariableSpec(type_spec.TypeSpec):
  """Type specification for a `ShardedVariable`."""

  __slots__ = ['_variable_specs']

  value_type = property(lambda self: ShardedVariable)

  def __init__(self, *variable_specs):
    self._variable_specs = tuple(variable_specs)

  def _serialize(self):
    return self._variable_specs

  @property
  def _component_specs(self):
    return self._variable_specs

  def _to_components(self, value):
    return value.variables

  def _from_components(self, variables):
    return ShardedVariable(variables)
  
  
# Override the behavior of embedding_lookup(sharded_variable, ...)
@dispatch.dispatch_for_types(embedding_ops.embedding_lookup, ShardedVariable)
def embedding_lookup(params,
                     ids,
                     partition_strategy='mod',
                     name=None,
                     validate_indices=True,
                     max_norm=None):
  if isinstance(params, list):
    params = params[0]
  return embedding_ops.embedding_lookup(params.variables, ids,
                                        partition_strategy, name,
                                        validate_indices, max_norm)
```

#### framework

* device_spec
  * `DeviceSpec(job="ps", replica=0, task=0, device_type="CPU", device_index=0)`
  * 如果disable eager了，需要对device spec做to_string才能with tf.device

* op_def_library
  * apply_op->_apply_op_helper -> Graph.create_op
    * 第一步是从op_def_registry中拿到OpDef
    * 然后调用 Graph.create_op，构造 `node_def = _NodeDef(op_type, name, attrs)`
    * 消除序列化，python实时注册op，由op_def_library调用

```python
import tensorflow as tf

a = tf.placeholder(tf.int16)  
b = tf.placeholder(tf.int16)
add = tf.add(a, b)  
mul = tf.mul(a, b)  

with tf.Session() as sess:
    print('a+b=',sess.run(add, feed_dict={a: [[2,3],[3,4]], b: [[1,2],[6,7]]}))
    print('a*b=',sess.run(mul, feed_dict={a: [[2,3],[3,4]], b: [[1,2],[6,7]]}))
```

* python_op_gen_internal
  * _op_def_lib.apply_op
* tensor_shape.py
  * init
    * 当 TensorShapeProto 的某一维度大小为-1 时，将其转换为 None 的表示

  * 工厂方法: as_shape
  * 如果 rank 大小未知，称该 TensorShape 未知;如果 rank 大小已知，则称该 TensorShape 部分定义(unknown_shape)
  * Properties: rank, dim, ndims
  * as_proto, as_list


```python
def num_elements(self):
  """Returns the total number of elements, or none for incomplete shapes."""
  if self.is_fully_defined():
    return functools.reduce(operator.mul, self.as_list(), 1)
  else
  	return None
```



##### ops.py

![image-20221230021056317](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/tensorflow/graph.png)

* Graph
  * 成员：
    * _nodes_by_id 和 _nodes_by_name 字典存数据
    * Finalize 冻结图
  
  * 分组
    * 为了更好地管理 Graph 中的节点，在每个 Operation 上打上特定的标签，实现了节点 的分类。相同类型的节点被划归在同一个 Collection 中，并使用唯一的 GraphKey 标识该集合。随后，便可以根据 GraphKey 快速索引相关的节点信息。其中，系统预定义了常用的 GraphKey，同时也支持自定义的 GraphKey。
    * class GraphKeys
      * tf.compat.v1.GraphKeys.VARIABLES 全局变量 (default)
      * tf.compat.v1.GraphKeys.LOCAL_VARIABLES: Key to collect local variables that are local to the machine and are not saved/restored
      * More
  
    * Graph.add_to_collection
    * Graph.get_collection_ref(KEY) -> list or map
      * `tf.compat.v1.get_default_graph().get_collection_ref(...)`
  
  * tf.Graph -> ScopedTFGraph (python) -> TF_Graph (C API) -> tensorflow::Graph (C++)
    * `_c_graph`: 直接将图实例传递给后端 C++，避免了前后端图实例序列化的开销。
  * create_op
  * with_init_scope
    * There is often a need to lift variable initialization ops out of control-flow scopes, function-building graphs, and gradient tapes.
    * 常用场景：代码逻辑中进行variable的初始化
  

```python
def _add_op(self, op, op_name):
  self._check_not_finalized()
  with self._lock:
    self._next_id_counter += 1
    op_id = self._next_id_counter
    self._nodes_by_id[op_id] = op
    self._nodes_by_name[op_name] = op
    self._version = max(self._version, op_id)
    return op_id
```

###### 图实例

* stack
  * global隐式图实例
* namescope
  * `@tf_contextlib.contextmanager def name_scope(self, name):` 用stack管理namescope
  * 在图构造期，OP 构造器更习惯于使用 tf.name_scope，它从输入的 Operation 或 Tensor 列表中尝试获取图实例;如果未能获取到，则返回默认的图实例。然后，再在该图实例上追加新的 name_scope
    * `graph_value = next((value for value in values if type(value) == Tensor), None)`

```python
with tf.Graph().as_default() as g:
	c = tf.constant(5.0)
	assert c.graph is g
  
_default_graph_stack = _DefaultGraphStack()

def get_default_graph():
  """Returns the default graph for the current thread."""
  return _default_graph_stack.get_default()

class Graph(object):
  def as_default(self):
    """Returns a context manager that makes this Graph the default graph.""" 
    return _default_graph_stack.get_controller(self)
```

###### 控制依赖

* 可以通过内嵌的 control_dependencies 合并外围的 control_dependencies，或通过 None 重置控制依赖集合为空。

```python
with g.control_dependencies([a, b]):
  # Ops constructed here run after `a` and `b`.
  with g.control_dependencies(None):
    # Ops constructed here not waiting for either `a` or `b`.
    with g.control_dependencies([c, d]):
      # Ops constructed here run after `c` and `d`,
      # also not waiting for either `a` or `b`.
	with g.control_dependencies([e, f]):
		# Ops constructed here run after `a, b, e, f`.
```

* _ControlDependenciesController 实现了一个控制依赖的控制器
  * Graph._control_dependencies_stack
  * _current_control_dependencies 用于规约所有外围的 control_inputs，直至当前层所依赖的 Operation 列表
  * _control_dependencies_for_inputs(self, input_ops)

###### Container

```python
with g.container('experiment0'):
  # All stateful Operations constructed in this context will be placed # in resource container "experiment0".
  v1 = tf.Variable([1.0])
  v2 = tf.Variable([2.0])
  with g.container("experiment1"):
    # All stateful Operations constructed in this context will be # placed in resource container "experiment1".
    v3 = tf.Variable([3.0])
    q1 = tf.FIFOQueue(10, tf.float32)
  # All stateful Operations constructed in this context will be # be created in the "experiment0".
  v4 = tf.Variable([4.0])
  q1 = tf.FIFOQueue(20, tf.float32)
  with g.container(""):
    # All stateful Operations constructed in this context will be # be placed in the default resource container.
    v5 = tf.Variable([5.0])
    q3 = tf.FIFOQueue(30, tf.float32)

# Resets container "experiment0", after which the state of v1, v2, v4, q1 # will become undefined (such as uninitialized).
tf.Session.reset(target, ["experiment0"])
```

```python
class Graph(object):
  @tf_contextlib.contextmanager
  def container(self, container_name):
"""Returns a context manager that specifies the resource container."""
    original_container = self._container
    try:
      self._container = container_name
      yield self._container
    finally:
      self._container = original_container
```



###### Import Graph

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

###### device

* 实现：用TraceableStack
  * _add_device_to_stack -> _UserDeviceSpec

* 调用：_apply_device_functions
  * LIFO, the most recently pushed function has the first chance to apply a device to the op
  * string_merge NodeDef （node_def.device 具有更高的优先级）





* Operation
  * session可以run op
  * Objects of type `Operation` are created by calling a Python op constructor (such as `tf.matmul`) within a `tf.function` or under a `tf.Graph.as_default` context manager.
  * init
    * input和output tensors作为输入，构建上下游消费关系
    * `self._c_op = _create_c_op(self._graph, node_def, inputs, control_input_ops, op_def)`

  * properties
    * name, type, graph, node_def, op_def
      * name 表示图中节点的名称，包括 name_scope 的层次名称，在图实例的范围内是唯一的，例如 layer_2/ MatMul
      * type 则表示该 OP 类型唯一的名称，例如 MatMul, Variable

    * id, device, _device_assignments, _colocation_dict
    * _output_types, outputs, inputs, _input_types
    * control_inputs, _control_outputs
    * traceback

* Tensor
  * 使用 op:index 的 二元组信息在图中唯一标识一个 Tensor 实例
  * tensor是边，存了生产者和消费者op信息，持有上游op索引
  * properties
    * op, dtype, graph, name, device, shape, value_index

  * shape相关操作
  * Eval
    * tf.Session.run 的 fetches 列表可以混合接收 Operation, Tensor 实例

  * consumers

* traceable_stack
  * 记录文件和行数信息的stack
* default管理
  * _DefaultStack(threading.local)
    * `@tf_contextlib.contextmanager def get_controller(self, default):`
  * get_default_graph()
  * get_default_session()
  * default_session
    * Use with the "with" keyword to specify that Tensor.eval() and Operation.run() invocations within the scope of a block should be executed by a particular session.

#### lib/io

* gfile
  * IsDirectory
  * MakeDirs

#### lib/core

* safe_pyobject_ptr
  * `using Safe_PyObjectPtr = std::unique_ptr<PyObject, detail::PyDecrefDeleter>;`

#### ops

* control_flow_ops
  * tf.group
    * `c = tf.group(a, b)` will compute the same graph as this: `with tf.control_dependencies([a, b]): c = tf.no_op()`
* data_flow_ops
  * [GPUCompatiableQueue](https://github.com/tensorflow/tensorflow/commit/f98b3bc7012085096d8171fe56f6004677461567#)
    * size方法是一个op
* metrics_impl
  * metric_variable
    * synchronization "ON_READ"
    * 通过 `distribution_strategy_context.get_replica_context().merge_call(fn, args=args)` 
  
  * precision_at_top_k
    * 两个variable，一个update op
  
    * _streaming_sparse_true_positive_at_k
  
* summary_ops_v2.py
  * summary.create_file_writer

  * ResourceSummaryWriter 存 C++ SummaryWriterInterface 的 handle

* variables.py
  * Variable 
    * 存了optimizer成员

  * PartitionedVariable
    * 可以用iter访问variables

  * tf.compat.v1.get_variable，参数有Partitioner
* Variable的magic方法
  * Variable重载了tensor的ops
  * 在math_ops中有 `ops.Tensor._override_operator("__neg__", gen_math_ops.neg)` `ops.Tensor._override_operator("__abs__", abs)`


```python
@classmethod
def _OverloadAllOperators(cls):  # pylint: disable=invalid-name
  """Register overloads for all operators."""
  for operator in ops.Tensor.OVERLOADABLE_OPERATORS:
    cls._OverloadOperator(operator)
  # For slicing, bind getitem differently than a tensor (use SliceHelperVar
  # instead)
  # pylint: disable=protected-access
  setattr(cls, "__getitem__", array_ops._SliceHelperVar)

@classmethod
def _OverloadOperator(cls, operator):  # pylint: disable=invalid-name
  """Defer an operator overload to `ops.Tensor`.

  We pull the operator out of ops.Tensor dynamically to avoid ordering issues.

  Args:
    operator: string. The operator name.
  """
  # We can't use the overload mechanism on __eq__ & __ne__ since __eq__ is
  # called when adding a variable to sets. As a result we call a.value() which
  # causes infinite recursion when operating within a GradientTape
  # TODO(gjn): Consider removing this
  if operator == "__eq__" or operator == "__ne__":
    return

  tensor_oper = getattr(ops.Tensor, operator)

  def _run_op(a, *args, **kwargs):
    # pylint: disable=protected-access
    return tensor_oper(a.value(), *args, **kwargs)

  functools.update_wrapper(_run_op, tensor_oper)
  setattr(cls, operator, _run_op)
```

```python
  OVERLOADABLE_OPERATORS = {
      # Binary.
      "__add__",
      "__radd__",
      "__sub__",
      "__rsub__",
      "__mul__",
      "__rmul__",
      "__div__",
      "__rdiv__",
      "__truediv__",
      "__rtruediv__",
      "__floordiv__",
      "__rfloordiv__",
      "__mod__",
      "__rmod__",
      "__lt__",
      "__le__",
      "__gt__",
      "__ge__",
      "__ne__",
      "__eq__",
      "__and__",
      "__rand__",
      "__or__",
      "__ror__",
      "__xor__",
      "__rxor__",
      "__getitem__",
      "__pow__",
      "__rpow__",
      # Unary.
      "__invert__",
      "__neg__",
      "__abs__",
      "__matmul__",
      "__rmatmul__"
  }
```



##### resource_variable_ops.py

* tf2默认是ResourceVariable
  
* BaseResourceVariable
  * assign
    * If `read_value` is `True`, this method will return the new value of the variable after the assignment has completed.
    * Otherwise, when in graph mode it will return the `Operation` that does the assignment, and when in eager mode it will return `None`.

  * 如果_cached_value不为None, 那么它的初值是什么, 怎样更新呢?
    - 初值是直接给的, 一般会用变量构图, 计算得到
    - 更新是通过_assign_dependencies上下文管理器中的control_dependencies实现, 这个上下文管理器会在如下两个方法中调用:
      - assign_add
      - assign_sub


```python
a = tf.Variable(1.0, use_resource=True)
a.initializer.run()
assign = a.assign(2.0)
with tf.control_dependencies([assign]):
  b = a.read_value()

with tf.control_dependencies([b]):
	other_assign = a.assign(3.0)

with tf.control_dependencies([other_assign]):
	# Will print 2.0 because the value was read before other_assign ran. If
	# `a` was a tf.Variable instead, 2.0 or 3.0 could be printed.
	tf.compat.v1.Print(b, [b]).eval()
```

```python
# 如果_cached_value不为空, 直接取_cached_value的值 , 否则用_read_variable_op取值
def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
  ...
	if as_ref:
    return self.read_value().op.inputs[0]
  else:
    return self.value()
  
def value(self):
  """A cached operation which reads the value of this variable."""
  if self._cached_value is not None:
    return self._cached_value
  with ops.colocate_with(None, ignore_existing=True):
    return self._read_variable_op()
  
  
@contextlib.contextmanager
def _assign_dependencies(self):
  """Makes assignments depend on the cached value, if any.

  This prevents undefined behavior with reads not ordered wrt writes.

  Yields:
    None.
  """
  if self._cached_value is not None:
    with ops.control_dependencies([self._cached_value]):
      yield
  else:
    yield
```

* Cache ResourceVariable

```python
with tf.device(None):  # 在本地create一个ResourceVariable当cached_value
   cached_var = resource_variable_ops.ResourceVariable(
     initial_value=var.initial_value,   # cached_value初始值
     trainable=False,  # 不让参与训练
     collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES],  # 本地变量
     shape=var.shape,
     dtype=var.dtype)
var._cached_value = cached_value(var, cached_var)
    
# 打断fetch_add的图上的dependency，不打断梯度
@tf.custom_gradient
def cached_value(var, async_cached_var):

  def grad(dy):
    return dy, None

  return async_cached_var, grad

```

* assign Variable

```python
def _get_valid_op_name(name: str):
  return name.replace(":", "_").replace("/", "_")
tf.compat.v1.assign(cached_var,
               			var._read_variable_op(),
			              name="fetch_from_{}".format(
                 					_get_valid_op_name(str(var.device))))
```

#### summary

* summary.py
  * core/framework/summary.proto
  * tensor_summary
  * merge, merge_all
* writer/writer.py
  * FileWriter封装EventFileWriter
* writer/event_file_writer_v2.py
  * reopen, add_event, flush, close

#### util

* tf_contextlib
  * `@tf_contextlib.contextmanager`
* tf_decorator

```python
def print_hello_before_calling(target):
    def wrapper(*args, **kwargs):
      print('hello')
      return target(*args, **kwargs)
    return tf_decorator.make_decorator(target, wrapper)
  
class CallCounter(tf_decorator.TFDecorator):
  def __init__(self, target):
    super(CallCounter, self).__init__('count_calls', target)
    self.call_count = 0

  def __call__(self, *args, **kwargs):
    self.call_count += 1
    return super(CallCounter, self).decorated_target(*args, **kwargs)

  def count_calls(target):
    return CallCounter(target)
```



#### training

* saver
  * 命名ckpt（counter）、管理ckpt（增删）
  * 参数
    * `max_to_keep`
    * `keep_checkpoint_every_n_hours`
  * 辅助类 BaseSaverBuilder
    * _AddShardedRestoreOps：获取restore ops per device
  * Note
    * 没有partial recovery（只对挂掉的ps做restore）的功能

```python
saver.save(sess, 'my-model', global_step=0) ==> filename: 'my-model-0'
...
saver.save(sess, 'my-model', global_step=1000) ==> filename: 'my-model-1000'


...
# Create a saver.
saver = tf.compat.v1.train.Saver(...variables...)
# Launch the graph and train, saving the model every 1,000 steps.
sess = tf.compat.v1.Session()
for step in xrange(1000000):
    sess.run(..training_op..)
    if step % 1000 == 0:
        # Append the step number to the checkpoint name:
        saver.save(sess, 'my-model', global_step=step)
```

```python
v1 = tf.Variable(..., name='v1')
v2 = tf.Variable(..., name='v2')

# Pass the variables as a dict:
saver = tf.compat.v1.train.Saver({'v1': v1, 'v2': v2})

# Or pass them as a list.
saver = tf.compat.v1.train.Saver([v1, v2])
# Passing a list is equivalent to passing a dict with the variable op names
# as keys:
saver = tf.compat.v1.train.Saver({v.op.name: v for v in [v1, v2]})
```

* basic_session_run_hook

  * StopAtStepHook: Request stop based on global_step

  - CheckpointSaverHook: saves checkpoint
    - 第一次也会save，实际上没必要

  - StepCounterHook

  - LoggingTensorHook: outputs one or more tensor values to log

  - NanTensorHook: Request stop if given `Tensor` contains Nans.

  - SummarySaverHook: saves summaries to a summary writer
  - GlobalStepWaiterHook: 用于distributed setting下的slow starting
  - FinalOpsHook
  - FeedFnHook
  - ProfilerHook
    - https://github.com/catapult-project/catapult/blob/master/tracing/README.md


* session_run_hook
  * tf.estimator.SessionRunHook
  * 常见的SessionRunHook，见basic_session_run_hook
  * SessionRunContext 的一些接口：session、request_stop
  * hook传入tf.Estimator.EstimatorSpec的training_hooks变量，Estimator封装好了MonitoredTrainingSesssion
  

```python
# For more specific needs, you can create custom hooks:
class ExampleHook(SessionRunHook):
  def begin(self):
    # You can add ops to the graph here.
    print('Starting the session.')
    self.your_tensor = ...
    self.your_op = ...

  def after_create_session(self, session, coord):
    # When this is called, the graph is finalized and
    # ops can no longer be added to the graph.
    print('Session created.')

  def before_run(self, run_context):
    print('Before calling session.run().')
    return SessionRunArgs(self.your_tensor)

  def after_run(self, run_context, run_values):
    print('Done running one step. The value of my tensor: %s',
          run_values.results)
    run_context.session.run(self.your_op)
    if you-need-to-stop-loop:
      run_context.request_stop()

  def end(self, session):
    print('Done with the session.')

# To understand how hooks interact with calls to `MonitoredSession.run()`, look at following code:
with MonitoredTrainingSession(hooks=your_hooks, ...) as sess:
  while not sess.should_stop():
    sess.run(your_fetches)
```

* training_util
  * _get_or_create_global_step_read()
  * get_global_step

#### 杂项

* dropout

  * keras/layers/core.py: `Class Dropout(Layer)` -> `nn.dropout`
  * ops/nn_ops.py: _dropout()
    * 注意是在training时进行scale，推理时忽略
  * 用法：`X = tf.nn.dropout(X, rate=1 - keep_prob)`
* clip_by_global_norm
  * https://stackoverflow.com/questions/44796793/difference-between-tf-clip-by-value-and-tf-clip-by-global-norm-for-rnns-and-how



### core

* 包括平台，实用函数库，基础框架，Protobuf 定义，本地运行时，分布式运行时，图操作，OP 定义，以及 Kernel 实现等组成
  * 68w行C++ code

```
tensorflow/core
|-- api_def
|   |-- base_api
|   |-- java_api
|   `-- python_api
|-- common_runtime
|   |-- eager
|   `-- gpu
|-- data
|   `-- service
|-- debug
|-- distributed_runtime
|   |-- eager
|   `-- rpc
|-- example
|   `-- testdata
|-- framework
|-- graph
|-- grappler
|   |-- clusters
|   |-- costs
|   |-- graph_analyzer
|   |-- inputs
|   |-- optimizers
|   |-- utils
|   `-- verifiers
|-- kernels
|   |-- batching_util
|   |-- boosted_trees
|   |-- data
|   |-- fuzzing
|   |-- hexagon
|   |-- image
|   |-- linalg
|   |-- mkl
|   |-- mlir_generated
|   |-- neon
|   |-- rnn
|   |-- sparse
|   |-- special_math
|   |-- spectrogram_test_data
|   `-- tensor_forest
|-- lib
|   |-- bfloat16
|   |-- bmp
|   |-- core
|   |-- db
|   |-- gif
|   |-- gtl
|   |-- hash
|   |-- histogram
|   |-- io
|   |-- jpeg
|   |-- llvm_rtti
|   |-- lmdb
|   |-- math
|   |-- monitoring
|   |-- png
|   |-- psnr
|   |-- random
|   |-- ssim
|   |-- strings
|   `-- wav
|-- nccl
|-- ops
|   `-- compat
|-- platform
|   |-- cloud
|   |-- default
|   |-- hadoop
|   |-- profile_utils
|   |-- s3
|   |-- testdata
|   `-- windows
|-- profiler
|   |-- builds
|   |-- convert
|   |-- g3doc
|   |-- internal
|   |-- lib
|   |-- protobuf
|   |-- rpc
|   `-- utils
|-- protobuf
|   |-- data
|   `-- tpu
|-- public
|-- summary
|-- tpu
|   |-- graph_rewrite
|   |-- kernels
|   `-- ops
|-- user_ops
`-- util
    |-- ctc
    |-- proto
    |-- rpc
    |-- sparse
    `-- tensor_bundle
```

#### common_runtime

##### session

* C++后端的session流程

```c++
// create/load graph ...
tensorflow::GraphDef graph;
// local runtime, target is ""
tensorflow::SessionOptions options;
// create Session
std::unique_ptr<tensorflow::Session> sess(tensorflow::NewSession(options));
// create graph at initialization.
tensorflow::Status s = sess->Create(graph); if (!s.ok()) { ... }
// run step
std::vector<tensorflow::Tensor> outputs; s = session->Run(
  {},               // inputs is empty
  {"output:0"},     // outputs names
  {"update_state"}, // target names
  &outputs);        // output tensors
if (!s.ok()) { ... }
// close
session->Close();
```



* session
  * Session 在 C++ 层的代码中是一个集成了计算资源和用于 graph 处理的对象。
  * Session的生命周期：
    * 创建、run、close、delete
  * Session 在创建时能获取当前机器上所有可用的计算资源（CPU、GPU 等），并使用 device_mgr_类对其进行管理。Session 根据计算资源维护一个动态的线程池，当获取到一个 graph 并启动运行时，Session 就会将 graph 调度到线程池中交由空闲的计算资源来完成计算。
    * NewSession: 从 SessionFactory 中拿
    * DirectSession构造：线程池
  * 支持 NewSession、Create、Extend、Run
  * Run 接口支持传入 RunOptions 获取 RunMetadata
    * A Session allows concurrent calls to Run(), though a Session must be created / extended by a single thread.
  * Extend 接口，调用 GraphExecutionState 的 MakeForBaseGraph、Extend接口
    * 系统新实现：在创建 OP 时，节点实时添加至后端 C++ 系统的图实例中
    * 旧机制：session run的时候，首次create，之后extend
    * self._extend_graph
  * ListDevices、LocalDeviceManager
  * a "warmup" phase to reduce the memory consumed by the session:
    * Call `Session::Create()`.
    * Call `Session::MakeCallable()` for all subgraphs that you will execute in the session.
    * Call `Session::Finalize()` to release global graph-related state.
    * Call `Session::RunCallable()` with the handle(s) created in step 2.
* direct_session
  * Run()
    * 累加 session 计数器
    * GetOrCreateExecutors(): 根据输入输出 tensor、目标节点，从当前 Session 中已有的 executor 中找是否存在一个相同任务的 executor，找到则将其返回，否则创建一个新的 executor
      * CreateExecutors
        * CreateGraphs
        * 构造 LocalExecutorParams params，作为Executor的参数，有关create和delete kernel的逻辑
          * OpSegment::ShouldOwnKernel
        * optimizer.Optimize
        * NewExecutor
      * 优化sort：先concat直接fast lookup再sort and concat，map内用原key和sorted key插入两份，value是shared_ptr
    * FeedArgs的构建，考虑了 DT_RESOURCE
      * 关于DT_RESOURCE一般会在使用ResourceVariable时才会碰到。ResourceVariable与Variable相比具有很多新特性，这些特性是TF2.0中主推的内容。关于它的优势我们不在这里展开，只对其Op的类型做一个说明。Variable在C++层面的Op类型是VariableV2，而ResourceVariable在C++层面的Op类型为VarHandleOp。后者产生的Tensor就是一种DT_RESOURCE
      * 见 python/ops/resource_variable_ops.py
    * RunInternal() executor 用于具体的执行
      * Global_handler、pool、sync执行，三个层面的三种抽象执行方式
      * executor->RunAsync 启动 graph 的运行
        * 配合 Notification、ExecutorBarrier 实现同步
      * rendezvous
      * 其它
        * ProfilerSession
        * CancellationManager
    * 接收输出
      * 处理重复的output_name
  * RunState
    * For each live Run() call, the session maintains a RunState.  'status' is the current status of the execution.

```c++
DirectSession::DirectSession(){
  ...
	for (auto d : device_mgr_->ListDevices()) {
    devices_.push_back(d);
    device_set_.AddDevice(d);
    d->op_segment()->AddHold(session_handle_);

    // The first device added is special: it is the 'client device' (a
    // CPU device) from which we feed and fetch Tensors.
    if (devices_added == 0) {
      device_set_.set_client_device(d);
    }
    ++devices_added;
  }
}

// 编码风格
Status DirectSession::Extend(GraphDef&& graph) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  mutex_lock l(graph_state_lock_);
  return ExtendLocked(std::move(graph));
}
```

```
Thread 1 "python" hit Breakpoint 2, 0x00007f91e7425a54 in tensorflow::(anonymous namespace)::ExecutorImpl::RunAsync(tensorflow::Executor::Args const&, std::function<void (tensorflow::Status const&)>) ()
   from /usr/local/lib/python3.6/dist-packages/tensorflow/python/../libtensorflow_framework.so.2
(gdb) bt
#0  0x00007f91e7425a54 in tensorflow::(anonymous namespace)::ExecutorImpl::RunAsync(tensorflow::Executor::Args const&, std::function<void (tensorflow::Status const&)>) ()
   from /usr/local/lib/python3.6/dist-packages/tensorflow/python/../libtensorflow_framework.so.2
#1  0x00007f91ef8718fb in tensorflow::DirectSession::RunInternal(long long, tensorflow::RunOptions const&, tensorflow::CallFrameInterface*, tensorflow::DirectSession::ExecutorsAndKeys*, tensorflow::RunMetadata*, tensorflow::thread::ThreadPoolOptions const&) ()
   from /usr/local/lib/python3.6/dist-packages/tensorflow/python/_pywrap_tensorflow_internal.so
```

* grpc_session
  * ExtendImpl
    * 如果判断引用 Master 的 handle 不为空，则执行 Extend；否则，执行 Create 的语义，建立与 Master 的连接，并持有 MasterSession 的 handle

* session_factory
  * static锁、static factories



* graph_execution_state
  * 创建executable graph、改图
  * executable graph和graph def的区别：Placed, meaning that each Node is assigned to a single Device in the available set.
  * OptimizeGraph: Grappler



##### Other Code

* colocation_graph
  * ColocateResourceOrRefEdge: placer将DT_RESOURCE相关的op放置在resource所在device上

* device_set

* entry: executor的输入，tensor or const or ref
  * Ref_tensor的生命周期管理是个难点


```c++
union {
  // A tensor value. Valid iff `state_ == HAS_VALUE`.
  ManualConstructor<Tensor> val;

  // A pointer to a constant tensor value. Valid iff `state_ ==
  // HAS_CONST_TENSOR`.
  const Tensor* const_tensor;

  // A tensor reference and associated mutex. Valid iff `state_ ==
  // HAS_REF_TENSOR`.
  struct {
    Tensor* tensor;
    mutex* mu;
  } ref_tensor;
};
```



* executor
  * RunAsync
    * 创建一个 ready 队列
    * 把入度为 0 的 node 加入 ready 队列
    * 调用 ScheduleReady() 把 ready 队列中的内容调度起来运行
      * ScheduleReady->RunTask(Process())->ProcessAsync->NodeDone->ScheduleReady

  * RunTask
    * Align the atomic variables at 64 bytes to avoid false-sharing, assuming the cacheline size is 64 bytes or smaller.

  * PrepareInputs: 由Entry构造TensorValueVec
    
  * "step_id" is a process-wide unique identifier for the step being run. Executors on different devices may receive the same step_id in the case that a step runs Ops on more than one device. The step_id is used for tracking resource usage of a given step.


```c++
Graph* graph = ...;
    ... construct graph ...
Executor* executor;
TF_CHECK_OK(NewSimpleExecutor(my_device, graph, &executor));
Rendezvous* rendezvous = NewNaiveRendezvous();
TF_CHECK_OK(rendezvous->Send("input", some_input_tensor));
TF_CHECK_OK(executor->Run({ExecutorOpts, rendezvous, nullptr}));
TF_CHECK_OK(rendezvous->Recv("output", &output_tensor));
```



* public
  * session.h
  * session_options.h
  * version.h
    * GraphDefVersion

```c++
tensorflow::GraphDef graph;
// ... Create or load graph into "graph".
// This example uses the default options which connects
// to a local runtime.
tensorflow::SessionOptions options;
std::unique_ptr<tensorflow::Session>
session(tensorflow::NewSession(options));

// Create the session with this graph.
tensorflow::Status s = session->Create(graph);
if (!s.ok()) { ... }

// Run the graph and fetch the first output of the "output"
// operation, and also run to but do not return anything
// for the "update_state" operation.
std::vector<tensorflow::Tensor> outputs;
s = session->Run({}, {"output:0"}, {"update_state"}, &outputs);
if (!s.ok()) { ... }

// Map the output as a flattened float tensor, and do something
// with it.
auto output_tensor = outputs[0].flat<float>();
if (output_tensor(0) > 0.5) { ... }

// Close the session to release the resources associated with
// this session.
session->Close();

```



* kernel launch
  * executor.cc: Process() / ProcessAsync()
  * eager/kernel_and_device.cc : KernelAndDeviceOp::Run

#### distributed_runtime

https://www.tensorflow.org/api_docs/python/tf/distribute

* 概念
  * Server：通常一台机器上运行一个 Server，用于管理机器上的计算资源、运行 graph 以及完成对远程服务的通信。
    * 每个 Server 中都会运行 2 种服务，Master Service 和 Worker Service。 
    * Master Service：一个 RPC 进程，用于启动 Session 和管理远程的 Worker Service。也作为创建tf.Session的target 
    * Worker Service：一个 RPC 进程，负责在本地设备执行TF中的计算子图。 
  * Client：用户操作的那个程序进程叫 Client，即写代码的那一个进程。 创建TF的计算图，通过建立Session与cluster中的设备进行交互。建立Session时会指定target，对应的就是server的master。
    * 这个worker master负责资源调度（图如何计算，在哪个设备计算等）
* Replicated Training
  * **In-graph replication：**只构建一个client，这个client构建一个Graph，Graph中包含一套模型训练的参数，放置在ps上，同时Graph中包含模型计算部分的多个副本，每个副本放在一个worker上，这样多个worker就可以同时训练复制的模型。
  * **Between-graph replication：**每个worker都构建一个client，各个client构建相同的Graph（similar graph，因为chief worker会有额外的Op），但是参数仍然放在ps上，这样即使worker的client挂了，不影响其他worker的训练。
    - 这个方式下，对于一些公共操作比如模型参数初始化与checkpoint文件保存等，如果每个worker都做则会浪费资源，这个时候需要一个chief worker来作为各个worker的管家，协调训练，并且完成这些公共操作
  
* 再高一层： 
  * Task：Task 对应了一个 Server，用于具体处理某些任务。 
  * Job：一组 Task 可以共同完成一个 Job。
  * Cluster：一整个分布式运行时的集合，包括运行任务、设备等等。一个 Cluster 中包含了一个或多个 Job，每个 Job 又可分为一个或多个 Task。

```
tf.train.ClusterSpec({
 "worker": [
 "worker0.example.com:2222",
 "worker1.example.com:2222",
 "worker2.example.com:2222"
 ],
 "ps": [
 "ps0.example.com:2222",
 "ps1.example.com:2222"
]})

/job:worker/task:0
/job:worker/task:1
/job:worker/task:2
/job:ps/task:0
/job:ps/task:1
```

* `server = tf.distribute.Server.create_local_server()`

```
#0  0x00007f91ea36b350 in tensorflow::RpcRendezvousMgr::RpcRendezvousMgr(tensorflow::WorkerEnv const*)@plt ()

#1  0x00007f91eb0dffa2 in tensorflow::(anonymous namespace)::NewRpcRendezvousMgr(tensorflow::WorkerEnv const*) ()

#2  0x00007f91eb0e4aa6 in tensorflow::GrpcServer::Init(tensorflow::GrpcServerOptions const&) ()

#3  0x00007f91eb0e5ece in tensorflow::GrpcServer::Create(tensorflow::ServerDef const&, tensorflow::Env*, tensorflow::DeviceMgr const*, std::unique_ptr<tensorflow::ServerInterface, std::default_delete<tensorflow::ServerInterface> >*) ()

#4  0x00007f91eb0e61f4 in tensorflow::(anonymous namespace)::GrpcServerFactory::NewServer(tensorflow::ServerDef const&, tensorflow::ServerFactory::Options const&, std::unique_ptr<tensorflow::ServerInterface, std::default_delete<tensorflow::ServerInterface> >*) ()

#5  0x00007f91f00b9566 in tensorflow::NewServer(tensorflow::ServerDef const&, std::unique_ptr<tensorflow::ServerInterface, std::default_delete<tensorflow::ServerInterface> >*) ()

#6  0x00007f91eafc21f5 in TF_NewServer ()
```

* worker 0

```python
import tensorflow.compat.v1 as tf
cluster = tf.train.ClusterSpec({"worker": ["localhost:22222","localhost:22223"]})
server = tf.train.Server(cluster, job_name="worker", task_index=0)
tf.disable_eager_execution()
with tf.device("/job:worker/task:1"):
  a = tf.constant(1)
  b = tf.constant(2)
  c = a + b

with tf.Session("grpc://localhost:22222") as sess: # tf.Session(server.target)
    sess.run(c)
```

* worker 1

```python
import tensorflow.compat.v1 as tf
cluster = tf.train.ClusterSpec({"worker":["localhost:22222","localhost:22223"]})
server = tf.train.Server(cluster, job_name="worker", task_index=1)
server.join()
```



* gdb distribute

  * 对worker 0打断点, `gdb --args python 0.py`

  * `b tensorflow::(anonymous namespace)::RpcRemoteRendezvous::RecvFromRemoteAsync`

```
#0  0x00007fffeef6aaf4 in tensorflow::(anonymous namespace)::RpcRemoteRendezvous::RecvFromRemoteAsync(tensorflow::RendezvousInterface::ParsedKey const&, tensorflow::RendezvousInterface::Args const&, std::function<void (tensorflow::Status const&, tensorflow::RendezvousInterface::Args const&, tensorflow::RendezvousInterface::Args const&, tensorflow::Tensor const&, bool)>) ()

#1  0x00007fffeef7338e in tensorflow::BaseRemoteRendezvous::RecvAsync(tensorflow::RendezvousInterface::ParsedKey const&, tensorflow::RendezvousInterface::Args const&, std::function<void (tensorflow::Status const&, tensorflow::RendezvousInterface::Args const&, tensorflow::RendezvousInterface::Args const&, tensorflow::Tensor const&, bool)>) ()

#2  0x00007fffec2ce3da in tensorflow::RecvOp::ComputeAsync(tensorflow::OpKernelContext*, std::function<void ()>) ()

#3  0x00007fffea0c7d16 in tensorflow::Device::ComputeAsync(tensorflow::AsyncOpKernel*, tensorflow::OpKernelContext*, std::function<void ()>) ()
```

* master
  * 管理 MasterSession 对象，key为session handle
  * CreateSession
  * CloseSession

* master_session
  * DoRegisterPartitions: 收到 graph_handle
  * RunPartitionsHelper
    * RunPartitions 的实现，when at least two of the {client, master, worker} are in the same process，多了一次tensor copy
* worker
  * RegisterGraphAsync
    * 由 `MasterSession::ReffedClientGraph::DoRegisterPartitions` 调用
  * RunGraphAsync (DoRunGraph)
    * PrepareRunGraph
    * `session->graph_mgr()->ExecuteAsync`
      * 对每个partition执行
    * `session->graph_mgr()->RecvOutputs(step_id, out)`
* grpc_remote_worker
  * RecvTensorAsync
    * 由 RpcRemoteRendezvous 调用
  
* grpc_session
  * Close() 调用 Master::CloseSession
  



* rendezvous
  * [Send-Driven & Rendezvous-Bypass 优化](https://tech.meituan.com/2021/12/09/meituan-tensorflow-in-recommender-systems.html)





#### framework

* [Walkthrough of TensorFlow Architecture](https://www.gresearch.co.uk/blog/article/walkthrough-of-tensorflow-architecture/)
  * Row-major order
  * cpu op实现基于eigen
  * 构图
    * topological sort
    * SymbolicGradientBuilder::SumGradients
      * TensorFlow adds a SumGradients op as required by the chain rule.

* allocator.h
  * AllocatorAttributes: 是entry的属性之一
    * on_host、nic_compatible、gpu_compatible



##### op_kernel

##### 如何写一个Op

* [tf guide](https://www.tensorflow.org/guide/create_op)、https://github.com/huangrt01/custom-op
  * Register the new op in a C++ file.
  * Implement the op kernel in C++.
    * To write a multi-threaded CPU kernel, the Shard function in [`work_sharder.h`](https://www.tensorflow.org/code/tensorflow/core/util/work_sharder.h) can be used. This function shards a computation function across the threads configured to be used for intra-op threading (see intra_op_parallelism_threads in [`config.proto`](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto)).
  * Create a Python wrapper (optional). This wrapper is the public API that's used to create the op in Python. A default wrapper is generated from the op registration, which can be used directly or added to.
  * Write a function to compute gradients for the op (optional).
  * Test the op. We usually do this in Python for convenience, but you can also test the op in C++. If you define gradients, you can verify them with the Python `tf.test.compute_gradient_error`. See [`relu_op_test.py`](https://www.tensorflow.org/code/tensorflow/python/kernel_tests/relu_op_test.py) as an example that tests the forward functions of Relu-like operators and their gradients.
  * TODO: Advanced features

* [Extending TensorFlow with Custom C++ Operations](https://www.gresearch.co.uk/blog/article/extending-tensorflow-with-custom-c-operations/)
  * Motivation
    * the op’s outputs are dependent only on its inputs and not influenced by any of the subsequent operations. One instance where **this greedy strategy can lead to poor performance is computation involving many elementwise operations on the GPU** (as in the case of the [Gaussian Error Linear Unit](https://arxiv.org/abs/1606.08415) mentioned in our previous post).
  * 基础版本 tf_cdist，耗时长
  * tf_cdist_vectorized，显存使用多，O(mnp)

```python
@tf.function
def tf_cdist(x, y):
    n, p = x.shape
    m, p = y.shape
    rows = []
    for i in range(n):
      row_elems = []
      for j in range(m):
        manhattan_dist_ij = tf.math.reduce_sum(tf.abs(x[i, :] - y[j, :]))
        row_elems.append(manhattan_dist_ij)
      rows.append(tf.stack(row_elems, axis=0))
    return tf.stack(rows, axis=0)

@tf.function
def tf_cdist_vectorised(x, y):
    n, p = tf.shape(x)
    m, p = tf.shape(y)
    z = tf.abs(tf.reshape(x, (n, 1, -1)) - tf.reshape(y, (1, m, -1)))
    return tf.sum(z, axis=-1)
```



* Op Registration Defines
  * input tensors
  * output tensors
  * allowable types of the tensors
  * shape checking functionality (to ensure inputs and outputs conform to the shapes required by the computation)

```c++
REGISTER_OP("my_op_name")
    .Attr("<name>:<type>")
    .Attr("<name>:<type>=<default>")
    .Input("<name>:<type-expr>")
    .Input("<name>:Ref(<type-expr>)")
    .Output("<name>:<type-expr>")
    .Doc(R"(
<1-line summary>
<rest of the description (potentially many lines)>
<name-of-attr-input-or-output>: <description of name>
<name-of-attr-input-or-output>: <description of name;
  if long, indent the description on subsequent lines>
)");
```

![image-20230103000701765](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/tensorflow/register-opdef.png)

* op.h
  * OpRegistry（Graph持有）
  * 实现细节参考 platform/macros、platform/selective_registration.h
* load_library
  * LoadDynamicLibrary，调用OpRegistry相关函数

* OpKernelContext
  * The context object simply allows the op access to basic functionality such as querying the device, fetching the op’s inputs, and allocating space for new tensors.
  * `ctx->GetAttr("config", &config_serialized_)` 类型是string
  * allocate tensor
    * allocate_persistent: only needed for Tensors that will be stored by the Op between invocations
    * allocate_output
    * allocate_temp
      * 将input、persistent tensor设为output，set_output or set_output_ref
      * 注意如果AllocatorAttributes不一致，可能有额外copy
      * 用这个接口可能是为了LogMemory

```c++
// tensorflow/core/ops/pairwise_manhattan_distance_ops.cc
REGISTER_OP("PairwiseManhattanDistance")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {float, double}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x, y;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &y));
      DimensionHandle n_x = c->Dim(x, 0);
      DimensionHandle n_y = c->Dim(y, 0);
      c->set_output(0, c->Matrix(n_x, n_y));
      return tensorflow::Status::OK();
    })

namespace functor {
  void Compute(OpKernelContext* ctx) override {
      const Tensor* x_tensor = nullptr;
      const Tensor* y_tensor = nullptr;
      // Retrieve all of the inputs
      OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));
      OP_REQUIRES_OK(ctx, ctx->input("y", &y_tensor));
      const int64 n = x_tensor->dim_size(0);
      const int64 m = y_tensor->dim_size(0);
      const int64 p = x_tensor->dim_size(1);
			
      // Allocate space for the output
      Tensor* z_tensor = nullptr;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output("z", TensorShape({n, m}),
                                    &z_tensor));
    
    	const Device& device = ctx->eigen_device<Device>();
      // Call a device specific implementation to fill the output
      functor::ManhattanDistance<Device, T>(n, m, p)(
          ctx, device,
          x_tensor->matrix<T>(), y_tensor->matrix<T>(),
          z_tensor->matrix<T>());
  }
  
  template <typename T>
  void ManhattanDistance(
      OpKernelContext* ctx, const CPUDevice& d,
      typename TTypes<T>::ConstMatrix x,
      typename TTypes<T>::ConstMatrix y,
      typename TTypes<T>::Matrix z) {
    auto n = x.dimension(0);
    auto m = y.dimension(0);
    auto p = y.dimension(1);
    T diff;
    T dist;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        dist = static_cast<T>(0);
        for (int k = 0; k < p; k++) {
          diff = x(i, k) - y(j, k);
          dist += Eigen::numext::abs(diff);
        }
        z(i, j) = dist;
      }
    }
  }
  
  // GPU Op
  template <typename T>
  __global__ void manhattan_distance(
      const T* x,
      const T* y,
      T* z,
      const int n,
      const int m,
      const int p) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = i * m + j;
    if ((i >= n) || (j >= m)) return;
    T dist = 0.0;
    T diff;
    for (int k = 0; k < p; k++) {
      diff = x[i * p + k] - y[j * p + k];  // x[i, k] - y[j, k]
      dist += Eigen::numext::abs(diff);
    }
    z[index] = dist;
  }
  
  // tensorflow/core/kernels/pairwise_manhattan_distance_gpu.cu.cc
  template <typename T>
  void ManhattanDistance(
      OpKernelContext* ctx, const GPUDevice& d,
      typename TTypes<T>::ConstMatrix x,
      typename TTypes<T>::ConstMatrix y,
      typename TTypes<T>::Matrix z) {
    const auto& cu_stream = GetGpuStream(ctx);
    const int n = x.dimension(0);
    const int m = y.dimension(0);
    const int p = x.dimension(1);

    dim3 block_dim_2d(32, 8);
    auto grid_dim_x = (m + block_dim_2d.x - 1) / block_dim_2d.x;
    auto grid_dim_y = (n + block_dim_2d.y - 1) / block_dim_2d.y;
    dim3 grid_dim_2d(grid_dim_x, grid_dim_y);
    manhattan_distance<T><<<grid_dim_2d, block_dim_2d, 0, cu_stream>>>(
        x.data(), y.data(), z.data(), n, m, p);
  }

}
```

* Backward Op
  * With a few exceptions, for every op registration there is a corresponding gradient operation. Exceptions include:
    * operations that define their gradients in Python
    * operations that have no gradient at all (e.g. argsort)
    * operations that are only run in inference (the *PairwiseManhattanDistance* could be an instance of this).

```c++
REGISTER_OP("PairwiseManhattanDistanceGrad")
    .Input("x: T")
    .Input("y: T")
    .Input("z_grad: T")
    .Output("x_grad: T")
    .Output("y_grad: T")
    .Attr("a: int = 1")
    .Attr("T: {float, double}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x, y;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &y));
      DimensionHandle n_x = c->Dim(x, 0);
      DimensionHandle n_y = c->Dim(y, 0);
      DimensionHandle feature_dim = c->Dim(x, 1);
      c->set_output(0, c->Matrix(n_x, feature_dim));
      c->set_output(1, c->Matrix(n_y, feature_dim));
      return tensorflow::Status::OK();
    })
```



* 用op管理ResouceBase

```c++
void Compute(OpKernelContext* ctx) override {
  absl::MutexLock l(&mu_);
  if (my_resource_ == nullptr) {
    ResourceMgr* rmgr = ctx->resource_manager();
    OP_REQUIRES_OK(ctx, cinfo_.Init(rmgr, def()));
    auto creator = [this, ctx](MyResource** out_resource) {
      *out_resource = ...;
      return Status::OK();
    };
    OP_REQUIRES_OK(
          ctx, rmgr->LookupOrCreate<MyResource>(
                   cinfo_.container(), cinfo_.name(),
            			 &my_resource_, creator));
  }
  OP_REQUIRES_OK(
    ctx, MakeResourceHandleToOutput(
      ctx, 0, cinfo_.container(), cinfo_.name(),
      TypeIndex::Make<MyResource>()));
}

private:
  std::string config_serialized_;
  absl::Mutex mu_;
  MyResource* my_resource_ ABSL_GUARDED_BY(mu_) = nullptr;
  ContainerInfo cinfo_ ABSL_GUARDED_BY(mu_);

REGISTER_OP("MyResource")
    .Input("dependent_resource: resource")
    .Output("handle: resource")
    .Attr("config: string")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

class MyResource : public ResourceBase {
  
};
```

* Parallel Run

```c++
std::function<void(int64, int64)> fn = ...;
auto workers = ctx->device()->tensorflow_cpu_worker_threads()->workers;
workers->ParallelFor(
    task_count,
    thread::ThreadPool::SchedulingParams(
        thread::ThreadPool::SchedulingStrategy::kFixedBlockSize,
        absl::nullopt, 1),
    fn);
```





##### proto文件

* node_def.proto
* op_def.proto
  * ArgDef
* event.proto

```protobuf
// Protocol buffer representing an event that happened during
// the execution of a Brain model.
message Event {
  // Timestamp of the event.
  double wall_time = 1;

  // Global step of the event.
  int64 step = 2;

  oneof what {
    // An event file was started, with the specified version.
    // This is use to identify the contents of the record IO files
    // easily.  Current version is "brain.Event:2".  All versions
    // start with "brain.Event:".
    string file_version = 3;
    // An encoded version of a GraphDef.
    bytes graph_def = 4;
    // A summary was generated.
    Summary summary = 5;
    // The user output a log message. Not all messages are logged, only ones
    // generated via the Python tensorboard_logging module.
    LogMessage log_message = 6;
    // The state of the session which can be used for restarting after crashes.
    SessionLog session_log = 7;
    // The metadata returned by running a session.run() call.
    TaggedRunMetadata tagged_run_metadata = 8;
    // An encoded version of a MetaGraphDef.
    bytes meta_graph_def = 9;
  }
}
```

* summary.proto
  * summary is a list of value

```protobuf
message Summary.Value {
  // Tag name for the data. Used by TensorBoard plugins to organize data. Tags
  // are often organized by scope (which contains slashes to convey
  // hierarchy). For example: foo/bar/0
  string tag = 1;

  // Contains metadata on the summary value such as which plugins may use it.
  // Take note that many summary values may lack a metadata field. This is
  // because the FileWriter only keeps a metadata object on the first summary
  // value with a certain tag for each tag. TensorBoard then remembers which
  // tags are associated with which plugins. This saves space.
  SummaryMetadata metadata = 9;

  // Value associated with the tag.
  oneof value {
    float simple_value = 2;
    Image image = 4;
    HistogramProto histo = 5;
    Audio audio = 6;
    TensorProto tensor = 8;
  }
}
```



##### tensor

* shape and type metadata --> 支持 Indexing、Slicing
* Slicing:
  * numpy: [*strides* metadata](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html#:~:text=The strides of an array,position in the next row.)，支持tensor view
  * Tf: slicing a TensorFlow tensor will always create a copy *unless* you slice the first dimension
    * core/kernels/slice_op.cc
* class TensorBuffer : public core::RefCounted
* class Tensor
  * 构造函数：可以用 TensorBuffer 构造 Tensor
  * 复制构造：CopyFromInternal，特点是share buffer
  * shape相关
    * shape(), dims(), dim_size(), NumElements(), IsSameSize, SharesBufferWith
  * IsInitialized()、TotalBytes()、AllocatedBytes()
  * IsAligned()
  * `Tensor Slice(int64 dim0_start, int64 dim0_limit) const;`
    * The caller must check the returned tensor's alignment before calling certain methods that have alignment requirement
  * `Tensor SubSlice(int64 index) const;`
  * FromProto、AsProtoTensorContent、AsProtoField
  * vec(), matrix(), tensor(), bit_casted_tensor()
    * `tensorflow::TTypes<float>::Flat data`
  * flat(), unaligned_flat(), flat_inner_dims(), flat_outer_dims()
  * `std::string SummarizeValue(int64 max_entries, bool print_v2 = false) const;`
  * DeviceSafeDebugString()
  * `StringPiece Tensor::tensor_data()`
  * use the OpKernelConstruction/OpKernelContext allocate_* methods to allocate a new tensor, which record the kernel and step.

```c++
template <typename T>
Tensor::Tensor(T value, host_scalar_tag tag) {
  auto* value_and_buf = static_cast<Tensor::ValueAndTensorBuffer<T>*>(
      port::AlignedMalloc(sizeof(typename Tensor::ValueAndTensorBuffer<T>),
                          EIGEN_MAX_ALIGN_BYTES));
  new (&value_and_buf->value) T(std::move(value));
  new (&value_and_buf->tensor_buffer)
      typename Tensor::ValueAndTensorBuffer<T>::HostScalarTensorBuffer(
          value_and_buf);
  buf_ = &value_and_buf->tensor_buffer;
  set_dtype(DataTypeToEnum<T>::value);
}
```

```c++
typedef float T;
Tensor my_ten(...built with Shape{planes: 4, rows: 3, cols: 5}...);
// 1D Eigen::Tensor, size 60:
auto flat = my_ten.flat<T>();
// 2D Eigen::Tensor 12 x 5:
auto inner = my_ten.flat_inner_dims<T>();
// 2D Eigen::Tensor 4 x 15:
auto outer = my_ten.shaped<T, 2>({4, 15});
// CHECK fails, bad num elements:
auto outer = my_ten.shaped<T, 2>({4, 8});
// 3D Eigen::Tensor 6 x 5 x 2:
auto weird = my_ten.shaped<T, 3>({6, 5, 2});
// CHECK fails, type mismatch:
auto bad   = my_ten.flat<int32>();
```

```c++
friend Status batch_util::CopyElementToSlice(
  Tensor element, Tensor* parent,
  int64 index);  // For access to base<T>().
friend Status batch_util::CopySliceToElement(
  const Tensor& parent, Tensor* element,
  int64 index);  // For access to base<T>().
friend Status batch_util::MaybeMoveSliceToElement(
  Tensor* parent, Tensor* element,
  int64 index);  // For access to base<T>().
friend Status batch_util::CopyContiguousSlices(
  const Tensor& src, int64 src_offset, int64 dst_offset, int64 num_slices,
  Tensor* dst);  // For access to base<T>().
```

##### Other Code

* bounds_check
  * FastBoundsCheck
  * internal::SubtleMustCopy
  
* run_handler TODO
  * schedule inter/intra-op closures to run on a global pool shared across all Session::Run(s)
  * Pimpl
* op_def_builder
  * FinalizeInputOrOutput
    * 如果arg type是DT_RESOURCE，则将op设为stateful
  
* op_kernel
  * TensorValue
    * Holds a tensor or tensor reference. For tensor references, we need a mutex to prevent concurrent access to the tensor.

  * `#define REGISTER_KERNEL_BUILDER(kernel_builder, ...)`

* op_segment
  * keeps track of OpKernels registered for sessions running on a device
    * Ref 管理，AddHold
  * ShouldOwnKernel
    * OpSegment should not own kernel if the node is stateless, or a function.

* register_types

```c++
#define REGISTER_SLICE(type)                             \
  REGISTER_KERNEL_BUILDER(Name("Slice")                  \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("begin")       \
                              .HostMemory("size"),       \
                          SliceOp<CPUDevice, type>)

TF_CALL_POD_STRING_TYPES(REGISTER_SLICE);
TF_CALL_QUANTIZED_TYPES(REGISTER_SLICE);
#undef REGISTER_SLICE
```

* Type_index
  * 避免RTTI，使用参考MakeResourceHandleToOutput

##### resource_mgr

* Resource_handle
  * Not valid across executions, but can be serialized back and forth from within a single run.
  * EncodeResourceHandleList 利用 `port::StringListEncoder`
  * ANONYMOUS_NAME：GUID for anonymous resources. Resources with this shared_name will have their shared_name replaced with a GUID at creation time

```protobuf
message ResourceHandleProto {
  // Unique name for the device containing the resource.
  string device = 1;

  // Container in which this resource is placed.
  string container = 2;

  // Unique name of this resource.
  string name = 3;

  // Hash code for the type of the resource. Is only valid in the same device
  // and in the same execution.
  uint64 hash_code = 4;

  // For debug-only, the name of the type pointed to by this handle, if
  // available.
  string maybe_type_name = 5;

  // Protocol buffer representing a pair of (data type, tensor shape).
  message DtypeAndShape {
    DataType dtype = 1;
    TensorShapeProto shape = 2;
  }

  // Data types and shapes for the underlying resource.
  repeated DtypeAndShape dtypes_and_shapes = 6;

  reserved 7;
}
```



* resource_mgr: 用来表示状态

  * A ResourceMgr instance keeps track of named and typed resources grouped into containers. (可以用来keep op的states)
    * scoped to Device objects (very weird lifetime/sharing)
      * 比如：可能outlive session、eager模式的细微区别
    * ResourceHandle是一个scalar，ResourceMgr依赖它create/lookup
      *  `Tensor handle(DT_RESOURCE, TensorShape({})); handle.scalar<ResourceHandle>()() = handle_;`

  * Each resource must be represented as a sub-class of ResourceBase, which is reference counted explicitly.  Each named resource is registered with ResourceMgr under a named "container" name. At any time, there is at most one instance of a resource given the container name, the resource type and the resource name.
  * All resources for a given container can be dropped by one call of Cleanup().

```c++
struct MyVar : public ResourceBase {
  mutex mu;
  Tensor val;
}

ResourceMgr rm;

// Create a var.
MyVar* my_var = new MyVar;
my_var->val = Tensor(DT_FLOAT, my_shape);
my_var->val.flat<float>().setZeros();   // 0 initialized.
ctx->SetStatus(rm.Create("my_container", "my_name", my_var));

// += a variable.
MyVar* my_var = nullptr;
Status s = rm.Lookup("my_container", "my_name", &my_var);
if (s.ok()) {
  my_var->val.flat<float>() += grad;
}
my_var->Unref();   // Or use ScopedUnref().
ctx->SetStatus(s);
```

* ResourceMgr

  * Create
    * DoCreate的实现，用了unordered_map的value_type的技巧，实现
  * LookUp、LookUpMany
  * Clear
    * We do the deallocation outside of the lock to avoid a potential deadlock in case any of the destructors access the resource manager.
  * `typedef std::unordered_map<Key, ResourceAndName, KeyHash, KeyEqual> Container;`

* OpKernelContext + ResourceMgr

  * MakeResourceHandle

  * MakeResourceHandleToOutput
    * 用到了TypeIndex

  * HandleFromInput
  * LookupResource(s), LookupOrCreateResource(s), DeleteResource

```c++
OP_REQUIRES_OK(
    ctx, MakeResourceHandleToOutput(
      ctx, 0, cinfo_.container(), cinfo_.name(),
      TypeIndex::Make<MyResource>()));

const ResourceHandle& HandleFromInput(OpKernelContext* ctx, int input) {
  return ctx->input(input).flat<ResourceHandle>()(0);
}

Status DeleteResource(OpKernelContext* ctx, const ResourceHandle& p) {
  TF_RETURN_IF_ERROR(internal::ValidateDevice(ctx, p));
  return ctx->resource_manager()->Delete(p);
}

// Helper for kernels to obtain 'resource' from the
// ctx->resource_manager().
//
// "input_name" specifies the kernel's ref input which gives a string
// tensor with two elements, which specifies the container and
// resource name.
//
// Returns OK if the resource is found and transfers one ref of
// *resource to the caller. Otherwise, returns an error.
template <typename T>
Status GetResourceFromContext(OpKernelContext* ctx,
                              const std::string& input_name, T** resource);
```

* Kernels
  * IsResourceInitialized
  * ResourceHandleOp
    * ANONYMOUS_NAME时会创建一个新handle
  * ResourceHandlesOp
* macros
  * REGISTER_RESOURCE_HANDLE_OP(Type)
  * REGISTER_RESOURCE_HANDLE_KERNEL(type)
* ContainerInfo
  * 私有的resource，递增命名

```c++
resource_is_private_to_kernel_ = true;
static std::atomic<int64> counter(0);
name_ = strings::StrCat("_", counter.fetch_add(1), "_", ndef.name());
```

* ResourceDeleter

#### graph

* 计算图是 TensorFlow 领域模型的核心

![image-20230103002049555](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/tensorflow/graph-object.png)

##### graph

* Edge
  * 普通边:用于承载数据 *(*以 Tensor 表示*)*，表示节点间“生产者*-*消费者”的数据 依赖关系，常用实线表示;
  * 控制依赖:不承载数据，用于表示节点间的执行依赖关系，常用虚线表示
  * 持有两个索引 src_output, dst_input
    * `IsControlEdge()`
  * TensorId ::= node_name:src_output
    * 参考InputTensor、OutputTensor
      * OutputTensor::Hash， Hash64Combine
    * 缺省地，src_output 默认为 0;也就是说，node_name 与 node_name:0 两者等价。
    * 特殊地，当 src_output 等于-1 时，表示该边为「控制依赖边」，TensorId 可以标识为 node_name，标识该边依赖于 node_name 所在的节点
  * Properties 用 shared_ptr 实现COW
* Node
  * 节点可以拥有零条或多条输入/输出的边，并使用 in_edges, out_edges 分别表 示输入边和输出边的集合
  * Node 持有 NodeDef, OpDef。其中，NodeDef 包含设备分配信息，及其 OP 的属性值列表；OpDef 持有 OP 的元数据，包括 OP 输入输出类型等信息
    * input_edge(): 线性搜索
    * input_node()
    * def() provides the NodeDef the user supplied, but the specifics of this Node may have changed due to placement, optimization, etc. In particular:
      * def().name() will match name();
      * def().op() will match type_string() and op_def().name();
      * def().input() is not reliable, use "in_edges()" below instead;
      * def().device() is the "user's requested device" and may not match the actual assigned device, see assigned_device_name() below;
      * def().attr() is authoritative.
* Graph
  * Predefined source and sink nodes, id 0 and 1
    * 它们之间的控制依赖边，其 src_output, dst_input 值都为-1。
  * 边相关方法：AddEdge, RemoveEdge

#### grappler

* https://www.tensorflow.org/guide/graph_optimization TODO
  * “Op fusion” (or “Remapper Optimizer” as it’s referred to in the TensorFlow docs) is one of the many optimisations that can be applied.
    * less overhead from multiple GPU kernel launches
    * e.g. Matmul + BiasAdd + Activation pattern

```python
@tf.function
def unfused_power_four(x):
    x_squared = tf.multiply(x, x, name="x_square")
    x_cubed = tf.multiply(x, x_squared, name="x_cubed")
    x_four = tf.multiply(x, x_cubed, name="x_four")
    return x_four

@tf.function
def power_four(x):
    return tf.math.pow(x, 4)
```



* optimizers/meta_optimizer
  * https://web.stanford.edu/class/cs245/slides/TFGraphOptimizationsStanford.pdf MUSTDO
  * 公共表达式消除，常量折叠等

#### kernels

* The backbone of every op is its `Compute` method that determines what the op does with the input tensors to generate the output tensors and this is normally implemented in C++ and/or CUDA. 
  * Operations take 0 or more tensors as input and produce 0 or more tensors as output. 
  * Ops typically have an explicit, corresponding gradient op which takes the gradients of the original op’s outputs as input and returns the gradients with respect to the op’s inputs.

* 如何写op
  * allocate_xxx
  * framework/bounds_check

```c++
Tensor* out = nullptr;
OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({rank}), &out));
```

* slice_op
  * SharedSliceCommonCases: 共享buffer的两种case，is_identity和slice_dim0
  * input参数：begin_tensor and size_tensor
    * A size[i] of -1 means "all elements from begin[i] to dim_size(i)"
  * 矩阵形式的实现优化，利用prefetch
    * `port::prefetch<port::PREFETCH_HINT_T0>(&output_t(i + 1, 0));`
  * 普通case，用`functor::Slice`

#### lib/core

* status_test_util.h
  * TF_EXPECT/ASSERT/CHECK_OK

#### lib/io

* 读写TFRecord
  * record_writer
  * record_reader

#### ops

* resource_variable_ops
  * 

* state_ops
  * Ref(dtype) 表示stateful

```c++
REGISTER_OP("VariableV2")
    .Output("ref: Ref(dtype)")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape);
```



#### platform

* Refcount

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

* default/logging
  * DCHECK_EQ

* mutex

```c++
mutex_lock l(mu_);
tf_shared_lock l(mu_);

mutable mutex mu_;
```

* StrCat
  * strings::StrAppend

* selective_registration
  * TF_INIT_ON_STARTUP_IF(cond) ，利用运算符优先级
  * InitOnStartupMarker

```c++
#define TF_INIT_ON_STARTUP_IF(cond)                \
  (::std::integral_constant<bool, !(cond)>::value) \
      ? ::tensorflow::InitOnStartupMarker{}        \
      : ::tensorflow::InitOnStartupMarker {}

// Wrapper for generating unique IDs (for 'anonymous' InitOnStartup definitions)
// using __COUNTER__. The new ID (__COUNTER__ already expanded) is provided as a
// macro argument.
//
// Usage:
//   #define M_IMPL(id, a, b) ...
//   #define M(a, b) TF_NEW_ID_FOR_INIT(M_IMPL, a, b)
#define TF_NEW_ID_FOR_INIT_2(m, c, ...) m(c, __VA_ARGS__)
#define TF_NEW_ID_FOR_INIT_1(m, c, ...) TF_NEW_ID_FOR_INIT_2(m, c, __VA_ARGS__)
#define TF_NEW_ID_FOR_INIT(m, ...) \
  TF_NEW_ID_FOR_INIT_1(m, __COUNTER__, __VA_ARGS__)
```

* errors.h

```c++
string attr_shared_name;
TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "shared_name", &attr_shared_name));
if (!attr_shared_name.empty() && (attr_shared_name[0] == '_')) {
  return errors::InvalidArgument("shared_name cannot start with '_':",
                                  attr_shared_name);
}
```

* status.h
  * 

* tensor_coding

  * StringListEncoder
  * CordStringListEncoderImpl

  * Usage: 见下面ResourceHandle中的使用

```c++
void EncodeResourceHandleList(const ResourceHandle* p, int64 n,
                              std::unique_ptr<port::StringListEncoder> e) {
  ResourceHandleProto proto;
  for (int i = 0; i < n; ++i) {
    p[i].AsProto(&proto);
    e->Append(proto);
  }
  e->Finalize();
}

bool DecodeResourceHandleList(std::unique_ptr<port::StringListDecoder> d,
                              ResourceHandle* ps, int64 n) {
  std::vector<uint32> sizes(n);
  if (!d->ReadSizes(&sizes)) return false;

  ResourceHandleProto proto;
  for (int i = 0; i < n; ++i) {
    if (!proto.ParseFromArray(d->Data(sizes[i]), sizes[i])) {
      return false;
    }
    ps[i].FromProto(proto);
  }
  return true;
}
```





##### macros

* TF_ATTRIBUTE_ANNOTATE
  * 加上llvm的注解
* TF_MUST_USE_RESULT
* TF_NEW_ID_FOR_INIT id自增

#### profiler

* lib/traceme_encode

```c++
TraceMe trace_me("my_trace");
...
trace_me.AppendMetadata([value1]() {
  return TraceMeEncode({{"key1", value1}, {"key2", 42}});
});
```



#### public

* session_options
  * Env 
  * target: local / ip:port / host:port
  * config.proto

#### lib/monitoring

* gauge

```c++
# sizeof...()
static_assert(
      sizeof...(Labels) == NumLabels,
      "Mismatch between Gauge<ValueType, NumLabels> and number of labels "
      "provided in GetCell(...).");
```



#### util/cuda_solvers.cc

* GpuSolverHandle
  * 显存不够直接挂

### c

* 面向用户的 C++编程 API
* c_api.cc
  * 被 python/client 中的代码调用
  * Tf_NewSession
    * `TF_Session* TF_NewSession(TF_Graph* graph, const TF_SessionOptions* opt, TF_Status* status)`
    * graph作为参数，图实例在多个 Session 实例中共享
    * 调用 core/common_runtime: NewSession()
    * New nodes can still be added to `graph` after TF_NewSession().
  * TF_CloseSession
  * TF_SessionRun
    * The caller retains ownership of `input_values` (which can be deleted using TF_DeleteTensor). The caller also retains ownership of `run_options` and/or `run_metadata` (when not NULL) and should manually call TF_DeleteBuffer on them.
  * TF_Buffer

```c++
typedef struct TF_Buffer {
  const void* data;
  size_t length;
  void (*data_deallocator)(void* data, size_t length);
} TF_Buffer;
```



### cc



### stream_executor

```
tensorflow/stream_executor
|-- cuda
|-- gpu
|-- host
|-- lib
|-- platform
|   `-- default
|-- rocm
`-- tpu
```

* StreamExecutor 是 Google 另一个开源组件库，它提供了主机端 (host-side) 的编程模 型和运行时环境，实现了 CUDA 和 OpenCL 的统一封装。使得在主机端的代码中，可以将 Kernel 函数无缝地部署在 CUDA 或 OpenCL 的计算设备上执行。
  * 目前，StreamExecutor 被大量应用于 Google 内部 GPGPU 应用程序的运行时。其中， TensorFlow 运行时也包含了一个 StreamExecutor 的快照版本，用于封装 CUDA 和 OpenCL 的运行时。本书将简单介绍 CUDA 的编程模型和线程模型，并详细介绍 StreamExecutor 的系统架构与工作原理，揭示 Kernel 函数的实现模式和习惯用法。
  * 4w行C++

### compiler

```
tensorflow/compiler
|-- aot
|-- jit
|-- mlir
|-- plugin
|-- tests
|-- tf2tensorrt
|-- tf2xla
|-- xla
`-- xrt
```

### Tf2 vs Tf1

#### [Effective Tf2](https://www.tensorflow.org/guide/effective_tf2#overview)

* [migration guide](https://www.tensorflow.org/guide/migrate?_gl=1*1cflv2n*_ga*MTc4NDU2MTQ0My4xNjY1NDIzNTQ0*_ga_W0YLR4190T*MTY3MDI1OTI5NS44LjEuMTY3MDI2MDUwOS4wLjAuMA..)
* Refactor your code into smaller modules
* Use tf.Modules and  [`tf.keras.layers.Layer`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer)s  to manage variables

  * `variables` and `trainable_variables` properties
  * Keras layers/models inherit from `tf.train.Checkpointable` and are integrated with [`@tf.function`](https://www.tensorflow.org/api_docs/python/tf/function), which makes it possible to directly checkpoint or export SavedModels from Keras objects. You do not necessarily have to use Keras' [`Model.fit`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit) API to take advantage of these integrations.
  * e.g. [transfer learning and fine-tuning](https://www.tensorflow.org/guide/keras/transfer_learning#transfer_learning_fine-tuning_with_a_custom_training_loop)
* Combine [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)s and [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function)

  * replaces Python iteration with the equivalent graph operations using AutoGraph.

```python
@tf.function
def train(model, dataset, optimizer):
  for x, y in dataset:
    with tf.GradientTape() as tape:
      # training=True is only needed if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      prediction = model(x, training=True)
      loss = loss_fn(prediction, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

* Use Keras training loops
* Customize training and write your own loop

  *  [customizing `fit`](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit)
  * [`tf.keras.callbacks.Callback`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback).

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.02),
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10)
])

optimizer = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(inputs, labels):
  with tf.GradientTape() as tape:
    predictions = model(inputs, training=True)
    regularization_loss=tf.math.add_n(model.losses)
    pred_loss=loss_fn(labels, predictions)
    total_loss=pred_loss + regularization_loss
		gradients = tape.gradient(total_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

for epoch in range(NUM_EPOCHS):
  for inputs, labels in train_data:
    train_step(inputs, labels)
  print("Finished epoch", epoch)
```

* Take advantage of [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) with Python control flow
  * [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) provides a way to convert data-dependent control flow into graph-mode equivalents like [`tf.cond`](https://www.tensorflow.org/api_docs/python/tf/cond) and [`tf.while_loop`](https://www.tensorflow.org/api_docs/python/tf/while_loop).
  * Read the [`tf.function` guide](https://www.tensorflow.org/guide/function) for a more information.
  * Keras RNN: https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN

```python
class DynamicRNN(tf.keras.Model):

  def __init__(self, rnn_cell):
    super(DynamicRNN, self).__init__(self)
    self.cell = rnn_cell

  @tf.function(input_signature=[tf.TensorSpec(dtype=tf.float32, shape=[None, None, 3])])
  def call(self, input_data):

    # [batch, time, features] -> [time, batch, features]
    input_data = tf.transpose(input_data, [1, 0, 2])
    timesteps =  tf.shape(input_data)[0]
    batch_size = tf.shape(input_data)[1]
    outputs = tf.TensorArray(tf.float32, timesteps)
    state = self.cell.get_initial_state(batch_size = batch_size, dtype=tf.float32)
    for i in tf.range(timesteps):
      output, state = self.cell(input_data[i], state)
      outputs = outputs.write(i, output)
    return tf.transpose(outputs.stack(), [1, 0, 2]), state
  
  
  
lstm_cell = tf.keras.layers.LSTMCell(units = 13)

my_rnn = DynamicRNN(lstm_cell)
outputs, state = my_rnn(tf.random.normal(shape=[10,20,3]))
print(outputs.shape)
```

* New-style metrics and losses
  * 见tensorboard
  
* Debugging
  * Use eager execution to run your code step-by-step to inspect shapes, data types and values. Certain APIs, like [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function), [`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras), etc. are designed to use Graph execution, for performance and portability. When debugging, use [`tf.config.run_functions_eagerly(True)`](https://www.tensorflow.org/api_docs/python/tf/config/run_functions_eagerly) to use eager execution inside this code.
  * Notes:
    - [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) methods such as `fit`, `evaluate`, and `predict` execute as [graphs](https://www.tensorflow.org/guide/intro_to_graphs) with [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) under the hood.
    - When using [`tf.keras.Model.compile`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile), set `run_eagerly = True` to disable the `Model` logic from being wrapped in a [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function).
    - Use [`tf.data.experimental.enable_debug_mode`](https://www.tensorflow.org/api_docs/python/tf/data/experimental/enable_debug_mode) to enable the debug mode for [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data). Read the [API docs](https://www.tensorflow.org/api_docs/python/tf/data/experimental/enable_debug_mode) for more details.


```python
@tf.function
def f(x):
  if x > 0:
    import pdb
    pdb.set_trace()
    x = x + 1
  return x

tf.config.run_functions_eagerly(True)
f(tf.constant(1))
```

* Do not keep `tf.Tensors` in your objects
  * These tensor objects might get created either in a [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) or in the eager context, and these tensors behave differently. Always use [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor)s only for intermediate values.
  * To track state, use [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable)s as they are always usable from both contexts. Read the [`tf.Variable` guide](https://www.tensorflow.org/guide/variable) to learn more.











* Graph Mode vs Eager Execution
  * TensorFlow 2.x still allows graph mode execution as it can offer better performance and enable use of TensorFlow models in environments without an available Python interpreter, such as mobile applications. Using the `tf.keras` `model.fit` and `model.predict` functionality will use graph execution unless it is explicitly disabled with the `run_eagerly=True` argument to `model.compile`.
  * TensorFlow 2.x also includes a [`tf.autograph`](https://www.tensorflow.org/guide/intro_to_graphs#taking_advantage_of_graphs) library that converts Python and TensorFlow code to a TensorFlow graph. The `tf.function` decorator can be used to convert a Python function into a graph.


```python
# eager execution

x = [[2.]]
m = tf.matmul(x, x)
# Result of the matmul is returned immediately in Python
>> [[4.]]

# graph mode

# Define the input "placeholder" data to the computation graph and the operations that should be run
x = tf.placeholder(tf.float32)
mul = tf.matmul(x, x)
# Run the operations on some data i.e. feed data into the computation graph
with tf.Session() as sess:
    m = sess.run(mul, feed_dict={x: [[2.]]})
>> [[4.]]
```



* contrib 是第三方贡献的编程库，它也是 TensorFlow 标准化之前的实验性编程接口， 犹如 Boost 社区与 C++ 标准之间的关系。当 contrib 的接口成熟后，便会被 TensorFlow 标准化，并从 contrib 中搬迁至 core, python 中，并正式对外发布。
  * Tf2 中没有了

### Other Code

#### api_template.\__init__.py

* from tensorboard.summary._tf import summary
* estimator

```python
losses = keras.losses
metrics = keras.metrics
optimizers = keras.optimizers
initializers = keras.initializers
```

#### 轮子

* BlockingCounter
  * 参考 master_session 中的用法
* `gtl::InlinedVector<Call, 4> calls(num);`
  * 参考 master_session 中的用法

#### cc/saved_model/loader.cc

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

#### tsl/framework/allocator_registry.*

* AllocatorFactoryRegistry

```c++
// 宏的写法，用到了 __COUNTER__，保证register时命名不重复
// CPUAllocator的priority是100，可以设一个大于它的值覆盖默认项
#define REGISTER_MEM_ALLOCATOR(name, priority, factory)                     \
  REGISTER_MEM_ALLOCATOR_UNIQ_HELPER(__COUNTER__, __FILE__, __LINE__, name, \
                                     priority, factory)
```

#### 代码风格

* TF_GUARDED_BY、TF_EXCLUSIVE_LOCKS_REQUIRED
  * tsl/platform/thread_annotations.h
  * [现代C++开发之线程安全注解 by 陈硕](https://zhuanlan.zhihu.com/p/47837673)

#### tensorflow-serving

* [experimental/.../ops/remote_predict](https://github.com/tensorflow/serving/tree/2e7ca90c18f310c542ed0dcde92d676db6454285/tensorflow_serving/experimental/tensorflow/ops/remote_predict)
  * 写死了server的ip:port


### 总体介绍

![image-20221201011141948](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/tensorflow/tensorflow-stack.png)

* [一篇很好的回答，梳理tf的核心框架思路](https://www.zhihu.com/question/51216952/answer/124708405) MUSTDO
* tf guides https://www.tensorflow.org/guide/basics




#### 图优化相关

计算图中的每个节点->对应一个tensorflow op->在不同的device上对应不同kernel->gpu kernel调用cuda/cudnn/cublas进行gpu计算

![constant_folding](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/tensorflow/constant_folding.png)

* [XLA](https://www.tensorflow.org/xla)
  * 优化内容
    * 设备无关优化
      * Common expression elimination等
      * Op fusion
      * Memory allocation
    * 设备相关优化和代码生成
  * enabled by [setting an environment variable](https://www.tensorflow.org/xla#auto-clustering) or controlled more explicitly by passing the `jit_compile=True` argument to the `tf.function` decorator.
    * `TF_XLA_FLAGS="--tf_xla_auto_jit=2"`
  * jit编译：`XlaCompile Op` --> `XlaRun Op` ，warmup缓存以输入shape为key的图
* Tvm
  * tf与tvm的协同：用custom_op加载tvm生成的子图
  * tvm schedule严格安排了多线程的利用，如果引入inter_op并行，会破坏schedule的安排，所以思路是做单个op的充分并行
* [Inter-op/Intra-op parallelism](https://stackoverflow.com/questions/41233635/meaning-of-inter-op-parallelism-threads-and-intra-op-parallelism-threads)
  * Intra是单个op并行；inter是多个无相互依赖的op共享thread pool并行



#### 基本 API 用法

##### 写模型

* placeholder
  * None 表示未确定的样本数目，此处表示 batch_size 的大小;当 Session.run 时，将通过 feed_dict 的字典提供一个 mini-batch 的 样本数据集，从而自动推导出 tf.placeholder 的大小
* 单层感知器

```python
# Create the model
x = tf.placeholder(tf.float32, [None, 784]) 
W = tf.Variable(tf.zeros([784, 10]))
logits = tf.matmul(x, W)
y = tf.nn.softmax(logits)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 防止自己log(0)导致NAN（softmax输出值过小）
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits( logits=logits, labels=y_)

learning_rate = 0.5
W_grad = tf.gradients(cross_entropy, [W])[0]
train_step = tf.assign(W, W - learning_rate * W_grad)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003) # train_step = optimizer.minimize(cross_entropy)

is_correct = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# tf1
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
  
with tf.Session() as sess:
	for step in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, t: batch_ys})
	if step % 100 == 0:
		acc, loss = sess.run([accuracy, cross_entropy],
			feed_dict={x: batch_xs, t: batch_ys})
		acc, loss = sess.run([accuracy, cross_entropy],
			feed_dict={x: mnist.test.images, t: mnist.test.labels})
```

* 多层感知器

```python
K = 200
L = 100
M = 60
N = 30
w1 = tf.Variable(tf.truncated_normal([28*28, K] ,stddev=0.1))
b1 = tf.Variable(tf.zeros([K]))
w2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
b2 = tf.Variable(tf.zeros([L]))
w3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
b3 = tf.Variable(tf.zeros([M]))
w4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
b4 = tf.Variable(tf.zeros([N]))
w5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
b5 = tf.Variable(tf.zeros([10]))

y1 = tf.nn.sigmoid(tf.matmul(x,  w1) + b1)
y2 = tf.nn.sigmoid(tf.matmul(y1, w2) + b2)
y3 = tf.nn.sigmoid(tf.matmul(y2, w3) + b3)
y4 = tf.nn.sigmoid(tf.matmul(y3, w4) + b4)
y  = tf.nn.softmax(tf.matmul(y4, w5) + b5)

# sigmoid替换为relu
b1 = tf.Variable(tf.ones([L])/10)
y1 = tf.nn.relu(tf.matmul(x,  w1) + b1)
```

* Adam Optimizer，lr decay

```python
lr = tf.placeholder(tf.float32)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

def lr(step):
	max_lr, min_lr, decay_speed = 0.003, 0.0001, 2000.0
	return min_lr + (max_lr - min_lr) * math.exp(-step/decay_speed)

with tf.Session() as sess:
  for step in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,
             feed_dict={x: batch_xs, t: batch_ys, pkeep: 0.75, lr: lr(step)})
```

* conv

```python
w1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))
y1 = tf.nn.relu(tf.nn.conv2d(x,  w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
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

##### 实用op

```python
features['label'] = tf.where(
            tf.math.equal(tf.reshape(features['actions'], shape=(-1,)), 2),
            tf.ones(shape=(batch_size,), dtype=tf.float32),
            tf.zeros(shape=(batch_size,), dtype=tf.float32))

x = tf.reshape(x, [-1, 784])
```



##### gradients

* [about grad_ys](https://stackoverflow.com/questions/50967885/tf-gradients-how-can-i-understand-grad-ys-and-use-it)

```python
# gradients 返回 grad_ys*dy/dx，ys和xs是tensor
with tf.name_scope(""):
        with tf.name_scope("Gradients/%s/" % scope_name):
            gs = tf.gradients(ys=ys, xs=xs, name=name, colocate_gradients_with_ops=colocate_gradients_with_ops, grad_ys=grad_ys)
```

* replace_gradients_zero

```python
@tf.custom_gradient
def replace_gradient_zero(tensor, gradient):
    def grad(dy):
        return tf.cast(gradient, tensor.dtype), None
    return tf.cast(tf.zeros_like(tensor), tf.float32), grad
```

##### variable

* [Understanding variable_scope and name_scope in tensorflow and variable sharing](https://stackoverflow.com/questions/36237427/understanding-variable-scope-and-name-scope-in-tensorflow-and-variable-sharing)

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

##### mask input

tf.contrib.graph_editor: [consumers](https://www.kite.com/python/docs/tensorflow.contrib.graph_editor.SubGraphView.consumers)

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

##### tensor

https://www.tensorflow.org/guide/tensor

###### ragged_tensor

* 在 TensorFlow Extended 上快速高效地部署 BERT 模型（下） - 谷歌开发者的文章 - 知乎 https://zhuanlan.zhihu.com/p/270339910
  * RaggedTensors 的实现在 NLP 应用中尤为实用。例如，在将语句的一维数组标记化为具有不同数组长度的二维 RaggedTensor 时，该张量类型便能发挥用处。



### TensorFlow Internals

#### chpt 1 介绍

* 概念：数据流图、DAG、本地设备集

* DistBelief: 异步SGD（model replicas），主从结构

* TensorFlow: 
  * 延迟计算
  * 原子OP
  * 抽象设备（CPU、GPU、ASIC）
  * 抽象任务（基于任务的PS）


#### chpt 2 编程环境

#### chpt 3 破冰之旅

* 单层/多层感知器，见【基本 API 用法】
* 卷积网络，见【ML笔记】

#### chpt 4 系统架构

![image-20221212015113480](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/tensorflow/tf-arch.png)

* 图操作的角度
  * 表达图:构造计算图，但不执行图
  * 编排图:将计算图的节点以最佳的执行方案部署在集群中各个计算设备上执行
  * 运行图:按照拓扑排序执行图中的节点，并启动每个 OP 的 Kernel 计算
* Client：将 Protobuf 格式的 GraphDef 序列化后传递给 Master，启动计算图的执行过程
  * 调用 Master 的 CreateSession 方法，传递 GraphDef
  * CreateSessionResponse 中存了 graph handle
  * Client 会启动迭代执行的过程，称每次迭代为一次 RunStep

* Master：
  * 根据 Session.run 传递给它的 fetches, feeds 参数列表，反向遍历 Full Graph，并按照依赖关系，对其实施剪枝，最终计算得到最小的依赖子图，常称为 Client Graph
  * SplitByTask，将 Graph Partition 分别注册到相应的 Worker 上，再通知 Worker 启动运算
    * 其中，Worker 之间可能存在数据依赖关系，Master 并不参与两者之间的数据交换，它们两两之间互相通信，独立地完成交换数据，直至完成所有计算
  * Master 完成子图注册后，将广播所有 Worker 并发执行所有子图。这个过程是通过 Master 发送 RunGraphRequest 消息给 Worker 完成的。其中，消息中携带 (session_handle, graph_handle, step_id) 三元组的标识信息，用于 Worker 索引相应的子图。

```python
def run_partitions(rendezvous, executors_and_partitions, inputs, outputs): 
	rendezvous.send(inputs)
	for (executor, partition) in executors_and_partitions:
    executor.run(partition)
    rendezvous.recv(outputs)
```

* Worker:
  * 处理来自 Master 的请求;
  * 对注册的 Graph Partition 按照本地计算设备集实施二次分裂 (SplitByDevice)，
  * 并通知各个计算设备并发执行各个 Graph Partition;
  * 按照拓扑排序算法在某个计算设备上执行本地子图，并调度 OP 的 Kernel 实现;
  * 协同任务之间的数据通信。 TODO 找到这段代码
    * 本地 CPU 与 GPU 之间，使用 cudaMemcpyAsync 实现异步拷贝;
    * 本地 GPU 之间，使用端到端的 DMA 操作，避免主机端 CPU 的拷贝
* Kernel:
  * Kernel 是 OP 在某种硬件设备的特定实现，它负责执行 OP 的具体运算
    * 包括数值计算，多维数组操作，控制流，状态管理等
* 图控制
  * 图分裂、子图控制
    * 插入 Send 和 Recv 节点
    * Master 通过调用 RegisterGraph 接口，将子图注册给相应的 Worker 上，并由相应的 Worker 负责执行运算

#### chpt 5 C API: 分水岭

* [Replace SWIG with pybind11](https://github.com/tensorflow/community/blob/master/rfcs/20190208-pybind11.md)
  * `tf_python_pybind_extension`
* pybind11 https://pybind11.readthedocs.io/en/stable/index.html
  * `py::handle` - equivalent to `PyObject*` (no automatic refcounting)
  * `py::object` - does automatic refcounting. Subclass of `py::handle`.
* 讲了一些session的源码

#### chpt 6 计算图

* Python前端: python/framework/ops
  * Operation 表示图中的 Node 实例，而 Tensor 表示 图中的 Edge 实例
  * 有向边存在两种类型，一种承载数据，并使用 Tensor 表示;另一种不承载数据，仅表示计算依赖关系
  * Operation 的元数据由 OpDef 与 NodeDef 持有，它们以 ProtoBuf 的格式存在，它描述 了 Operation 最本质的东西。其中，OpDef 描述了 OP 的静态属性信息，例如 OP 的名称， 输入/输出参数列表，属性集定义等信息。而 NodeDef 描述了 OP 的动态属性值信息，例如 属性值等信息

![image-20221228025418798](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/tensorflow/op.png)

* 图构造
  * 模块 gen_array_ops 是构建版本时自动生成的，它主要完成所有 array_ops 类 型的 OpDef 的定义，并自动注册到 OpDefLibrary 的仓库实例中，并提供按名查找 OpDef 的 服务接口
  * Graph: Op工厂+Op仓库
  * python_op_gen_*
* op相关，参考【如何写一个op】

#### chpt 7 设备

设备规范 (Device Specification) 用于描述 OP 存储或计算设备的具体位置

* 形式化
  * 完整指定：`/job:ps/replica:0/task:0/device:GPU:0`
  * 部分指定：`/device:GPU:0`
  * 同位：`@other/node  # colocate with "other/node"`
  * `DeviceSpec(job="ps", replica=0, task=0, device_type="CPU", device_index=0)`

```python
DEVICE_SPEC ::= COLOCATED_NODE | PARTIAL_SPEC
COLOCATED_NODE ::= "@" NODE_NAME
PARTIAL_SPEC ::= ("/" CONSTRAINT) * 
CONSTRAINT ::= ("job:" JOB_NAME)
              | ("replica:" [1-9][0-9]*)
              | ("task:" [1-9][0-9]*)
              | ( ("gpu" | "cpu") ":" ([1-9][0-9]* | "*") )
```

* 上下文管理器
  * 实现设备规范的闭包、合并、覆盖等特性
    * 内部指定的优先级更高
    * 重置：None
  * 设备规范函数

`def device(device_name_or_function):
 return get_default_graph().device(device_name_or_function)`

```python
def matmul_on_gpu(n):
 if n.type == "MatMul":
   return "/gpu:0"
 else:
   return "/cpu:0"
with g.device(matmul_on_gpu):
  # All OPs of type "MatMul" constructed in this context
  # will be placed on GPU 0; all other OPs will be placed
  # on CPU 0.
```

* 实现：参考python/framework/ops

#### chpt 8 会话 Session



#### chpt 10 队列 Queue

```python
class XMonitor():
  """A monitor that use to detect X status."""

  def __init__(self, num):
    self._queues = {}
    self._enqueue_ops = {}
    self._qsize_ops = {}
    for i in range(num):
      device = X.device(i)
      with tf.device(device):
        queue = tf.queue.FIFOQueue(1,
                                   tf.int32,
                                   shared_name="X_" + str(i))
        self._queues[device] = queue
        self._enqueue_ops[device] = queue.enqueue(1)
        self._qsize_ops[device] = queue.size()

  def is_X_uninitialized(self, sess, device):
    if device in self._qsize_ops:
      return sess.run(self._qsize_ops[device]) == 0
    return True

  def setup_X_initialized_state(self, sess):
    for device in self._queues.keys():
      if sess.run(self._qsize_ops[device]) == 0:
        sess.run(self._enqueue_ops[device])
```





### Inside Tensorflow 系列

#### tf.data



* 利用pipeline优化：https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras#debug_performance_bottlenecks



#### [Parameter server training](https://www.youtube.com/watch?v=B2Tpv_N7wkg&list=PLQY2H8rRoyvzIuB8rZXs7pfyjiSUs8Vza&index=1)

* PS training overview
  * Async training and gradient staleness
    * More workers => more stale gradients
    * adaptive learning rate
      * Discount lr based on the staleness
  * Sync training
    * Tf1: SyncReplicasOptimizer
      * ConditionalAccumulator
      * use a queue as a barrier, but hangs in the case of task preemption
      * N-K backup workers, 丢弃多余的gradients
  * evaluation by estimator
    * Worker write ckpt, estimator load kept
    * 生产环境中有一些缺点（ckpt频率）
* Single-Client Distributed Training
  * Multi-client setup in tf1的缺点
    * worker间难协同（同步训练、early stopping、variable creation不重复）
    * less intuitive programming model
    * 不同worker create不同的nccl ops，导致cluster hang
  * Tf2: single-client setup
    * user program in coordinator，能克服一些multi-client setup的缺点
    * problem
      * ParameterServerStrategy下，single coordinator是否成为瓶颈
      * single point of failure
      * limitation with remote functions
* Single-Client APIs
  * Thread-pool like programming model
    * easy for load balance, fault tolerance and dynamic scaling
    * At-least-once: put back functions to coordinator
    * class ClusterCoordinator: work in conjunction with tf.distribute.Strategy
    * function本质上是multi-device function
  * Per-worker dataset
    * `coordinator.create_per_worker_dataset(dataset_fn)`
  * limitations
    * no visitation guarantee
    * ps preemption needs the coordinator to restart
    * performance
      * threading & GIL overhead, latency
    * can only shift tf.function

![image-20221207040106427](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/tensorflow/training-loop.png)

![image-20221207040138897](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/tensorflow/training-loop-ps.png)

* Inline Evaluation
  * 在coordinator或者和训练相同的worker pool上跑，用户友好
    * 在coordinator上跑很省事
    * 在worker上跑
      * no visitation guarantee
        * Possible solutions: tf.data.service, virtual shards
      * At-least-once but not exactly-once
* Variable Sharding
  * ShardedVariable 配合 tf.nn.embedding_lookup
  * converted to tensor via concatenation
  * 是一种 model parallelism
  * future work
    * packed representation
* PSStrategy in Eager Runtime
  * distributed functions in ps training
  * send/recv ops
* Performance Measurement and Improvement
  * ps-strategy-1的性能和estimator有差距，来自于 network RTT 和 python GIL
  * Multi-step packing
    * ps-strategy-5: pack five training steps in one function (通常5-10个step)
    * TPUStrategy 类似的有 “host training loop” suggestion
  * pros and cons
    * Pros: 性能、无代码改动迁移 (keras compile/fit)
    * Cons: 
      * failure forfeits more work than single-step function
      * Less fine granularity of train steps, return values and metrics
      * 代码改动（custom training loop）

```python
@tf.function
def train_fn():
  for _ in tf.range(N):
    model_fn()
    
for _ in tf.range(N):
  coord.schedule(train_fn)
```

* Scalability and Fault Tolerance
  * Preemptions 占绝大多数 failure
    * 解决这一问题，能用上preemptible resources，降低60%的成本
  * 要求
    * fast failover
      * `UpdateServerDef`
    * efficient recovery: rebuild runtime state automatically when worker rejoin
* Continuous Multi-Worker Testing Framework

![image-20221207175516465](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/tensorflow/multi-work-testing-framework.png)

#### [Resources and Variants](https://www.youtube.com/watch?v=uaRO0AV6Tto&list=PLQY2H8rRoyvzIuB8rZXs7pfyjiSUs8Vza&index=16)

* State: "an op is stateful if either executing it has a side effect or if its output depends on sth other than the value of its input."
  * kernel需要标注为stateful -> disable constant folding (buggy with datasets)
  * sth go either way (tf2 fix了一些不必stateful的op)
    * Stacks: tf1里有状态，tf2里无状态
    * tensor lists: 同上
  * SetIsStateful() when REGISTER_OP
    * no constant folding
    * no common subexpression elimination
    * Kernel instances reused across sessions (OpSegment)
    * a few more obscure behavior changes
      * ParallelFor不太一样

```python
# Statefulness in practice

tf.print("Print is stateful")

d = tf.data.Dataset.from_tensor_slices(["datasets are not stateful"])

iter(d) # Iterators are stateful
```

* Variables
  * output:Ref(dtype) 也表示stateful，目的是persistent across session.run calls

![image-20230114014540672](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/tensorflow/variable-example.png)

* 一个intricate例子
  * same device + assign不改变variable shape
    * Print the value after assignment
  * add和variable不在一个device上 + assign改变variable shape
    * 会看到assignment之前的value
    * 因为identity看到old TensorBuffer 
  * 涉及string类型的时候会seg fault

* ResourceMgr
  * RTTI
  * DT_RESOURCE相关的逻辑：ColocateResourceAndRefEdges、op_def_builder的set_is_stateful
  * 如何优雅实现read->compute->assign的pattern
    * 实现
      * COW for dense
      * copy-on-read for embeddings
      * Internals: Var中的copy_on_read_mode属性
    * 优化：XLA
      * ResourceGatherOp

![image-20230117023755071](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/tensorflow/variable-example-2.png)



#### [Summaries and TensorBoard](https://www.youtube.com/watch?v=OI4cskHUslQ&list=PLQY2H8rRoyvzIuB8rZXs7pfyjiSUs8Vza&index=15)

* 一些特性
  * slice and dice data by run and tag

* New-style metrics and losses
  * `pip install -U tensorboard-plugin-profile`
  * `tensorboard --logdir /tmp/summaries --bind_all`
  * For more info, read the [`tf.summary` guide](https://www.tensorflow.org/tensorboard/migrate#in_tf_2x).
  


```python
summary_writer = tf.summary.create_file_writer('/tmp/summaries')
with summary_writer.as_default():
  tf.summary.scalar('loss', 0.1, step=42)
  
def train(model, optimizer, dataset, log_freq=10):
  avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
  for images, labels in dataset:
    loss = train_step(model, optimizer, images, labels)
    avg_loss.update_state(loss)
    if tf.equal(optimizer.iterations % log_freq, 0):
      tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
      avg_loss.reset_states()
```

```python
model.compile(
    optimizer = tf.keras.optimizers.Adam(0.001),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['acc', 'accuracy', tf.keras.metrics.SparseCategoricalAccuracy(name="my_accuracy")])
history = model.fit(train_data)


history.history.keys()
# dict_keys(['loss', 'acc', 'accuracy', 'my_accuracy'])
```

* Trace viewer
  * `WASD` 控制界面、`1234` 控制模式、`M` measure
  * Which part of the system (host or device) executed an op. Typically, the host executes input operations, preprocesses training data and transfers it to the device, while the device executes the actual model training
* Tensorflow Stats
  * self time: measures the time spent inside the function body, excluding the time spent in the function it calls.
* 一些有用的信息
  * 权重分布图

##### tf.summary

* 本质：structured logging for your model
  * Instrumentation: tf.summary.scalar(), histogram(), image(), audio() and text()
  * Writing: ->disk->TensorBoard
* framework calls it: Estimator、Keras
* Most data gets to tensorboard this way
  * 例外：tfdbg

![image-20230108032949430](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/tensorflow/tf-summary-behavior.png)

* tf.summary in TF 1.x
  * Limitation 1: 支持数据类型少，改造成本高（proto、op、kernel、python API）
  * Limitation 2: data flow through the graph，merge_all利用data_collection
    * unsafe to use inside control flow/functions

```python
tf.summary.scalar("foo", 0.5)
...
merge_op = tf.summary.merge_all()
...
summary = sess.run(merge_op)
w = tf.summary.FileWriter("logs")
w.add_summary(summary, step)
```

* tensorboard.summary in TF 1.x
  * Tensor itself as the generic data container
  * Python logic + tf.summary.tensor_summary()
  * 结果：uptake for new ops (like pr_curve), not for existing ones
* tf.contrib.summary in TF 1.x
  * summary_ops_v2.py
  * instrumentation ops are now stateful
    * C++ writer resource
  * Python writer objects mostly but not completely manage state
    * C++ resource shared by multiple Python objects
    * not quite TF 2.0 paradigm of 1:1 python state to runtime state

```python
tf.enable_eager_execution()
tfcs = tf.contrib.summary
w = tfcs.create_file_writer("logs")
with w.as_default():
  with tfcs.always_record_summaries():
    tfcs.scalar("foo", 0.5, step=1)
```

* tf.summary in TF 2.0
  * Fuse了前面的，tf.contrib.summary的逻辑、tb.summary的generic tensor data format
    * glue + circular import fixes so the two halves actually interoperate
    * 具体操作：
      * original tf.summary exposes writing APIs like create_event_writer()
      * tensorboard shim module does the merge
        * Wildcard import of tf.summary
        * Regular imports of TensorBoard-defined instrumentation APIs
      * api_template.\__init__.py import tb.summary as tf.summary
    * Result: single module with both sets of symbols
  * Writers map 1:1 to resources/event files; no more resource sharing

```python
# eager execution
w = tf.summary.create_file_write("logs")
with w.as_default():
  for step in range(100):
    ...
    tf.summary.scalar("foo", 0.5, step=1)
    ...
    w.flush()
    # 默认每次离开default ctx会flush一次
    
# tf.function
w = tf.summary.create_file_write("logs")

@tf.function
def my_func(step):
  with w.as_default():
    tf.summary.scalar("foo", 0.5, step=1)
    
for step in range(100):
  my_func(step)
  w.flush()
  
# legacy graph execution
with tf.compat.v1.Graph().as_default():
  w = tf.summary.create_file_write("logs")
  with w.as_default():
    tf.summary.scalar("foo", 0.5, step=1)
  
  sess = tf.compat.v1.Session()
  sess.run(w.init())
  sess.run(tf.compat.v1.summary.\
          		all_v2_summary_ops())
  sess.run(w.flush())
```

* summary data format: Logdirs, event files, and more

* Event file format - TFRecord
  * 支持zlib/snappy compression
  * 无法seek ahead

```
// Format of a single record:
//  uint64    length
//  uint32    masked crc of length
//  byte      data[length]
//  uint32    masked crc of data
```

##### Best Practices

* group relevant data in subdir
* --reload_interval=0
* avoid remote filesystems overhead
* avoid multiple writers for the same subdir



##### Internals

![img](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/tensorflow/tensorboard-arch.png)

![image-20230108032257999](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/tensorflow/tensorboard-webfrontend.png)

* TensorBoard maps logdir structure to "runs"
  * define "run": direct children中至少有一个event file
  * define "event file":  名字中含有tfevents
* Logdir traversal
  * polls the logdir for new data, --reload_interval
  * First pass: finding new runs
    * avoid slow walk() on some filesystems(GCS) via glob()ing
  * Second pass: reading new event file data from each run, in series
* Reading event files
  * 按creation order遍历，对于每个文件，sequentially read直到EOF
  * new reload cycle, read from same offset
  * TensorBoard never revisits earlier files within a run
    * 假设last file是唯一active的file，避免重复check
    * 某些情况比如multiple writer，可能需要重启tb

* load summary data into memory
  * raw logs不支持随机存取
  * 内存中index data by run, plugin, and tag
  * downsampling data优化内存
  * Reservoir sampling: pool size k, add nth item with probability k/n
    * tune k with `--samples_per_plugin`

### AddOns

* https://github.com/tensorflow/addons/
* optimizers/lazy_adam.py
  * 对于sparse updates效率比adam高

### Profiling

* [Tf2 Profiler](https://www.tensorflow.org/guide/profiler)
  * https://github.com/tensorflow/profiler
  * https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras
  * [Performance profiling in TF 2 (TF Dev Summit '20)](https://www.youtube.com/watch?v=pXHAQIhhMhI)
  * 多种使用方式：keras callback、profile API in loops、profile server

```python
options = tf.profiler.experimental.ProfilerOptions(
          host_tracer_level=2,
          python_tracer_level=1,
          device_tracer_level=1)
tf.profiler.experimental.start(logdir, options)
```

```python
for step in range(NUM_STEPS):
    with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
        train_data = next(dataset)
        train_step(train_data)
```

* sampling mode

```python
# Start a profiler server before your model runs.
tf.profiler.experimental.server.start(6009)
# (Model code goes here).

#  Send a request to the profiler server to collect a trace of your model.
tf.profiler.experimental.client.trace('grpc://localhost:6009',
                                      'gs://your_tb_logdir', 2000)

# Profiling multiple workers
# E.g. your worker IP addresses are 10.0.0.2, 10.0.0.3, 10.0.0.4, and you
# would like to profile for a duration of 2 seconds.
tf.profiler.experimental.client.trace(
    'grpc://10.0.0.2:8466,grpc://10.0.0.3:8466,grpc://10.0.0.4:8466',
    'gs://your_tb_logdir',
    2000)

```



* TraceMe

```c++
profiler::TraceMe activity([]() { return "Name"; });
...
{
 	profiler::TraceMe activity([]() { return "Name"; });
  ...
}


profiler::TraceMe trace_me([this] {
    return profiler::TraceMeEncode("Name",
                                   {{"tag", tag_value}});
  });
```

* 一些性能优化建议
  * tf.config.threading
  * When working with smaller models on NVIDIA® GPUs, you can set `tf.compat.v1.ConfigProto.force_gpu_compatible=True` to force all CPU tensors to be allocated with CUDA pinned memory to give a significant boost to model performance. However, exercise caution while using this option for unknown/very large models as this might negatively impact the host (CPU) performance.

* issue
  * Step-time Graph 不显示：https://github.com/tensorflow/profiler/issues/282
    * solution: `with tf.profiler.experimental.Trace("TraceContext", graph_type="train", step_num=step): train_fn()`



### Networking

* seastar
  * [github link](https://github.com/tensorflow/networking/tree/master/tensorflow_networking/seastar)



### TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems [2015]

node对应operation

edge对应tensor

* 也可是control dependencies，应用：控制算法、control the peak memory usage

operations and kernels

* attributes -> make operations polymorphic 
* kernel: a particular implementation of an operation that can be run on a particular type of device (e.g., CPU or GPU)
* 定义operations and kernels: registration mechanism

Sessions: 支持Extend和Run

Variables: a special kind of operation that returns a handle to a persistent mutable tensor that survives across executions of a graph. Handles to these persistent mutable tensors can be passed to a handful of special operations, such as `Assign` and `AssignAdd` (equivalent to +=) that mutate the referenced tensor. 

#### 3.Implementation

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



### TensorFlow: A system for large-scale machine learning [OSDI, 2016]

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