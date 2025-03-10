Q: DDP中buffer是如何sync的?
Q: 混合精度训练 + DDP是如何work的？
_module_wait_for_copy_hook
_fire_reducer_autograd_hook


DP&DDP源码解析 https://zhuanlan.zhihu.com/p/343951042


原理也参考「MLSys.md - 并行训练」


### DP

nn/parallel/data_parallel.py

class DataParallel(Module):

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__()

        # 检查是否有可用的 GPU
        device_type = _get_available_device_type()
        if device_type is None:
            self.module = module
            self.device_ids = []
            return
        
        # 默认使用所有可见的 GPU
        if device_ids is None:
            device_ids = _get_all_device_indices()

        # 默认 server 是 device_ids 列表上第一个
        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device(device_type, self.device_ids[0])

        # 检查负载是否平衡， 不平衡（指内存或者处理器 max/min > 0.75 会有警告）
        if device_type == "cuda":
            _check_balance(self.device_ids)

        # 单卡
        if len(self.device_ids) == 1:
            self.module.to(self.src_device_obj)

    def forward(self, *inputs, **kwargs):

        # 没 GPU 可用
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        # 运行前 GPU device_ids[0] （即我们的 server ）上必须有 parallelized module 的parameters 和 buffers
        # 因为 DP 保证 GPU device_ids[0] 和 base parallelized module 共享存储
        # 所以在device[0] 上的 in-place 更新也会被保留下来，其他的则不会

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        # nice 现在 device[0] 上已经有了 module 和 input， 接下来我们就要开始 PS 算法了
        # 可以开始看正文了

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)

        # 如果仅有单卡可用，直接单卡计算，不用并行
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])

        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids, not torch.is_grad_enabled())

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""

    # 主要函数
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []

    # 用空项补全使 inputs 和 kwargs 长度相当
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    # 返回 tuple
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs

# scatter_gather.py

scatter 函数，负责将 tensor 分成大概相等的块并将他们分给不同的 GPU。对其他的数据类型，则是复制分散给不同的 GPU 。


def scatter(inputs, target_gpus, dim=0):
    r"""Slice tensors into approximately equal chunks and distributes them across given GPUs.

    Duplicates references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if _is_namedtuple(obj):
            return [type(obj)(*args) for args in zip(*map(scatter_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(scatter_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]
        return [obj for _ in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None  # type: ignore[assignment]
    return res

class Scatter(Function):
    @staticmethod
    def forward(ctx, target_gpus, chunk_sizes, dim, input):
        target_gpus = [_get_device_index(x, True) for x in target_gpus]
        ctx.dim = dim
        ctx.input_device = input.get_device() if input.device.type != "cpu" else -1
        streams = None
        if torch.cuda.is_available() and ctx.input_device == -1:
            # Perform CPU to GPU copies in a background stream
            streams = [
                _get_stream(torch.device("cuda", device)) for device in target_gpus
            ]
        outputs = comm.scatter(input, target_gpus, chunk_sizes, ctx.dim, streams)
        # Synchronize with the copy stream
        if streams is not None:
            for i, output in enumerate(outputs):
                with torch.cuda.device(target_gpus[i]):
                    main_stream = torch.cuda.current_stream()
                    main_stream.wait_stream(streams[i])
                    output.record_stream(main_stream)
        return outputs

    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, Gather.apply(ctx.input_device, ctx.dim, *grad_output)

class BroadCast
    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None,) + ReduceAddCoalesced.apply(ctx.input_device, ctx.num_inputs, *grad_outputs)

def replicate(network, devices, detach=False):
    # 需要复制到哪些 GPU， 复制多少份
    devices = [_get_device_index(x, True) for x in devices]
    num_replicas = len(devices)

    # 复制 parameters
    params = list(network.parameters())
    param_indices = {param: idx for idx, param in enumerate(params)}

    param_copies = _broadcast_coalesced_reshape(params, devices, detach)
    ...

    modules = list(network.modules())
    module_copies: List[List[Module]] = [[] for _ in devices]
    module_indices: Dict[Module, int] = {}

    for i, module in enumerate(modules):
            module_indices[module] = i
            for j in range(num_replicas):
                replica = module._replicate_for_data_parallel()
                # This is a temporary fix for DDP. DDP needs to access the
                # replicated model parameters. It used to do so through
                # `mode.parameters()`. The fix added in #33907 for DP stops the
                # `parameters()` API from exposing the replicated parameters.
                # Hence, we add a `_former_parameters` dict here to support DDP.
                replica._former_parameters = OrderedDict()

                module_copies[j].append(replica)

    # 接下来分别复制 module，param，buffer

    for i, module in enumerate(modules):
        for key, child in module._modules.items():
            if child is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._modules[key] = None
            else:
                module_idx = module_indices[child]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    setattr(replica, key, module_copies[j][module_idx])
        for key, param in module._parameters.items():
            if param is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._parameters[key] = None
            else:
                param_idx = param_indices[param]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    param_copy = param_copies[j][param_idx]
                    # parameters in replicas are no longer leaves,
                    # so setattr them as non-parameter attributes
                    setattr(replica, key, param_copy)
                    # expose the parameter for DDP
                    replica._former_parameters[key] = param_copy
        for key, buf in module._buffers.items():  # type: ignore[assignment]
            if buf is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._buffers[key] = None
            else:
                if buf.requires_grad and not detach:
                    buffer_copies = buffer_copies_rg
                    buffer_idx = buffer_indices_rg[buf]
                else:
                    buffer_copies = buffer_copies_not_rg
                    buffer_idx = buffer_indices_not_rg[buf]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    setattr(replica, key, buffer_copies[j][buffer_idx])

    return [cast(T, module_copies[j][0]) for j in range(num_replicas)]


返回的object：
module.py

def _replicate_for_data_parallel(self):
        replica = self.__new__(type(self))
        replica.__dict__ = self.__dict__.copy()

        # replicas do not have parameters themselves, the replicas reference the original
        # module.
        replica._parameters = {}
        replica._buffers = replica._buffers.copy()
        replica._modules = replica._modules.copy()
        replica._is_replica = True  # type: ignore[assignment]

        return replica

# parallel_forward.py

# DP 代码
outputs = self.parallel_apply(replicas, inputs, kwargs)

# threading 实现，用前面准备好的 replica 和输入数据，然后
# for 循环启动多线程

# 源码
def parallel_apply(modules, inputs, kwargs_tup=None, devices=None):

        # 每个 GPU 都有模型和输入
    assert len(modules) == len(inputs)

    # 确保每个 GPU 都有相应的数据，如没有就空白补全
    if kwargs_tup is not None:
      # 咱们在 scatter 已经补全了
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)

    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)

    devices = [_get_device_index(x, True) for x in devices]

    # 多线程实现

    lock = threading.Lock()
    results = {}
    grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()

    # 定义 worker
    def _worker(i, module, input, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device), autocast(enabled=autocast_enabled):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module(*input, **kwargs)
            with lock:
              # 并行计算得到输出
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:

      # 如有一个进程控制多个 GPU ，起多个线程
      # 需要强调一下，虽然 DDP 推荐单卡单进程，即每次调用 DDP device_ids 都只输入一张卡的 id（通常是 args.local_rank），但是如果输入多个 device_id，此时 DDP 就是单进程多线程控制多卡，和 DP 一样，关于 DDP 的解读可以看下文

        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device))
                   for i, (module, input, kwargs, device) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
      # 一个 GPU 一个进程 （ DDP 推荐操作）
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]

        # error handle
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    # 输出 n 个计算结果
    return outputs



### DDP

DDP 通过 Reducer 来管理梯度同步。为了提高通讯效率， Reducer 会将梯度归到不同的桶里
（按照模型参数的 reverse order， 因为反向传播需要符合这样的顺序），一次归约一个桶。其中桶的大小为参数 bucket_cap_mb 默认为 25，可根据需要调整。

DDP 通过在构建时注册 autograd hook 进行梯度同步。
反向传播时，当一个梯度计算好后，相应的 hook 会告诉 DDP 可以用来归约。
当一个桶里的梯度都可以了，Reducer 就会启动异步 allreduce 去计算所有进程的平均值。
allreduce 异步启动使得 DDP 可以边计算边通信，提高效率。当所有桶都可以了，Reducer 会等所有 allreduce 完成，然后将得到的梯度写到 param.grad。

Design Note: https://pytorch.org/docs/main/notes/ddp.html


代码架构：
distributed.py
--
reducer.h, comm.h
--
ProcessGroup.hpp (NCCL, GLOO, MPI, RR)


Prerequisite:
DDP relies on c10d ProcessGroup for communications. 
Hence, applications must create ProcessGroup instances before constructing DDP.

Construction:
The DDP constructor takes a reference to the local module, and broadcasts state_dict() from the process with rank 0 to all other processes in the group
to make sure that all model replicas start from the exact same state. 
Then, each DDP process creates a local Reducer, which later will take care of the gradients synchronization during the backward pass. 
To improve communication efficiency, the Reducer organizes parameter gradients into buckets, and reduces one bucket at a time. 
Bucket size can be configured by setting the bucket_cap_mb argument in DDP constructor. 
The mapping from parameter gradients to buckets is determined at the construction time, based on the bucket size limit and parameter sizes.


 Model parameters are allocated into buckets in (roughly) the reverse order of Model.parameters() from the given model. 
 The reason for using the reverse order is
 - because DDP expects gradients to become ready during the backward pass in approximately that order
 - DDP requires Reducer instances on all processes to invoke allreduce in exactly the same order,
   which is done by always running allreduce in the bucket index order instead of actual bucket ready order. Mismatched allreduce order across processes can lead to wrong results or DDP backward hang.

还注册了autograd hooks during construction, one hook per parameter


Forward Pass:

解决执行subgraph部分梯度不存在的问题：proactively marking them ready at the end of the forward pass

获知梯度存在信息：
find_unused_parameters： During the backward pass, the Reducer would only wait for unready parameters, but it would still reduce all buckets.
- when the optimizer uses gradient absence information to skip updating momentum values.
- 坏处是遍历autograd graph有开销
- DDP uses a bitmap to keep track of local param-
eter participants and launches one additional AllReduce to
collect globally unused parameters.



Backward Pass: 
DDP uses autograd hooks registered at construction time to trigger gradients synchronizations
When gradients in one bucket are all ready, the Reducer kicks off an asynchronous allreduce on that bucket to calculate mean of gradients across all processes.
When all buckets are ready, the Reducer will block waiting for all allreduce operations to finish.
When this is done, averaged gradients are written to the param.grad field of all parameters.

Optimizer Step:
From the optimizer’s perspective, it is optimizing a local model.
Model replicas on all DDP processes can keep in sync because they all start from the same state
and they have the same averaged gradients in every iteration.
- optimizer开销大吗？比完整参数传输开销小


# Gradient Accumulation


ddp = DistributedDataParallel(net)
with ddp.no_sync ():
    for inp , exp in zip(inputs , expected_outputs):
        # no synchronization , accumulate grads
        loss_fn(ddp(inp), exp).backward()
# synchronize grads
loss_fn(ddp(another_inp), another_exp).backward()
opt.step()


### distributed.py

# 细节
1.超参
process_group: to specify a process group
instance for DDP to run AllReduce, which helps to avoid
messing up with the default process group

bucket_cap_mb: to control the AllReduce bucket size, where applications
should tune this knob to optimize training speed,

find_unused_parameters: to toggle whether DDP should detect unused parameters by traversing the autograd graph.

2.Model Device Aﬃnity
treats the multi-device model as one entirety

3.Model Buﬀers
are necessary when layers need to keep
track of states like the running variance and the running
mean (e.g., BatchNorm).

DDP supports model buﬀers by letting the process with the rank 0 to take the authority.

If the model contains buﬀers, DDP will broadcast the buﬀer values
from rank 0 process to all other processes before starting
the forward pass on the local model. This behavior is also
compatible with the no sync mode. When no sync mode is
enabled, it sets a ﬂag in the forward pass properly to indi-
cate whether it expects gradient reductions in the immediate
backward pass. If the communication takes place, DDP will
then broadcast buﬀers prior to the subsequent forward pass

# 代码

def forward


intra-process parameter synchronization when one DDP process works on multiple devices,
and it also broadcasts model buffers from the process with rank 0 to all other processes
def _sync_param


the inter-process parameter synchronization happens in Reducer.cpp.



### reducer.h, comm.h

# comm.h

implements the coalesced broadcast helper function which is invoked to broadcast model states during initialization
and synchronize model buffers before the forward pass.


# reducer.h

Reducer: The constructor is called in distributed.py which registers Reducer::autograd_hook() to gradient accumulators.

- Parameter-to-Bucket Mapping:
-- make sure that all parameters in the same bucket are on the same device
-- launches AllReduce in the reverse order of model.parameters()


- autograd_hook()
-- function will be invoked by the autograd engine when a gradient becomes ready.
-- each bucket keeps a count of pending gradients. Each post-hook function decrements the count, and
DDP marks a bucket as ready when that count reaches zero.
-- In the next forward pass, DDP replenishes the pending gradient count for every bucket.

- Bucket AllReduce
-- By default, each bucket is 25MB in size.

- Globally Unused Parameters
-- prepare_for_backward() is called at the end of DDP forward pass in distributed.py.
   It traverses the autograd graph to find unused parameters when find_unused_parameters is set to True in DDP constructor.
-- DDP maintains local unused parameter information in a bitmap, and launches an additional AllReduce to gather a global bitmap.
-- all parameters in the model share the same bitmap，没有per-bucket的设计
-- 存在CPU上
-- DDP maintains another bitmap on the same device as the ﬁrst model parameter,
  and invokes a non-blocking copy to move the CPU bitmap to the device bitmap for collective communications.



### ProcessGroup.hpp (NCCL, GLOO, MPI, RR)

All ProcessGroup instances construct at the same time by
using a rendezvous service, where the ﬁrst arrival will block
waiting until the last instance joins

For NCCL backend, the
ProcessGroup maintains a dedicated set of CUDA streams
for communication, so that communications will not block
the computation in the default stream

DistributedDataParallel uses ProcessGroup::broadcast()
to send model states from the process with rank 0 to others during initialization and ProcessGroup::allreduce() to sum gradients.

# round-robin ProcessGroups

PyTorch v1.5 provides a composite round-
robin ProcessGroup implementation, which takes a list of
ProcessGroup instances and dispatches collective communi-
cations to those ProcessGroup instances in a round-robin
manner. By using round-robin ProcessGroups, DDP can at-
tain higher bandwidth utilization if a single NCCL, Gloo, or
MPI ProcessGroup is unable to saturate the link capacity.


# Store.hpp
assists the rendezvous service for process group instances to find each other.

### run.py

逻辑：elastic_launch

命令行启动：利用python的console script

entry_points = {
    "console_scripts": [
        "torchrun = torch.distributed.run:main",
    ],
    "torchrun.logs_specs": [
        "default = torch.distributed.elastic.multiprocessing:DefaultLogsSpecs",
    ],
}

### multiprocessing

pytorch 在 multiprocessing 又加了一个 wraper 以实现shared memory