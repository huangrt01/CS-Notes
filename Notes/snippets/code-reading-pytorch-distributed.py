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