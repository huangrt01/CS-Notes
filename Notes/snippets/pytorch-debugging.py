### Intro 

浅谈大模型训练排障平台的建设
https://cloud.tencent.com/developer/article/2359942


TORCH_SHOW_CPP_STACKTRACES=1
TORCH_DISTRIBUTED_DEBUG=DETAIL
TORCH_CPP_LOG_LEVEL=INFO
NCCL_DEBUG=INFO
PYTHONFAULTHANDLER=1
CUDA_LAUNCH_BLOCKING=1


### op

TORCH_SHOW_CPP_STACKTRACES=1


### DDPModel

* NCCL_DEBUG=INFO
* 断点打在上面会卡住，打到后面

torch.distributed.breakpoint(rank)


* TORCH_DISTRIBUTED_DEBUG = OFF (default), INFO, or DETAIL

# 非常有用！

For fine-grained control of the debug level during runtime the functions torch.distributed.set_debug_level(),
torch.distributed.set_debug_level_from_env(), and torch.distributed.get_debug_level() can also be used.

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank):
    dist.init_process_group("nccl", rank=rank, world_size=2)
    torch.cuda.set_device(rank)
    tensor = torch.randn(10 if rank == 0 else 20).cuda()
    dist.all_reduce(tensor)
    torch.cuda.synchronize(device=rank)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    mp.spawn(worker, nprocs=2, args=())


# debug rank hang: dist.monitored_barrier

import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank):
    dist.init_process_group("nccl", rank=rank, world_size=2)
    # monitored barrier requires gloo process group to perform host-side sync.
    group_gloo = dist.new_group(backend="gloo")
    if rank not in [1]:
        dist.monitored_barrier(group=group_gloo, timeout=timedelta(seconds=2))


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    mp.spawn(worker, nprocs=2, args=())