### pin_memory

https://zhuanlan.zhihu.com/p/9616453650

pinned=True+ non_blocking=True + multiple cuda streams ，传输数据最快

https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html

Generally, asynchronous copies to a device are safe without explicit synchronization
only when the target is a CUDA-enabled device and the original tensor is in pageable memory.

In summary, copying data from CPU to GPU is safe when using non_blocking=True,
but for any other direction, non_blocking=True can still be used but the user must make sure that a device synchronization
is executed before the data is accessed. (torch.cuda.synchronize())


### distributed ckpt

https://github.com/pytorch/tutorials/blob/main/recipes_source/distributed_checkpoint_recipe.rst

### async distributed ckpt

https://pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html

https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/state_dict_saver.py

import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

CHECKPOINT_DIR = "checkpoint"


class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"]
        )

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355 "

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def run_fsdp_checkpoint_save_example(rank, world_size):
    print(f"Running basic FSDP checkpoint saving example on rank {rank}.")
    setup(rank, world_size)

    # create a model and move it to GPU with id rank
    model = ToyModel().to(rank)
    model = FSDP(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    checkpoint_future = None
    for step in range(10):
        optimizer.zero_grad()
        model(torch.rand(8, 16, device="cuda")).sum().backward()
        optimizer.step()

        # waits for checkpointing to finish if one exists, avoiding queuing more then one checkpoint request at a time
        if checkpoint_future is not None:
            checkpoint_future.result()

        state_dict = { "app": AppState(model, optimizer) }
        checkpoint_future = dcp.async_save(state_dict, checkpoint_id=f"{CHECKPOINT_DIR}_step{step}")

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running async checkpoint example on {world_size} devices.")
    mp.spawn(
        run_fsdp_checkpoint_save_example,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )