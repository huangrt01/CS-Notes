### DDP

# 细节：DDP仅支持所有worker同一设备类型，不支持部分worker GPU、部分worker CPU


def setup(rank: int, world_size: int):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() and world_size < torch.cuda.device_count() else "cpu")
  backend = "nccl" if device.type == "cuda" else "gloo"
  dist.init_process_group(backend, rank=rank, world_size=world_size)

def _distributed_task(rank, world_size, ckpt_dir, task_name, result_dir):
  setup(rank, world_size)
  ...
  device = None
  device_ids = None
  if torch.cuda.is_available() and world_size <= torch.cuda.device_count():
    device = torch.device(f"cuda:{rank}")
    device_ids = [rank]
  else:
    print(f"invalid cuda device count: {torch.cuda.device_count()}, use cpu instead")
    device = torch.device("cpu")

  model = model.to(device)
  ddp_model = DDPModel(model, device_ids=device_ids)
  ...
  dist.barrier()
  all_loaders = [None] * world_size
  dist.all_gather_object(all_loaders, loader.state_dict())
  dist.barrier()
  if rank == 0:
    ...
  dist.destroy_process_group()


### TorchRun

https://pytorch.org/docs/stable/elastic/run.html

torchrun provides a superset of the functionality as torch.distributed.launch with the following additional functionalities:

- Worker failures are handled gracefully by restarting all workers.

- Worker RANK and WORLD_SIZE are assigned automatically.

- Number of nodes is allowed to change between minimum and maximum sizes (elasticity).

# single node multi worker

torchrun
    --standalone
    --nnodes=1
    --nproc-per-node=$NUM_TRAINERS
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

# Stacked single-node multi-worker

torchrun
    --rdzv-backend=c10d
    --rdzv-endpoint=localhost:0
    --nnodes=1
    --nproc-per-node=$NUM_TRAINERS
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

torchrun
    --nnodes=$NUM_NODES
    --nproc-per-node=$NUM_TRAINERS
    --max-restarts=3
    --rdzv-id=$JOB_ID
    --rdzv-backend=c10d
    --rdzv-endpoint=$HOST_NODE_ADDR
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

import torch.distributed as dist
dist.init_process_group(backend="gloo|nccl")