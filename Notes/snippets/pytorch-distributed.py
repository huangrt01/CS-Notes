def setup(rank, world_size):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  dist.init_process_group("nccl", rank=rank, world_size=world_size)

def _distributed_task(rank, world_size, ckpt_dir, task_name, result_dir):
  setup(rank, world_size)
  ...
  device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  ddp_model = DDPModel(model, device_ids=[rank])
  ...
  dist.barrier()
  all_loaders = [None] * world_size
  dist.all_gather_object(all_loaders, loader.state_dict())
  dist.barrier()
  if rank == 0:
    ...
  dist.destroy_process_group()