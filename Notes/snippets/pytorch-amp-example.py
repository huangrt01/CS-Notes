import torch, time, gc
import numpy as np
import random, os

# Timing utilities
start_time = None


def start_timer():
  global start_time
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_max_memory_allocated()
  torch.cuda.synchronize()
  start_time = time.time()


def end_timer_and_print(local_msg):
  torch.cuda.synchronize()
  end_time = time.time()
  print("\n" + local_msg)
  print("Total execution time = {:.3f} sec".format(end_time - start_time))
  print("Max memory used by tensors = {} bytes".format(
      torch.cuda.max_memory_allocated()))

def set_seed(seed: int = 37) -> None:
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)  # 适用于所有PyTorch后端，包括CPU和所有CUDA设备

  os.environ["PYTHONHASHSEED"] = str(seed)
  print(f"设置随机数种子为{seed}")

def make_model(in_size, out_size, num_layers):
  layers = []
  for _ in range(num_layers - 1):
    layers.append(torch.nn.Linear(in_size, in_size))
    layers.append(torch.nn.ReLU())
  layers.append(torch.nn.Linear(in_size, out_size))
  return torch.nn.Sequential(*tuple(layers)).cuda()

if __name__ == "__main__":
  set_seed()
  batch_size = 512  # Try, for example, 128, 256, 513.
  in_size = 4096
  out_size = 4096
  num_layers = 3
  num_batches = 50
  epochs = 3

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  torch.set_default_device(device)

  # Creates data in default precision.
  # The same data is used for both default and mixed precision trials below.
  # You don't need to manually change inputs' ``dtype`` when enabling mixed precision.
  data = [torch.randn(batch_size, in_size) for _ in range(num_batches)]
  targets = [torch.randn(batch_size, out_size) for _ in range(num_batches)]

  loss_fn = torch.nn.MSELoss().cuda()

  set_seed() # 必须在这里，原因是前面调用了randn
  # warm up
  net = make_model(in_size, out_size, num_layers)
  opt = torch.optim.SGD(net.parameters(), lr=0.001)
  for epoch in range(epochs):
    for input, target in zip(data, targets):
      output = net(input)
      loss = loss_fn(output, target)
      loss.backward()
      opt.step()
      opt.zero_grad() # set_to_none=True here can modestly improve performance
  fp32_result_1 = net(data[0])

  set_seed()
  net = make_model(in_size, out_size, num_layers)
  opt = torch.optim.SGD(net.parameters(), lr=0.001)
  start_timer()
  for epoch in range(epochs):
    for input, target in zip(data, targets):
      output = net(input)
      loss = loss_fn(output, target)
      loss.backward()
      opt.step()
      opt.zero_grad() # set_to_none=True here can modestly improve performance
  end_timer_and_print("Default precision:")
  fp32_result = net(data[0])

  if torch.allclose(fp32_result, fp32_result_1, atol=5e-2, rtol=0):
    print("✅ Default precision match")
  else:
    print("❌ Default precision differ")
    print(fp32_result, fp32_result_1)

  set_seed()
  net = make_model(in_size, out_size, num_layers)
  opt = torch.optim.SGD(net.parameters(), lr=0.001)

  # Constructs a ``scaler`` once, at the beginning of the convergence run, using default arguments.
  # If your network fails to converge with default ``GradScaler`` arguments, please file an issue.
  # The same ``GradScaler`` instance should be used for the entire convergence run.
  # If you perform multiple convergence runs in the same script, each run should use
  # a dedicated fresh ``GradScaler`` instance. ``GradScaler`` instances are lightweight.
  scaler = torch.amp.GradScaler("cuda")

  start_timer()
  for epoch in range(epochs):
    for input, target in zip(data, targets):
      with torch.autocast(device_type=device, dtype=torch.float16):
        output = net(input)
        # output is float16 because linear layers ``autocast`` to float16.
        assert output.dtype is torch.float16
        loss = loss_fn(output, target)
        # loss is float32 because ``mse_loss`` layers ``autocast`` to float32.
        assert loss.dtype is torch.float32
      # Exits ``autocast`` before backward().
      # Backward passes under ``autocast`` are not recommended.
      # Backward ops run in the same ``dtype`` ``autocast`` chose for corresponding forward ops.
      # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
      scaler.scale(loss).backward()
      scaler.unscale_(opt)
      # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)
      # otherwise, optimizer.step() is skipped.
      scaler.step(opt)
      # Updates the scale for next iteration.
      scaler.update()
      # checkpoint = {"model": net.state_dict(),
      #               "optimizer": opt.state_dict(),
      #               "scaler": scaler.state_dict()}
      opt.zero_grad() # set_to_none=True here can modestly improve performance
  end_timer_and_print("Mixed precision:")
  fp16_result = net(data[0]).to(torch.float32)

  if torch.allclose(fp32_result, fp16_result, atol=5e-2, rtol=0):
    print("✅ Default and Mixed precision match")
  else:
    print("❌ Default and Mixed precision differ")
    print(fp32_result, fp16_result)