https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html


### torch.compile generate Triton Kernel

# compile_square.py

import torch
compiled_square = torch.compile(torch.square)
x = torch.randn(10, 10).cuda()
result = compiled_square(x)

TORCH_LOGS="output_code" python compile_square.py


@torch.compile
def square_2(a):
  a = torch.square(a)
  a = torch.square(a)
  return a