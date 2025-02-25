### toolkit

# 根据nvidia-smi的驱动版本

https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

# CUDA版本

% nvcc -V 

# note
nvcc是准确的。  nvidia-smi是最大支持版本
一个CUDA Driver版本可以支持多个CUDA版本



### basic

CUDA_VISIBLE_DEVICES='0,1,2'

import torch

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print("CUDA设备数量:", device_count)

    for i in range(device_count):
        print("CUDA设备{}型号:".format(i), torch.cuda.get_device_name(i))
else:
    print("CUDA不可用")

print(torch.cuda.current_device())


### utils

import GPUtil
GPUtil.showUtilization()