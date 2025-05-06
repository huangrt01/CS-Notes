# 关于驱动

dpkg -l | grep nvidia
cat /proc/driver/nvidia/version

根据nvidia-smi的驱动版本

https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

# CUDA版本

% nvcc -V 

# note
nvcc是准确的。  nvidia-smi是最大支持版本
一个CUDA Driver版本可以支持多个CUDA版本



### basic

>>> a = torch.cuda.get_device_properties(0)
>>> a.
a.L2_cache_size                    a.max_threads_per_multi_processor  a.total_memory
a.gcnArchName                      a.minor                            a.uuid
a.is_integrated                    a.multi_processor_count            a.warp_size
a.is_multi_gpu_board               a.name
a.major                            a.regs_per_multiprocessor

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

### nvcc

nvcc -o output xx.cu


### utils

import GPUtil
GPUtil.showUtilization()