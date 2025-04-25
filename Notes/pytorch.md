# PyTorch

## 资料

* 官网Intro https://pytorch.org/tutorials/
* 国产面试精华。。也适合速成 https://www.mstx.cn/pytorch.html



* PyTorch Internals
  * https://blog.ezyang.com/2019/05/pytorch-internals/



## API

> [PyTorch官方的API接口文档](https://pytorch.org/docs/stable/index.html)

* torch模块。这是针对tensor进行全局设置的模块，主要常用的函数有
  * 全局设置tensor的类型
  * 全局设置tensor的device
  * 全局设置打印tensor的精度等。
  * 生成tensor的各种函数，包括：
    * 随机生成符合正态分布的tensor
    * 随机生成指定大小的tensor等
    * 序列化tensor的save和load函数

* torch.nn模块
  * 卷积层，池化层，dropout层，归一化层，全连接层，rnn层等
  * 各种loss函数。

* torch.autograd模块
  * [PyTorch自动微分](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)

* torch.nn.init模块
  * 网络参数初始化

## 二次开发

* 参考 GPU-Mode Lecture 1
  * load_inline
  * torch.profile
* operator call
  * ![image-20250320022630969](./pytorch/image-20250320022630969.png)
  

## Tensor

* stride
  * ![20250319-223852](pytorch/20250319-223852.jpeg)
  * ![20250319-223859](./pytorch/20250319-223859.jpeg)
  * ![20250319-223904](./pytorch/20250319-223904.jpeg)
    * 这样的设计不同于numpy
  
* Dispatch
  * The first dispatch is based on the device type and layout of a tensor: e.g., whether or not it is a CPU tensor or a CUDA tensor (and also, e.g., whether or not it is a strided tensor or a sparse one).
  * ![image-20250319225317174](./pytorch/image-20250319225317174.png)
  * ![image-20250319225337097](./pytorch/image-20250319225337097.png)
  * 一种拓展思路：tensor的wrapper class，参考torchao

* Gather index
  * torch.gather
  * 数组深加工


![image-20241217021330875](./pytorch/image-20241217021330875.png)

* leaf node
  * 默认情况下，非叶节点的梯度值在反向传播过程中使用完后就会被清除，不会被保留，只有叶子节点的梯度值能够被保留下来。对于非叶子节点而言，PyTorch出于节省内存的考虑，通常不会保存节点的导数值。总之，一句话：在调用backward()时，只有当节点的requires_grad和is_leaf同时为真时，才会计算节点的梯度值，也就是说节点的grad属性才会赋值，否则为None
  * 所有用户创建的向量都是叶子结点
    * 显式、隐式
  * 非叶子节点：中间变量



## Graph

* 动态图
  * 几乎每进行一次运算都会拓展原先的计算图，最后生成完成。
  * 当反向传播完成，计算图默认会被清除，所以只能用生成的计算图进行一次反向传播
  * `retain_graph` 参数可以保持计算图，从而避免别清除掉，其用法为：`loss.backward(retain_graph=True)`

### 异步执行

* CPU：同步执行
* GPU Op：
  * 任务分配
    * GPU执行算子计算
    * CPU推导输出信息、创建输出tensor、定位算子
  * 核心：CPU给GPU提交任务，不等待，直接返回

### 利用Stream

![image-20250424023703685](./pytorch/image-20250424023703685.png)



## Autograd

* Grad:  Jacobians left-multiplied by a vector,

![image-20250319225735790](pytorch/image-20250319225735790.png)

* ![image-20250319231025721](./pytorch/image-20250319231025721.png)
  * **在多输出操作的反向传播中，前向传播的输出会作为反向传播对应模块的输入**。通过 `output_nr`（前向输出编号）和 `input_nr`（反向输入编号）的映射，PyTorch 的自动求导系统能准确传递梯度，保证复杂计算图（如特征值分解 + 矩阵乘法）的反向传播正确执行。
* ![image-20250320020726734](./pytorch/image-20250320020726734.png)

## Module

* 源码解读：https://zhuanlan.zhihu.com/p/340453841
  * ![image-20250305012942113](./pytorch/image-20250305012942113.png)
  * ![v2-2233d8b647f1b56e2ad9ef92d90e1706_1440w](./pytorch/v2-2233d8b647f1b56e2ad9ef92d90e1706_1440w.jpg)



## Optimizer

### Optimizing Optimizers —— [GPU Mode Lecture 6](https://www.youtube.com/watch?v=hIop0mWKPHc)

> 优化runtime

* kernel fusion —— vertical + horizontal
  * for loop
  * for each
  * entirely fused

![image-20250407020056387](./pytorch/image-20250407020056387.png)

## Serving

* 如果追求性能，可以用torch.fx改一下图，把手写op改进去
* torch.fx symbolic_trace可以变成静态图
* 静态图，会用torchscript trace出来整个图，然后在ir上做一些编译优化



## 调参

* 冻结网络参数的方式有三种：

  - 利用优化器的param_groups属性

  - 利用detach分离出某个网络层
    - detach
      - 将张量从当前计算图中分离出来，从而不需要跟踪张量的梯度变化
      - 将张量从GPU移动到CPU时

  - 利用dropout操作。

  * 前2种方法都是冻结整个层的训练参数，第3种方法使用dropout可以冻结部分网络参数，但是这是随机的



## Pytorch Lightning

![image-20241217014851707](./pytorch/image-20241217014851707.png)