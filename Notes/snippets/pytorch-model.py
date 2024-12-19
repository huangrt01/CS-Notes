### Linear

torch.nn.Linear(in_features, out_features, bias=True)

# y = Ax + b, A的形状是 (out_features, in_features)

# LazyLinear
torch.nn.LazyLinear(out_features, bias=True, device=None, dtype=None)


### Bilinear

bilinear_layer = nn.Bilinear(in1_features=5, in2_features=3, out_features=2)

从数学上看，它对两个输入向量的每个元素组合进行加权求和，权重由决定，这种操作能够捕捉两个输入之间的二阶交互信息。
用于RPN生成候选区域、推荐系统协同过滤等


### dropout

m = nn.Dropout(p=0.2)
input = torch.randn(20, 16)
output = m(input)


### Initializetion

# Xavier Init

Understanding the difficulty of training deep feedforward neural networks》论文中提出一个洞见：
激活值的方差是逐层递减的，这导致反向传播中的梯度也逐层递减。
要解决梯度消失，就要避免激活值方差的衰减，最理想的情况是，每层的输出值（激活值）保持高斯分布。
就这个想法，Xavier Glorot新的初始化方法，后来称之为“Xavier Initialization”

尤其适用于激活函数为tanh或sigmoid的情况


# 使用自定义的线性层

import torch.nn.init as init

class CustomLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinear, self).__init__(in_features, out_features, bias)
        # 使用自定义的权重初始化方法
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
custom_linear = CustomLinear(in_features=10, out_features=2)


### learning rate

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 25], gamma=0.1)
scheduler = ReduceLROnPlateau(optimizer,
                                mode='min',
                                factor=0.6,
                                patience=60,
                                min_lr=0.001)

for epoch in range(1, 31):
    # train
    optimizer.zero_grad()
    optimizer.step()
    print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]["lr"]))
    scheduler.step()


### Techniques

# 普通tensor无法赋值给网络权重

解决方法有两个：
- 使用torch.nn.Parameter包装普通张量(推荐使用)
- 将普通张量赋值给weight.data


# 查看权重

model = MyModel()
for name, param in model.named_parameters():
    print(name)
    print("-" * 24)
    print(param)
    print("=" * 24)

### Optimizer
optimizer_Adam = torch.optim.Adam(model.parameters(), lr=0.1)
print(optimizer_Adam.param_groups())

[
    {
        "params":   [Parameter containing:tensor([[-0.0386,  0.3979],
                            [ 0.2451, -0.5477],
                            [ 0.2848, -0.6663]], requires_grad=True),
                    Parameter containing:tensor([-0.6260,  0.6027,  0.0412], requires_grad=True), 
                    Parameter containing:tensor([[-0.2931, -0.3993,  0.1601],
                            [ 0.1608,  0.1821,  0.4538],
                            [ 0.3516, -0.4239, -0.5256],
                            [ 0.4598,  0.1838, -0.4019],
                            [-0.4469,  0.4455,  0.1316],
                            [-0.1232,  0.3769, -0.1184]], requires_grad=True),
                    Parameter containing:tensor([ 0.1404, -0.0542, -0.0085,  0.0995,  0.3741, -0.0223],requires_grad=True)], 
       "lr": 0.1, 
       "betas": (0.9, 0.999), 
       "eps": 1e-08, 
       "weight_decay": 0, 
       "amsgrad": False, 
       "maximize": False, 
       "foreach": None, 
       "capturable": False, 
       "differentiable": False, 
       "fused": None
    }
]

