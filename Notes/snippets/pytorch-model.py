### Linear

torch.nn.Linear(in_features, out_features, bias=True)

# y = Ax + b, A的形状是 (out_features, in_features)

# LazyLinear
torch.nn.LazyLinear(out_features, bias=True, device=None, dtype=None)


### Bilinear

bilinear_layer = nn.Bilinear(in1_features=5, in2_features=3, out_features=2)

从数学上看，它对两个输入向量的每个元素组合进行加权求和，权重由决定，这种操作能够捕捉两个输入之间的二阶交互信息。
用于RPN生成候选区域、推荐系统协同过滤等

### module list
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        for index, liner in enumerate(self.linears):
            x = self.linears[index // 2](x) + liner(x)
        return x

append(module)
extend(modules)
insert(index, module)

### Sequential

model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )

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

### 提取中间特征
# 自己的模型
class Feature_extractor(nn.module):
   def forward(self, input):
      self.feature = input.clone()
      return input
# pretrain模型
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchvision.models.feature_extraction import create_feature_extractor

transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 
                                0.456, 0.406], [0.229, 0.224, 0.225])])
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
feature_extractor = create_feature_extractor(model, return_nodes={"conv1": "output"})
original_img = Image.open("dog.jpg")
img = transform(original_img).unsqueeze(0)
out = feature_extractor(img)
plt.imshow(out["output"][0].transpose(0, 1).sum(1).detach().numpy())
plt.show()


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

### Loss

DEFAULT_EPS = 1e-10
PADDED_Y_VALUE = -1


def listMLE(y_pred,
            y_true,
            eps=DEFAULT_EPS,
            padded_value_indicator=PADDED_Y_VALUE,
            position_aware=False):
  """
    https://github.com/allegro/allRank/blob/master/allrank/models/losses/listMLE.py

    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
  # shuffle for randomised tie resolution
  random_indices = torch.randperm(y_pred.shape[-1])
  y_pred_shuffled = y_pred[:, random_indices]
  y_true_shuffled = y_true[:, random_indices]

  y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

  mask = y_true_sorted == padded_value_indicator

  preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
  preds_sorted_by_true[mask] = float("-inf")

  max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

  preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

  preds_sorted_by_true_minus_max_exp = preds_sorted_by_true_minus_max.exp(
  ).flip(dims=[1])
  cumsums = torch.cumsum(preds_sorted_by_true_minus_max_exp,
                         dim=1).flip(dims=[1])

  observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

  # logging.info(
  #     f"debug random_indices: {random_indices}, y_pred: {y_pred}, y_true: {y_true}, y_pred_shuffled: {y_pred_shuffled}, y_true_shuffled: {y_true_shuffled}, y_true_sorted: {y_true_sorted}, indices: {indices}, mask: {mask}, preds_sorted_by_true: {preds_sorted_by_true}, max_pred_values: {max_pred_values}, preds_sorted_by_true_minus_max: {preds_sorted_by_true_minus_max}, cumsums: {cumsums}, observation_loss: {observation_loss}"
  # )
  if position_aware:
    # Calculate the rank of each sample in the sorted y_true
    ranks = torch.arange(1,
                         y_true_sorted.shape[1] + 1,
                         device=y_true_sorted.device,
                         dtype=torch.float32)
    indices = torch.arange(len(ranks))
    finesort_show_indices = indices <= 12
    finesort_not_show_indices = indices > 12
    ranks[finesort_show_indices] = 2.05 - 0.05 * ranks[finesort_show_indices]
    ranks[finesort_not_show_indices] = 1.0 - 0.0025 * ranks[
        finesort_not_show_indices]

    ranks = ranks.unsqueeze(0).expand_as(y_true_sorted)

    # logging.info(f"debug ranks: {ranks}, observation_loss: {observation_loss}")

    # Calculate the position aware loss
    observation_loss = observation_loss * ranks
  observation_loss[mask] = 0.0
  return torch.mean(torch.sum(observation_loss, dim=1))

