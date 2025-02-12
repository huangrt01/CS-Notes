https://zhuanlan.zhihu.com/p/346205754

### usage

import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore') #ignore warnings

x = torch.linspace(-np.pi, np.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
for t in range(1, 1001):
    y_pred = model(xx)
    loss = loss_fn(y_pred, y)
    if t % 100 == 0:
        print('No.{: 5d}, loss: {:.6f}'.format(t, loss.item()))
    optimizer.zero_grad() # 梯度清零
    loss.backward() # 反向传播计算梯度
    optimizer.step() # 梯度下降法更新参数

# param_groups

optimizer = SGD([
                {'params': model.base.parameters()}, 
                {'params': model.fc.parameters(), 'lr': 1e-3} # 对 fc的参数设置不同的学习率
            ], lr=1e-2, momentum=0.9)

### Optimizer
torch.optim.optimizer
Optimizer._optimizer_step_post_hooks

self.state: DefaultDict[torch.Tensor, Any] = defaultdict(dict)
self.param_groups: List[Dict[str, Any]] = []

def __getstate__(self) -> Dict[str, Any]:  # noqa: D105
        return {
            "defaults": self.defaults,
            "state": self.state,
            "param_groups": self.param_groups,
        }

def add_param_group
 # This can be useful when fine tuning a pre-trained network as frozen layers can be made
   #      trainable and added to the :class:`Optimizer` as training progresses.
   # 检查 group之间的param_group["params"] 不冲突

# 注册hook
所有注册：register_optimizer_step_post_hook
单个注册：register_step_pre_hook

register_state_dict_post_hook
register_load_state_dict_post_hook

参数prepend

# 状态

state_dict()

``state`` is a Dictionary mapping parameter ids
            to a Dict with state corresponding to each parameter.

{
    'state': {
        0: {'momentum_buffer': tensor(...), ...},
        1: {'momentum_buffer': tensor(...), ...},
        2: {'momentum_buffer': tensor(...), ...},
        3: {'momentum_buffer': tensor(...), ...}
    },
    'param_groups': [
        {
            'lr': 0.01,
            'weight_decay': 0,
            ...
            'params': [0]
        },
        {
            'lr': 0.001,
            'weight_decay': 0.5,
            ...
            'params': [1, 2, 3]
        }
    ]
}

# zero_grad
set_to_none

set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).

for_each优化
torch._foreach_zero_(grads)


### lr_scheduler


### 随机参数平均 swa_utils


### 技巧
from typing_extensions import ParamSpec, Self, TypeAlias

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]
_P = ParamSpec("_P")
R = TypeVar("R")
T = TypeVar("T")

Args: TypeAlias = Tuple[Any,...]
Kwargs: TypeAlias = Dict[str, Any]
StateDict: TypeAlias = Dict[str, Any]

OrderedDict()

GlobalOptimizerPreHook: TypeAlias = Callable[
    ["Optimizer", Args, Kwargs], Optional[Tuple[Args, Kwargs]]
]

