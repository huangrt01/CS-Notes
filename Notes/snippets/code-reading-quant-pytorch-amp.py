https://zhuanlan.zhihu.com/p/348554267

* scaler策略：

因为 loss 和梯度的数值在变，scale factor 需要跟随 loss 动态变化。
健康的 loss 是振荡中下降，因此GradScaler设计的 scale factor 每隔N个 iteration 乘一个大于 1 的系数，再 scale loss；
并且每次更新前检查溢出问题（检查梯度中有没有inf和nan），如果有，scale factor 乘一个小于 1 的系数并跳过该 iteration 的参数更新环节，
如果没有，就正常更新参数。动态更新 scale factor 是 amp 实际操作中的流程。

总结 amp 动态 scale factor 的训练流程：
1.维护一个 FP32 数值精度模型的副本
2.初始化s
3.在每个 iteration：
- 拷贝并且转换成FP16模型
- 前向传播（FP16 的模型参数）
- loss 乘 scale factor s
- 反向传播（FP16 的模型参数和参数梯度） 
- 检查有没有inf或者nan的参数梯度
- 如果有：降低 s，回到第二小步
- 参数梯度乘 1/s  这一步的时候，就需要把fp16的梯度变成fp32了，否则乘以 1/S 有可能溢出
- 更新 FP32 的模型参数

一个典型的scaler state dict：{
  "scale": 4096.0,
  "growth_factor": 2.0,
  "backoff_factor": 0.5,
  "growth_interval": 2000,
  "_growth_tracker": 2
} 

* torch/amp/autocast_mode.py

def autocast_decorator(autocast_instance, func):
    @functools.wraps(func)
    def decorate_autocast(*args, **kwargs):
        with autocast_instance:
            return func(*args, **kwargs)

    decorate_autocast.__script_unsupported = "@autocast() decorator is not supported in script mode"  # type: ignore[attr-defined]
    return decorate_autocast



class autocast:

  def __enter__(self):
      self.prev = torch.is_autocast_enabled()
      torch.set_autocast_enabled(self._enabled)
      torch.autocast_increment_nesting()

  def __exit__(self, *args):
      ...
      if torch.autocast_decrement_nesting() == 0:
          torch.clear_autocast_cache()
      torch.set_autocast_enabled(self.prev)
      return False

  def __call__(self, func):
        if torch._jit_internal.is_scripting():
            return func
        return autocast_decorator(self, func)


其中torch.*autocast*函数是在 pytorch/aten/src/ATen/autocast_mode.cpp 里实现。
autocast_mode.cpp 实现策略是 cache fp16 casts of fp32 model weights

* aten/src/ATen/autocast_mode.h
- 关于支持的op

// Op lists for different policies.
// To make sure other backends can reuse the policy op list.
#define AT_FORALL_LOWER_PRECISION_FP(_)  \
  _(_convolution, deprecated)            \
  _(_convolution)                        \
  _(conv1d)                              \
  _(conv2d)                              \
  _(conv3d)                              \
  _(conv_tbc)                            \
  _(conv_transpose1d)                    \
  _(conv_transpose2d, input)             \
  _(conv_transpose3d, input)             \
  _(convolution)                         \
  _(prelu)                               \
  _(addmm)                               \
  _(addmv)                               \
  _(addr)                                \
  _(matmul)                              \
  _(einsum)                              \
  _(mm)                                  \
  _(mv)                                  \
  _(linalg_vecdot)                       \
  _(linear)                              \
  _(addbmm)                              \
  _(baddbmm)                             \
  _(bmm)                                 \
  _(chain_matmul)                        \
  _(linalg_multi_dot)                    \
  _(_thnn_fused_lstm_cell)               \
  _(_thnn_fused_gru_cell)                \
  _(lstm_cell)                           \
  _(gru_cell)                            \
  _(rnn_tanh_cell)                       \
  _(rnn_relu_cell)                       \
  _(_scaled_dot_product_flash_attention) \
  _(scaled_dot_product_attention)

- 核心函数 cached_cast：cache fp16 casts，存上fp32 tensor的weakref

Tensor cached_cast(at::ScalarType to_type, const Tensor& arg, DeviceType device_type) {
  if (is_eligible(arg, device_type) && (arg.scalar_type() != to_type)) {
    // Heuristic:  Do what Apex does, and cache lower_precision_fp casts of fp32 model weights (leaves).
    // See cached_casts declaration above for detailed strategy.
    bool can_try_cache = (to_type == get_lower_precision_fp_from_device_type(device_type) &&
                         arg.scalar_type() == at::kFloat && arg.requires_grad() &&
                         arg.is_leaf() && !arg.is_view() && cache_enabled &&
                         !at::caching::is_cached_tensor(arg));

    if (can_try_cache) {
      const std::lock_guard<std::mutex> lock(cached_casts_mutex);
      auto it = get_cached_casts().find(arg.unsafeGetTensorImpl());
      if (it != get_cached_casts().end()) {
        return std::get<1>(it->second);
      } else {
        auto casted_arg = arg.to(to_type);
        get_cached_casts().emplace(arg.unsafeGetTensorImpl(), val_type{weakref_type(arg.getIntrusivePtr()), casted_arg});
        return casted_arg;
      }
    } else {
      return arg.to(to_type);
    }
  } else {
    return arg;
  }
}

- tip: 推理时没有cache，考虑预转换

model = model.half()
input = input.half()
with torch.no_grad(), torch.cuda.amp.autocast():
    output = model(input)

* grad_scaler.py

torch.cuda.amp.GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True)

def scale(
        self,
        outputs: Union[torch.Tensor, Iterable[torch.Tensor]],
    ) -> Union[torch.Tensor, Iterable[torch.Tensor]]:
        if not self._enabled:
            return outputs

        # Short-circuit for the common case.
        if isinstance(outputs, torch.Tensor):
            if self._scale is None:
                self._lazy_init_scale_growth_tracker(outputs.device)
            assert self._scale is not None
            return outputs * self._scale.to(device=outputs.device, non_blocking=True)

        # Invoke the more complex machinery only if we're treating multiple outputs.
        stash: list[
            _MultiDeviceReplicator
        ] = []  # holds a reference that can be overwritten by apply_scale

        def apply_scale(val: Union[torch.Tensor, Iterable[torch.Tensor]]):
            if isinstance(val, torch.Tensor):
                if len(stash) == 0:
                    stash.append(_MultiDeviceReplicator(self._scale))
                return val * stash[0].get(val.device)
            if isinstance(val, abc.Iterable):
                iterable = map(apply_scale, val)
                if isinstance(val, (list, tuple)):
                    return type(val)(iterable)
                return iterable
            raise ValueError("outputs must be a Tensor or an iterable of Tensors")

        return apply_scale(outputs)

def unscale_
  optimizer_state["found_inf_per_device"] = self._unscale_grads_(
      optimizer, inv_scale, found_inf, False
  )
  optimizer_state["stage"] = OptState.UNSCALED


def step(optimizer, *args, **kwargs):
  optimizer_state = self._per_optimizer_states[id(optimizer)]
  ...
  optimizer_state["found_inf_per_device"]

1.对梯度 unscale，如果之前没有手动调用unscale方法的话
2.检查梯度溢出，如果没有nan/inf，就执行 optimizer 的 step，如果有就跳过
3.注意：GradScaler的step不支持传 closure。

def update(new_scale=None):
  
update方法在每个 iteration 结束前都需要调用，
如果参数更新跳过，会给 scale factor 乘backoff_factor，
或者到了该增长的 iteration，就给 scale factor 乘growth_factor。
也可以用new_scale直接更新 scale factor。


