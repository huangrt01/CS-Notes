### 写Op

TODO： https://pytorch.org/tutorials/advanced/cpp_extension.html

* 关于非contiguous

如果我们只是用[]/(), 索引，他们都是操作符重载，内部考虑了shape, stride, order, offset等，不会出错。在很多情况下可以节省大量内存
但是我们拿指针出来操作数据的所有情况，都要保证是contiguous的， 否则可能出错。


### argsort

sort_indices = torch.argsort(shard_ids, stable=True)

### concat split
split的输出是tuple，经常需要转为list


### jagged/padding


def generate_row_splits_from_row_lengths(
    row_lengths: torch.Tensor) -> torch.Tensor:
  row_splits = torch.cat([
      torch.tensor([0], dtype=torch.int32, device=row_lengths.device),
      torch.cumsum(row_lengths, dim=0, dtype=torch.int32)
  ])
  return row_splits


q = torch.ops.fbgemm.jagged_to_padded_dense(values=q_varlen,
                                          offsets=[row_splits],
                                          max_lengths=[max_length],
                                          padding_value=0)

### rearrange

from einops import rearrange

q = rearrange(q, 'b t (h d) -> b t h d', h=nheads)

### Example


class All2AllSingle(torch.autograd.Function):
  @staticmethod
  def forward(ctx: Any, tensor: torch.Tensor,
              output_split_sizes: list[int],
              input_split_sizes: list[int]) -> Any:
    ctx.output_split_sizes = output_split_sizes
    ctx.input_split_sizes = input_split_sizes

    output_shape = (sum(output_split_sizes), tensor.size(1))
    output_tensor = torch.empty(
      output_shape, dtype=tensor.dtype, device=tensor.device)
    dist.all_to_all_single(output_tensor, tensor,
                           output_split_sizes=output_split_sizes,
                           input_split_sizes=input_split_sizes)
    return output_tensor

  @staticmethod
  def backward(ctx: Any, *grad_outputs: Any) -> Any:
    assert len(grad_outputs) == 1
    tensor_grad = grad_outputs[0]
    output_split_sizes = ctx.output_split_sizes
    input_split_sizes = ctx.input_split_sizes
    output_shape = (sum(input_split_sizes), tensor_grad.size(1))
    output_tensor_grad = torch.empty(
      output_shape, dtype=tensor_grad.dtype, device=tensor_grad.device)

    assert tensor_grad.is_contiguous(), f"tensor_grad not contiguous, {tensor_grad.shape}, {tensor_grad}"
    dist.all_to_all_single(output_tensor_grad, tensor_grad,
                           output_split_sizes=input_split_sizes,
                           input_split_sizes=output_split_sizes)
    return output_tensor_grad, None, None