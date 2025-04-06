* aten/src/ATen/autocast_mode.h

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