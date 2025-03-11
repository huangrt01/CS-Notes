### flash attn varlen

https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py#L1375

### flash attn v1/v2

@triton.jit
def flash_attention_v1_kernel(
    # 输入指针
    q_ptr,      # Query 矩阵指针
    k_ptr,      # Key 矩阵指针
    v_ptr,      # Value 矩阵指针
    o_ptr,      # 输出矩阵指针

    # Q 矩阵的步长
    q_batch_stride,    # batch 维度的步长
    q_heads_stride,    # heads 维度的步长
    q_seq_stride,      # 序列维度的步长
    q_dim_stride,      # head_dim 维度的步长

    # K 矩阵的步长
    k_batch_stride,
    k_heads_stride,
    k_seq_stride,
    k_dim_stride,      # matrix K stride for columns, [seq_len, head_dim]

    # V 矩阵的步长
    v_batch_stride,
    v_heads_stride,
    v_seq_stride,
    v_dim_stride,

    # 输出矩阵的步长
    out_batch_stride,
    out_heads_stride,
    out_seq_stride,
    out_dim_stride,

    # 其他参数
    n_heads,           # 注意力头数量
    m_size,           # Q 矩阵的序列长度
    n_size,           # K/V 矩阵的序列长度
    BLOCK_DHEAD_SIZE: tl.constexpr,  # head_dim 维度的块大小
    BLOCK_M_SIZE: tl.constexpr,      # Q 序列维度的块大小
    BLOCK_N_SIZE: tl.constexpr,      # K/V 序列维度的块大小
    sm_scale,         # 注意力分数的缩放因子 1/sqrt(head_dim)
):
    """Flash Attention V1 的 CUDA kernel 实现
    
    参数:
        q_ptr, k_ptr, v_ptr: 输入矩阵的指针
        o_ptr: 输出矩阵的指针
        *_stride: 各个维度的步长
        n_heads: 注意力头数量
        m_size: Q 矩阵的序列长度
        n_size: K/V 矩阵的序列长度
        BLOCK_*_SIZE: 各个维度的块大小
        sm_scale: 注意力分数的缩放因子
    """
    
    # 获取当前程序块的索引
    block_m_idx = tl.program_id(0)  # 序列维度的块索引
    head_idx = tl.program_id(1)     # batch * heads 维度的索引

    # 计算当前处理的 batch 和 head 索引
    cur_batch_idx = head_idx // n_heads
    cur_head_idx = head_idx % n_heads

    # 生成各个维度的偏移量
    m_range_offs = tl.arange(0, BLOCK_M_SIZE)      # Q 矩阵行偏移
    n_range_offs = tl.arange(0, BLOCK_N_SIZE)      # K 矩阵行偏移
    dhead_range_offs = tl.arange(0, BLOCK_DHEAD_SIZE)  # head_dim 维度偏移

    # 计算 Q 矩阵当前块的实际行索引
    m_offs = block_m_idx * BLOCK_M_SIZE + m_range_offs

    # 计算各个矩阵在内存中的偏移地址
    ## m_offs[:, None] 这一操作利用了索引技巧来增加 m_offs 张量的维度。
    q_offs = (
        cur_batch_idx * q_batch_stride +
        cur_head_idx * q_heads_stride +
        (m_offs[:, None] * q_seq_stride + dhead_range_offs[None, :] * q_dim_stride)
    )

    k_offs = (
        cur_batch_idx * k_batch_stride +
        cur_head_idx * k_heads_stride +
        (n_range_offs[:, None] * k_seq_stride + dhead_range_offs[None, :] * k_dim_stride)
    )
    
    v_offs = (
        cur_batch_idx * v_batch_stride +
        cur_head_idx * v_heads_stride +
        (n_range_offs[:, None] * v_seq_stride + dhead_range_offs[None, :] * v_dim_stride)
    )

    o_offs = (
        cur_batch_idx * out_batch_stride +
        cur_head_idx * out_heads_stride +
        (m_offs[:, None] * out_seq_stride + dhead_range_offs[None, :] * out_dim_stride)
    )
    
    # 计算实际的内存地址
    q_ptrs = q_ptr + q_offs
    k_ptrs = k_ptr + k_offs
    v_ptrs = v_ptr + v_offs
    out_ptrs = o_ptr + o_offs

    # 初始化 online softmax 所需的变量
    m_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32) - float("inf")  # 最大值
    d_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32)                 # 分母
    o_i = tl.zeros((BLOCK_M_SIZE, BLOCK_DHEAD_SIZE), dtype=tl.float32)  # 累积输出

    # 加载 Q 矩阵数据
    ## 实际的序列长度 m_size 可能不是 BLOCK_M_SIZE 的整数倍
    q_mask = m_offs[:, None] < m_size
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # 分块处理 K、V 矩阵
    for block_n_start_idx in range(0, n_size, BLOCK_N_SIZE):
        block_n_offs = block_n_start_idx + n_range_offs
        k_mask = block_n_offs[:, None] < n_size
        k = tl.load(k_ptrs + block_n_start_idx * k_seq_stride, mask=k_mask, other=0.0)

        # 计算注意力分数 QK^T
        qk = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k))
        qk *= sm_scale  # 缩放注意力分数

        # 计算当前块的 softmax 统计量
        m_j = tl.max(qk, 1)                        # 当前块的最大值
        n_j = tl.exp(qk - m_j[:, None])           # 计算 exp(qk - max)
        d_j = tl.sum(n_j, 1)                      # 当前块的 softmax 分母

        # 更新 softmax 统计量
        m_new = tl.maximum(m_j, m_i)              # 更新全局最大值
        alpha = tl.exp(m_i - m_new)               # 旧数据的缩放因子
        beta = tl.exp(m_j - m_new)                # 新数据的缩放因子
        d_new = alpha * d_i + beta * d_j          # 更新分母

        # 重新缩放累积的输出
        scale1 = d_i / d_new * alpha 
        o_i = o_i * scale1[:, None]
        
        # 计算当前块的输出贡献
        p_scale = beta / d_new
        qk_softmax = n_j * p_scale[:, None]
        v_ptr_mask = block_n_offs[:, None] < n_size
        V = tl.load(v_ptrs + block_n_start_idx * v_seq_stride, mask=v_ptr_mask, other=0.0)
        o_i += tl.dot(qk_softmax, V)

        # 更新统计量
        m_i = m_new
        d_i = d_new

    # 存储最终结果
    out_mask = m_offs[:, None] < m_size
    tl.store(out_ptrs, o_i, mask=out_mask)


@torch.no_grad()
@custom_fwd(cast_inputs=torch.float16)
def flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale,
    attention_mask: Optional[torch.Tensor] = None,
):
    """Compute Flash-attention, can't support fp32 input
    
    参数:
        q: Query tensor, shape: [bs, n_heads, m_size, head_dim]
           在 decode 阶段, q 的 seq_len 和 k/v 不一致, 其值为 1
        k: Key tensor, shape: [bs, n_heads, n_size, head_dim]
        v: Value tensor, shape 与 k 相同
        sm_scale: 注意力分数的缩放因子
        attention_mask: 注意力掩码矩阵，可广播至 (batch, head_size, m_size, n_size)
        
    返回:
        output: 注意力输出张量，shape 与 q 相同
    """
    # 创建输出张量
    output = torch.empty_like(q)
    
    # 检查输入维度和类型
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert (
        q.dtype == k.dtype == v.dtype == output.dtype
    ), f"All tensors must have the same dtype: {q.dtype}, {k.dtype}, {v.dtype}, {output.dtype}"
    
    # 获取输入张量的维度
    bs, n_heads, m_size, head_dim = q.size()
    n_size = k.shape[2]
    
    # 定义计算网格
    ## grid的0维 和 block_m_idx = tl.program_id(0) 对应
    grid = lambda meta: (
        triton.cdiv(m_size, meta["BLOCK_M_SIZE"]),  # 序列维度的块数
        bs * n_heads,                                # batch 和 head 维度的总数
        1
    )

    # 启动 kernel 计算
    flash_attention_v1_kernel[grid](
        q, k, v, output,
        *q.stride(),      # (batch, heads, m_size, head_dim)
        *k.stride(),      # (batch, heads, n_size, head_dim)
        *v.stride(),      # (batch, heads, n_size, head_dim)
        *output.stride(), # (batch, heads, m_size, n_size)
        n_heads,
        m_size,
        n_size,
        head_dim,
        64,              # BLOCK_M_SIZE
        64,              # BLOCK_N_SIZE
        sm_scale
    )
    return output