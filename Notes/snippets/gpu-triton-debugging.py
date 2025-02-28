### debug utils

import os
from IPython.core.debugger import set_trace

import torch
import triton
import triton.language as tl

os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

def check_tensors_gpu_ready(*tensors):
    for t in tensors:
        assert t.is_contiguous, "A tensor is not contiguous"
        if not os.environ.get('TRITON_INTERPRET') == '1': assert t.is_cuda, "A tensor is not on cuda"

def test_pid_conds(conds, pid_0=[0], pid_1=[0], pid_2=[0]):
    '''Test if condition on pids are fulfilled
    E.g.:
        '=0'  checks that pid_0 == 0
        ',>1' checks that pid_1 > 1
        '>1,=0' checks that pid_0 > 1 and pid_1 == 0
    '''
    pids = pid_0[0], pid_1[0], pid_2[0]
    conds = conds.replace(' ','').split(',')
    for i, (cond, pid) in enumerate(zip(conds, pids)):
        if cond=='': continue
        op, threshold = cond[0], int(cond[1:])
        if op not in ['<','>','>=','<=','=', '!=']: raise ValueError(f"Rules may only use these ops: '<','>','>=','<=','=', '!='. Invalid rule: '{condition}'.")
        op = '==' if op == '=' else op
        if not eval(f'{pid} {op} {threshold}'): return False
    return True

assert test_pid_conds('')
assert test_pid_conds('>0', [1], [1])
assert not test_pid_conds('>0', [0], [1])
assert test_pid_conds('=0,=1', [0], [1], [0])

def breakpoint_if(conds, pid_0=[0], pid_1=[0], pid_2=[0]):
    '''Stop kernel, if any condition of pids is fulfilled'''
    if test_pid_conds(conds, pid_0, pid_1, pid_2): set_trace()

def print_if(txt, conds, pid_0=[0], pid_1=[0], pid_2=[0]):
    '''Print txt, if any condition of pids is fulfilled'''
    if test_pid_conds(conds, pid_0, pid_1, pid_2): print(txt)

def cdiv(a,b): return (a + b - 1) // b
assert cdiv(10,2)==5
assert cdiv(10,3)==4


### interpret mode


TRITON_INTERPRET=1 python interpret_triton_square.py

@triton.jit()
def square_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
  # The rows of the softmax are independent, so we parallelize across those
  row_idx = tl.program_id(0)
  # The stride represents how much we need to increase the pointer to advance 1 row
  row_start_ptr = input_ptr + row_idx * input_row_stride
  # The block size is the next power of two greater than n_cols, so we can fit each
  # row in a single block
  col_offsets = tl.arange(0, BLOCK_SIZE)
  breakpoint()
  input_ptrs = row_start_ptr + col_offsets
  # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
  row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

  square_output = row * row
  breakpoint()
  # Write back output to DRAM
  output_row_start_ptr = output_ptr + row_idx * output_row_stride
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)