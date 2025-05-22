cutlass = cutlass + cute

nvcc -std=c++17 examples/cute/tutorial/tiled_copy.cu -o examples/cute/tutorial/tiled_copy_app -I include -I tools/util/include

*** setup cutlass gemm

https://github.com/pytorch-labs/applied-ai/blob/main/kernels/cuda/cutlass_gemm/setup.py
https://research.colfax-intl.com/tutorial-python-binding-for-cuda-libraries-in-pytorch/

*** 编程模型

- underscores: some_tensor(2, _, _)
- local_tile
- local_partition
- partition_D
- partition_S
- _3

- [A..B) half-open integers

- CuTe的mode概念

	在CuTe的语境下：

	- Mode 是构成数据布局描述的基本单元。
	- 它存在于一个（可能） 嵌套的元组结构 中。
	- 一个mode可以是一个 标量值 （比如一个整数，代表一个维度的大小或步长），也可以是 另一个元组 （代表更深层次的结构或组合的维度特征）。
	这种设计允许CuTe以一种非常灵活和可组合的方式来描述和操作复杂的多维数据布局，这对于在GPU上高效实现线性代数运算（如GEMM）至关重要。
	例如，一个张量的形状 (M, N, K) 可以被看作是一个包含三个mode（M, N, K）的元组。如果这个张量在内存中是分块存储的，
	那么每个mode（M, N, K）自身可能又被一个更复杂的mode（可能是一个元组）所替代，用以描述其分块细节。

- indexing: dot_product((i,j,k), (1,M,MN))
	- layout: (M,N,K):(1,M,MN)
		- stride: (1,M,MN)
		- shape: (M,N,K)
	- 定义concat layout: (3:1, 2:3) = (3,2):(1,3)
	- nested layout: ((3,4), 2):((1,3), 40)
		- accept coordinates like ((1,2), 1)


e.g.  (3,4):(1,3)    -> column major
	  (3,4):(4,1)   -> row major

(a,b,c,d):LayoutLeft = (a,b,c,d):(1,a,ab,abc) -> Generalized Column Major
(a,b,c,d):LayoutRight = (a,b,c,d):(bcd,cd,d,1) -> Row Major 逐行存储


** tile

- layout：仅改变shape，stride和大tensor一致
- 多个tile之间：layout一致、engine(tensor offset)和shape不一致

e.g (5,6)起点的tile(3,5):(1,M) offset: 5*1+6*M


big tensor: T, tile: t

- with_shape
	function composition: T and layout(t)
	如果静态，可以compile time生成
- local_tile
- local_partition
- compose/composition 

A*B*C、a*b*c

T1 tiled by T2
T1/T2 -> T3

(A,B,C)/(a,b,c) = ((a,b,c), (A/a, B/b, C/c))
				  inner mode    outer mode

(A,B,C)/(a,c) = ((a,c), (A/a,B,C/c))


** layout的tile运算

layout/shape -> layout

(A,B,C):(1,A,AB) / (a,b,c)
->
(a,b,c):(1,M,MN),(A/a,B/b,C/c):(a,bM,cMN)

** nested layout

比如 (thread_idx, value_idx)对应 (threads_per_block, value_num), threads_per_block是一个mode


** 关于check
利用_3等概念，check at compile time

** non contiguous

3:2    ->    i=0,1,2, offset=2*i



*** cute/tutorials

* tiled_copy.cu

nvcc -std=c++17 examples/cute/tutorial/tiled_copy.cu -o examples/cute/tutorial/tiled_copy_app -I include -I tools/util/include
