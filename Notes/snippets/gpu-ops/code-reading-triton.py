


# atomic_cas，支持block处理

https://github.com/triton-lang/triton/pull/2514

- test_core中的用法测试



# reduce op

lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp


struct ReduceOpConversion
    : public ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp>
   	LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

  	// First reduce all the values along axis within each thread.
    reduceWithinThreads(helper, srcValues, accs, indices, rewriter);

    // Then reduce across threads within a warp.
    reduceWithinWarps(helper, accs, rewriter);

    storeWarpReduceToSharedMemory(helper, accs, indices, smemBases, rewriter);


    accumulatePartialReductions(helper, smemBases, rewriter);

    loadReductionAndPackResult(helper, smemShape, smemBases, rewriter);