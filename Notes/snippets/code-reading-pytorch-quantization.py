### torchao
speed-first

* https://github.com/pytorch-labs/ao
- Current best way to access all the pytorch native gpu quantization work,
- Used by sdxl-fast and segment-anything-fast

* https://github.com/pytorch-labs/gpt-fast
- (has GPTQ support for int4)
- Best for weight only quant

* https://github.com/pytorch-labs/segment-anything-fast
- Best for dynamic quant

* https://gist.github.com/HDCharles/287ac5e997c7a8cf031004aad0e3a941 ■ triton microbenchmarks



# float 8 
https://github.com/pytorch-labs/float8_experimental




### dynamic quantization

from torchao.quantization import apply_dynamic_quant
import copy
model2 = copy.deepcopy(model)
apply_dynamic_quant(model2)
torch.__inductor.config.force_fuse_int_mm_with_mul = True
model_c = torch.compile(model2, mode='max-autotune')
model_c(image)


config.force_fuse_int_mm_with_mul -> 对int8 matmul和pointwise scale操作进行fusion

def tuned_fused_int_mm_mul(mat1, mat2, mat3, out_dtype, *, layout=None):
    out_dtype = (
        torch.promote_types(mat3.get_dtype(), torch.int32)
        if out_dtype is None
        else out_dtype
    )
    m, n, k, layout, mat1, mat2, mat3 = mm_args(
        mat1, mat2, mat3, layout=layout, out_dtype=out_dtype
    )

    def mul_epilogue(v1, v2):
        return V.ops.mul(v1, v2)

    choices: List[Dict[Any, Any]] = []
    for config in int8_mm_configs(
        m, n, k, **mm_config_kwargs(ir.get_device_type(mat1))
    ):
        mm_template.maybe_append_choice(
            choices,
            input_nodes=(mat1, mat2, mat3),
            layout=layout,
            **dict(mm_options(config, m, n, k, layout), ACC_TYPE="tl.int32"),
            suffix_args=1,
            epilogue_fn=mul_epilogue,
        )
    return autotune_select_algorithm("int_mm", choices, [mat1, mat2, mat3], layout)


### weight-only quantization

from torchao.quantization import apply_weight_only_int8_quant
import copy
model2 = copy.deepcopy(model)
apply_weight_only_int8_quant(model2)
torch.__inductor.config.use_mixed_mm = True
model_c = torch.compile(model2, mode='max-autotune')
model_c(image)


torch/_inductor/kernel/mm.py

mm_template

TritonTemplate(...)
	if B_PROLOGUE_CAST_TYPE is not None:
     b = b.to(B_PROLOGUE_CAST_TYPE)

### int4

_inductor/kernel/unpack_mixed_mm.py
性能差

--> _convert_weight_to_int4pack

triton缺点：
L2 cache optimization, quantization时不好弄;
config heuristic



int4 CPULayout
https://github.com/pytorch/ao/pull/1278/files
https://github.com/pytorch/ao/issues/1117#issuecomment-2451252756.


# decompostion.py

基于 量化+triton，优化weighted sumpooling。 这样写之后，triton op会每列一个线程，速度快

config.coordinate_descent_tuning
if guard_size_oblivious(self.shape[0] == 1) or guard_size_oblivious(
            input2.shape[1] == 1
        ):
   return (self.unsqueeze(2) * input2.unsqueeze(0)).sum(dim=1)



### bitsandbytes
accuracy-first

https://huggingface.co/docs/bitsandbytes/main/en/index

* 8-bit optimizers uses block-wise quantization to maintain 32-bit performance at a small fraction of the memory cost.
* LLM.int8() or 8-bit quantization enables large language model inference with only half the required memory
 and without any performance degradation. This method is based on vector-wise quantization to quantize most features to 8-bits and separately treating outliers with 16-bit matrix multiplication.
* QLoRA or 4-bit quantization enables large language model training with several 
memory-saving techniques that don’t compromise performance. This method quantizes a model to 4-bits and inserts a small set of trainable low-rank adaptation (LoRA) weights to allow training.