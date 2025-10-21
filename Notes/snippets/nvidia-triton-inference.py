*** 容器

https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags

*** tritonserver

tritonserver --model-repository=/path [--strict-model-config=false]

** config.pbtxt

name: "resnet50_trt"
platform: "tensorrt_plan"
max_batch_size : 8
input [
	{
		name: "input"
		data_type: TYPE_FP32
		format: FORMAT_NCHW
		dims: [ 3, 256, 256 ]
	}
]
output [
	{
		name: "output"
		data_type: TYPE_FP32
		dims: [ 1000 ]
		label_filename: "labels.txt"
	}
]
dynamic_batching {
	preferred_batch_size: [ 2, 4 ]
}

version_policy:{ all{}}
version_policy:{latest {num_versions: 1}}
version_policy:{specific { versions:1,2}}

instance_group [
count:2
kind: KIND_CPU

count:1
kind: KIND_GPU
gpus: [ 0 ]

count:2
kind: KIND_GPU
gpus: [ 1, 2 ]

]

- dims不包括batch size维
- max_batch_size可以为0
- libtorch -1可变长
- torch script: INPUT__0
- dims后面可以配置 reshape { shape: [1, 3, 224, 224] }


Necessary Parameters
- platform / backend: to define which backend to use
- max_batch_size
- input and output

TensorRT, TensorFlow saved-model, and ONNX models do nnot require config.pbtxt when --strict-model-config=false


* 显式加载
--model-control-mode explicit
curl -X POST http://localhost:8000/v2/repository//models/cpu_1_ins/load  (unload)