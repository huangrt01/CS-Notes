### 细节的优化算子

torch._foreach_zero_


### 写Op

#include <torch/extension.h>

TORCH_LIBRARY(my_op_cpu, m) {
  m.class_<monotorch::FeatureMapCPU>("MyOpCPU")
    .def(torch::init<int64_t, double_t, int64_t, int64_t>())
    .def("reinit", &MyWorkspace::MyOpCPU::reinit)
}