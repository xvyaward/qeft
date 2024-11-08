#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
// #include "quantization_new/dequantize/dequantize_cuda.h"
#include "quantization_new/gemm/gemm_cuda.h"
#include "quantization_new/gemv/gemv_cuda.h"
#include "attention/ft_attention.h"
#include "layernorm/layernorm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 4bit
  // m.def("dequantize_weight_4bit", &dequantize_weight_4bit, "Dequantize 4-bit weight matrix to fp16, bf16 per-channel");
  // m.def("dequantize_weight_4bit_group", &dequantize_weight_4bit_group, "Dequantize 4-bit weight matrix to fp16, bf16 group");
  // m.def("dequantize_weight_4bit_qeft", &dequantize_weight_4bit_qeft, "Dequantize 4-bit weight matrix to fp16, bf16 group");

  // new awq-style kernel
  m.def("gemm_4bit", &gemm_4bit, "New quantized GEMM kernel.");
  m.def("gemv_4bit", &gemv_4bit, "New quantized GEMV kernel.");
  m.def("gemv_4bit_qeft", &gemv_4bit_qeft, "New quantized GEMV kernel with reordered high-precision weight.");

  // faster transformer
  m.def("layernorm_forward_cuda", &layernorm_forward_cuda, "FasterTransformer layernorm kernel");
  m.def("single_query_attention", &single_query_attention, "Attention with a single query",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("k_cache"), py::arg("v_cache"),
        py::arg("length_per_sample_"), py::arg("alibi_slopes_"), py::arg("timestep"), py::arg("rotary_embedding_dim")=0,
        py::arg("rotary_base")=10000.0f, py::arg("neox_rotary_style")=true);
}
