#include <pybind11/pybind11.h>

#include "api/fp8_sparse_decode.h"
#include "api/fp8_sparse_fwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_sparse_prefill_fwd", &fp8_sparse_attn_prefill_interface);
    m.def("fp8_sparse_decode_fwd", &fp8_sparse_attn_decode_interface);
}
