#include <pybind11/pybind11.h>

#include "api/mxfp8_sparse_decode.h"
#include "api/mxfp8_sparse_fwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mxfp8_sparse_prefill_fwd", &mxfp8_sparse_attn_prefill_interface);
    m.def("mxfp8_sparse_decode_fwd", &mxfp8_sparse_attn_decode_interface);
}
