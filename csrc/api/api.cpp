#include <pybind11/pybind11.h>

#include "sparse_fwd.h"
#include "sparse_bwd.h"
#include "sparse_decode.h"
#include "dense_decode.h"
#include "dense_fwd.h"
#include "mxfp8_sparse_fwd.h"
// #include "mxfp8_sparse_decode.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("sparse_decode_fwd", &sparse_attn_decode_interface);
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
    m.def("sparse_prefill_bwd", &sparse_attn_prefill_bwd_interface);
    m.def("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd", &FMHACutlassSM100BwdRun);
    m.def("mxfp8_sparse_prefill_fwd", &mxfp8_sparse_attn_prefill_interface);
    // m.def("mxfp8_sparse_decode_fwd", &mxfp8_sparse_attn_decode_interface);
}
