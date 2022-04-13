/*******************************************************************************
//
//  SYCL 2022 Conformance Test Suite
//
//  Provide verification for CUDA backend availability
//
*******************************************************************************/

#include "../common/common.h"

#ifdef SYCL_EXT_ONEAPI_BACKEND_CUDA

#include <cuda.h>
#include <sycl/ext/oneapi/experimental/backend/cuda.hpp>

#endif  // SYCL_EXT_ONEAPI_BACKEND_CUDA

#define TEST_NAME cuda_backend_availability

// now -> future
// SYCL_EXT_ONEAPI_BACKEND_CUDA -> SYCL_BACKEND_CUDA
// ext_oneapi_cuda -> cuda (present but deprecated)

namespace TEST_NAMESPACE {
using namespace sycl_cts;

class cuda_selector : public sycl::device_selector {
 public:
  int operator()(const sycl::device& dev) const override {
    return (dev.get_platform().get_info<sycl::info::platform::name>().find(
                "CUDA") != std::string::npos)
               ? 1
               : -1;
  }
};

class TEST_NAME : public sycl_cts::util::test_base {
  /** return information about this test
   */
  void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
   */
  void run(util::logger& log) override {
#ifdef SYCL_EXT_ONEAPI_BACKEND_CUDA
    {
      sycl::queue q{cuda_selector()};  // util::get_cts_object::queue()};

      //    sycl::is_backend_active is not implemented
      //    if (sycl::is_backend_active<sycl::backend::ext_oneapi_cuda>::value)
      //      FAIL(log, "sycl::backend::ext_oneapi_cuda is not active.");

      if (q.get_backend() != sycl::backend::ext_oneapi_cuda)
        FAIL(log, "sycl::backend::ext_oneapi_cuda is not available.");
    }
#else
    FAIL(log, "SYCL_BACKEND_CUDA preprocessor macro is not defined.");
#endif  // SYCL_EXT_ONEAPI_BACKEND_CUDA
  }
};

util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE
