/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests to check sub_group_mask extract_bits()
//
*******************************************************************************/

#include "sub_group_mask_common.h"

#define TEST_NAME sub_group_mask_extract_bits

namespace TEST_NAMESPACE {

using namespace sycl_cts;
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK

template <typename T>
void get_expected_bits(T &out, uint32_t mask_size, int pos) {
  if (pos >= mask_size - 1) return;
  int init;
  if (pos % 2 == 0)
    init = 1;  // 01
  else
    init = 2;  // 10
  out = init;
  for (int i = 2; i + 2 <= sizeof(T) * CHAR_BIT && i + 2 <= mask_size - pos;
       i = i + 2) {
    out <<= 2;
    out += init;
  }
}

template <typename T>
struct check_result_extract_bits {
  bool operator()(const sycl::ext::oneapi::sub_group_mask &sub_group_mask,
                  const sycl::sub_group &) {
    for (int pos = 0; pos <= sub_group_mask.size(); pos++) {
      T bits;
      sub_group_mask.extract_bits(bits, sycl::id(pos));
      T expected(0);
      get_expected_bits(expected, sub_group_mask.size(), pos);
      if (bits != expected) return false;
    }
    return true;
  }
};

template <typename T>
struct check_type_extract_bits {
  bool operator()(const sycl::ext::oneapi::sub_group_mask &sub_group_mask) {
    T bits;
    return std::is_same<void,
                        decltype(sub_group_mask.extract_bits(bits))>::value;
  }
};

template <typename T>
struct check_for_type {
  void operator()(util::logger &log, const std::string &typeName) {
    log.note("testing: " + type_name_string<T>::get(typeName));
    check_const_api<check_result_extract_bits<T>, check_type_extract_bits<T>,
                    even_predicate, T>(log);
  }
};
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK

/** test sycl::oneapi::sub_group_mask interface
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK
    for_all_types_and_marrays<check_for_type>(types, log);
#else
    log.note("SYCL_EXT_ONEAPI_SUB_GROUP_MASK is not defined, test is skipped");
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
