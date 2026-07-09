# Testing

## Running Tests

```bash
# From build directory:
ctest --output-on-failure         # All tests
ctest -R test_tensor               # Tensor tests only
ctest -R test_attention            # Attention tests only
ctest -R test_inference            # Inference + data pipeline tests
ctest -R "test_arc|test_ser"       # ARC + SER tests
```

## Test Categories

| Directory | Contents |
|-----------|----------|
| `tests/unit/` | Per-component unit tests |
| `tests/integration/` | Multi-component integration tests |
| `tests/benchmark/` | Performance benchmarks |
| `tests/security/` | S0-S3 crypto and security tests |
| `tests/fuzz/` | Fuzz harnesses (planned) |

## Adding a Test

1. Create `tests/unit/test_<component>.c`
2. Include `test_common.h` for ASSERT macros
3. Define `main()` with `run_test()` calls and `RUN_ALL_TESTS()`
4. Add to `CMakeLists.txt` in `tests/`

```c
#include "test_common.h"
#include "my_component.h"

static void test_foo(void) {
    ASSERT_EQ(SNEPPX_foo(2, 2), 4, "2+2 == 4");
}

int main(void) {
    run_test("foo", test_foo);
    RUN_ALL_TESTS();
}
```

## Known Issues

- **test_argon2**: 1 pre-existing timing edge case (3/4 pass)
- **test_ser_train**: 1 pre-existing flaky assertion (2/3 pass)
- **test_ed25519**: timeout on some hardware (slow vector verification)
- **test_thread**: timeout (sleep-based timing in single-threaded mode)

These are accepted as pre-existing and do not indicate regression.
