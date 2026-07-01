# ============================================================================
# ARIX-Algo Makefile — convenience targets
# ============================================================================
.PHONY: all build test clean format lint coverage docs stats

BUILD_DIR ?= build
BUILD_TYPE ?= Release
CMAKE_ARGS ?= -DARIX_BUILD_TESTS=ON -DARIX_BUILD_BENCHMARKS=ON

all: build

build:
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CMAKE_ARGS)
	cmake --build $(BUILD_DIR) -j$$(($$(nproc 2>/dev/null || echo 4)))

test: build
	cd $(BUILD_DIR) && ctest --output-on-failure --timeout 60

clean:
	rm -rf $(BUILD_DIR) build-coverage coverage_report

format:
	clang-format -i -style=file $$(find . -name '*.c' -o -name '*.h' | grep -v build/ | grep -v target/)

lint:
	clang-format --dry-run -Werror -style=file $$(find . -name '*.c' -o -name '*.h' | grep -v build/ | grep -v target/)

coverage:
	@scripts/coverage.sh

docs:
	@echo "See docs/ directory for project documentation."

stats:
	@scripts/stats.sh

help:
	@echo "Targets:"
	@echo "  build     — Configure and build (default)"
	@echo "  test      — Run test suite"
	@echo "  clean     — Remove build artifacts"
	@echo "  format    — Format source files in-place"
	@echo "  lint      — Check formatting without modifying"
	@echo "  coverage  — Generate coverage report (Linux, GCC)"
	@echo "  stats     — Show project statistics"
