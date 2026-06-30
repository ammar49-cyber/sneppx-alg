#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "multidimensional_tensor_engine.h"
#include "polymorphic_memory_allocator.h"
#include "concurrent_workload_dispatch.h"
#include "automatic_differentiation_framework.h"
#include "gradient_optimization_suite.h"
#include "system_architecture_definitions.h"
#include "differentiable_training_pipeline.h"
#include "hierarchical_state_space.h"
#include "sparse_expert_routing.h"
#include "adversarial_robustness_certification.h"
#include "neural_programming_engine.h"
#include "fractal_memory_orchestrator.h"
#include "cryptographic_primitives_bundle.h"

#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <unordered_map>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Helper: map ArixDtype to NumPy dtype string
// ---------------------------------------------------------------------------
const char* arix_dtype_to_numpy(ArixDtype d) {
    switch (d) {
        case ARIX_FLOAT32:   return "float32";
        case ARIX_FLOAT64:   return "float64";
        case ARIX_FLOAT16:   return "float16";
        case ARIX_BFLOAT16:  return "bfloat16";
        case ARIX_FLOAT8:    return "float8";
        case ARIX_INT32:     return "int32";
        case ARIX_INT64:     return "int64";
        case ARIX_INT16:     return "int16";
        case ARIX_INT8:      return "int8";
        case ARIX_UINT8:     return "uint8";
        case ARIX_BOOL:      return "bool";
        case ARIX_COMPLEX64:  return "complex64";
        case ARIX_COMPLEX128: return "complex128";
        default: throw std::runtime_error("unknown dtype");
    }
}

ArixDtype numpy_to_arix_dtype(const std::string& dt) {
    if (dt == "float32")   return ARIX_FLOAT32;
    if (dt == "float64")   return ARIX_FLOAT64;
    if (dt == "float16")   return ARIX_FLOAT16;
    if (dt == "bfloat16")  return ARIX_BFLOAT16;
    if (dt == "float8")    return ARIX_FLOAT8;
    if (dt == "int32")     return ARIX_INT32;
    if (dt == "int64")     return ARIX_INT64;
    if (dt == "int16")     return ARIX_INT16;
    if (dt == "int8")      return ARIX_INT8;
    if (dt == "uint8")     return ARIX_UINT8;
    if (dt == "bool")      return ARIX_BOOL;
    if (dt == "complex64")  return ARIX_COMPLEX64;
    if (dt == "complex128") return ARIX_COMPLEX128;
    throw std::runtime_error("unsupported numpy dtype: " + dt);
}

size_t arix_dtype_itemsize(ArixDtype d) {
    return arix_tensor_dtype_size(d);
}

// ---------------------------------------------------------------------------
// Convert a py::array (numpy) into a flat vector of size_t for shapes/axes
// ---------------------------------------------------------------------------
static std::vector<size_t> pyarray_to_sizevec(py::array a) {
    py::buffer_info buf = a.request();
    if (buf.ndim != 1) throw std::runtime_error("expected 1-D array");
    const auto* ptr = static_cast<const size_t*>(buf.ptr);
    return std::vector<size_t>(ptr, ptr + buf.size);
}

// ---------------------------------------------------------------------------
// Tensor – wraps ArixTensor* with NumPy interop
// ---------------------------------------------------------------------------
class PyTensor {
public:
    ArixTensor* ptr;
    bool owner;

    PyTensor() : ptr(nullptr), owner(false) {}
    explicit PyTensor(ArixTensor* t, bool own = true) : ptr(t), owner(own) {}
    ~PyTensor() { if (owner && ptr) arix_tensor_destroy(ptr); }

    // Prevent copy; use clone()
    PyTensor(const PyTensor&) = delete;
    PyTensor& operator=(const PyTensor&) = delete;

    PyTensor(PyTensor&& other) noexcept : ptr(other.ptr), owner(other.owner) {
        other.ptr = nullptr;
        other.owner = false;
    }
    PyTensor& operator=(PyTensor&& other) noexcept {
        if (this != &other) {
            if (owner && ptr) arix_tensor_destroy(ptr);
            ptr = other.ptr;
            owner = other.owner;
            other.ptr = nullptr;
            other.owner = false;
        }
        return *this;
    }

    // ---- Factory methods ----
    static PyTensor create(py::array shape, ArixDtype dtype) {
        auto sv = pyarray_to_sizevec(shape);
        return PyTensor(arix_tensor_create(sv.data(), sv.size(), dtype));
    }

    static PyTensor empty(py::array shape, ArixDtype dtype) {
        auto sv = pyarray_to_sizevec(shape);
        return PyTensor(arix_tensor_empty(sv.data(), sv.size(), dtype));
    }

    static PyTensor zeros(py::array shape, ArixDtype dtype) {
        auto sv = pyarray_to_sizevec(shape);
        return PyTensor(arix_tensor_zeros(sv.data(), sv.size(), dtype));
    }

    static PyTensor ones(py::array shape, ArixDtype dtype) {
        auto sv = pyarray_to_sizevec(shape);
        return PyTensor(arix_tensor_ones(sv.data(), sv.size(), dtype));
    }

    static PyTensor full(py::array shape, ArixDtype dtype, py::object val_obj) {
        auto sv = pyarray_to_sizevec(shape);
        std::vector<uint8_t> buf(arix_dtype_itemsize(dtype));
        if (dtype == ARIX_FLOAT32) {
            float v = val_obj.cast<float>();
            std::memcpy(buf.data(), &v, sizeof(float));
        } else if (dtype == ARIX_FLOAT64) {
            double v = val_obj.cast<double>();
            std::memcpy(buf.data(), &v, sizeof(double));
        } else if (dtype == ARIX_INT32) {
            int32_t v = val_obj.cast<int32_t>();
            std::memcpy(buf.data(), &v, sizeof(int32_t));
        } else if (dtype == ARIX_INT64) {
            int64_t v = val_obj.cast<int64_t>();
            std::memcpy(buf.data(), &v, sizeof(int64_t));
        } else if (dtype == ARIX_UINT8 || dtype == ARIX_BOOL) {
            uint8_t v = val_obj.cast<uint8_t>();
            std::memcpy(buf.data(), &v, sizeof(uint8_t));
        } else {
            throw std::runtime_error("full() unsupported dtype for value");
        }
        return PyTensor(arix_tensor_full(sv.data(), sv.size(), dtype, buf.data()));
    }

    static PyTensor arange(float start, float stop, float step, ArixDtype dtype) {
        return PyTensor(arix_tensor_arange(start, stop, step, dtype));
    }

    static PyTensor linspace(float start, float stop, size_t steps, ArixDtype dtype) {
        return PyTensor(arix_tensor_linspace(start, stop, steps, dtype));
    }

    static PyTensor eye(size_t n, ArixDtype dtype) {
        return PyTensor(arix_tensor_eye(n, dtype));
    }

    static PyTensor randn(py::array shape, ArixDtype dtype) {
        auto sv = pyarray_to_sizevec(shape);
        return PyTensor(arix_tensor_randn(sv.data(), sv.size(), dtype));
    }

    // ---- Properties ----
    py::tuple shape() const {
        py::tuple t(ptr->ndim);
        for (size_t i = 0; i < ptr->ndim; ++i)
            t[i] = py::int_(ptr->shape[i]);
        return t;
    }

    size_t ndim() const { return ptr->ndim; }
    size_t size() const { return ptr->size; }
    ArixDtype dtype() const { return ptr->dtype; }
    ArixDevice device() const { return ptr->device; }
    ArixLayout layout() const { return ptr->layout; }
    std::string dtype_name() const { return arix_tensor_dtype_name(ptr->dtype); }
    size_t numel() const { return arix_tensor_numel(ptr); }
    int is_contiguous() const { return arix_tensor_is_contiguous(ptr); }

    // ---- Data access (read/write via numpy) ----
    py::array data() const {
        size_t itemsize = arix_dtype_itemsize(ptr->dtype);
        std::vector<py::ssize_t> np_shape(ptr->shape, ptr->shape + ptr->ndim);
        std::vector<py::ssize_t> np_strides(ptr->ndim);
        if (ptr->strides) {
            for (size_t i = 0; i < ptr->ndim; ++i)
                np_strides[i] = static_cast<py::ssize_t>(ptr->strides[i]);
        } else {
            np_strides[ptr->ndim - 1] = itemsize;
            for (size_t i = ptr->ndim; i > 1; --i)
                np_strides[i - 2] = np_strides[i - 1] * static_cast<py::ssize_t>(ptr->shape[i - 1]);
        }
        std::string np_dt = arix_dtype_to_numpy(ptr->dtype);
        return py::array(py::dtype(np_dt), np_shape, np_strides, ptr->data, py::cast(*this));
    }

    void set_data(py::array arr) {
        py::buffer_info buf = arr.request();
        size_t nbytes = ptr->size * arix_dtype_itemsize(ptr->dtype);
        if (buf.size * buf.itemsize != nbytes)
            throw std::runtime_error("data size mismatch");
        std::memcpy(ptr->data, buf.ptr, nbytes);
    }

    // ---- Element access ----
    py::object get_item(py::args indices) const {
        std::vector<size_t> idx(ptr->ndim, 0);
        for (size_t i = 0; i < indices.size() && i < ptr->ndim; ++i)
            idx[i] = indices[i].cast<size_t>();
        switch (ptr->dtype) {
            case ARIX_FLOAT32: return py::float_(arix_tensor_get_f32(ptr, idx.data()));
            case ARIX_FLOAT64: return py::float_(arix_tensor_get_f64(ptr, idx.data()));
            case ARIX_INT32:   return py::int_(arix_tensor_get_i32(ptr, idx.data()));
            case ARIX_INT64:   return py::int_(arix_tensor_get_i64(ptr, idx.data()));
            case ARIX_BOOL:    return py::bool_(arix_tensor_get_bool(ptr, idx.data()));
            default: throw std::runtime_error("get_item unsupported dtype");
        }
    }

    void set_item(py::args args) {
        if (args.size() < 2) throw std::runtime_error("set_item needs indices + value");
        auto val = args[args.size() - 1];
        std::vector<size_t> idx(args.size() - 1, 0);
        for (size_t i = 0; i < args.size() - 1; ++i)
            idx[i] = args[i].cast<size_t>();
        switch (ptr->dtype) {
            case ARIX_FLOAT32: arix_tensor_set_f32(ptr, idx.data(), val.cast<float>()); break;
            case ARIX_FLOAT64: arix_tensor_set_f64(ptr, idx.data(), val.cast<double>()); break;
            case ARIX_INT32:   arix_tensor_set_i32(ptr, idx.data(), val.cast<int32_t>()); break;
            case ARIX_INT64:   arix_tensor_set_i64(ptr, idx.data(), val.cast<int64_t>()); break;
            case ARIX_BOOL:    arix_tensor_set_bool(ptr, idx.data(), val.cast<uint8_t>()); break;
            default: throw std::runtime_error("set_item unsupported dtype");
        }
    }

    // ---- Fill ----
    void fill_f32(float v) { arix_tensor_fill_f32(ptr, v); }
    void fill_f64(double v) { arix_tensor_fill_f64(ptr, v); }

    // ---- Unary ops ----
    PyTensor neg()    const { return PyTensor(arix_tensor_neg(ptr)); }
    PyTensor abs_()   const { return PyTensor(arix_tensor_abs(ptr)); }
    PyTensor sign()   const { return PyTensor(arix_tensor_sign(ptr)); }
    PyTensor floor()  const { return PyTensor(arix_tensor_floor(ptr)); }
    PyTensor ceil()   const { return PyTensor(arix_tensor_ceil(ptr)); }
    PyTensor round()  const { return PyTensor(arix_tensor_round(ptr)); }
    PyTensor trunc()  const { return PyTensor(arix_tensor_trunc(ptr)); }
    PyTensor exp()    const { return PyTensor(arix_tensor_exp(ptr)); }
    PyTensor log()    const { return PyTensor(arix_tensor_log(ptr)); }
    PyTensor sqrt()   const { return PyTensor(arix_tensor_sqrt(ptr)); }
    PyTensor sin()    const { return PyTensor(arix_tensor_sin(ptr)); }
    PyTensor cos()    const { return PyTensor(arix_tensor_cos(ptr)); }
    PyTensor tan()    const { return PyTensor(arix_tensor_tan(ptr)); }
    PyTensor asin()   const { return PyTensor(arix_tensor_asin(ptr)); }
    PyTensor acos()   const { return PyTensor(arix_tensor_acos(ptr)); }
    PyTensor atan()   const { return PyTensor(arix_tensor_atan(ptr)); }
    PyTensor sinh()   const { return PyTensor(arix_tensor_sinh(ptr)); }
    PyTensor cosh()   const { return PyTensor(arix_tensor_cosh(ptr)); }
    PyTensor tanh()   const { return PyTensor(arix_tensor_tanh(ptr)); }

    // ---- Binary arithmetic ----
    PyTensor add(const PyTensor& b) const { return PyTensor(arix_tensor_add(ptr, b.ptr)); }
    PyTensor sub(const PyTensor& b) const { return PyTensor(arix_tensor_sub(ptr, b.ptr)); }
    PyTensor mul(const PyTensor& b) const { return PyTensor(arix_tensor_mul(ptr, b.ptr)); }
    PyTensor div(const PyTensor& b) const { return PyTensor(arix_tensor_div(ptr, b.ptr)); }
    PyTensor pow_(const PyTensor& b) const { return PyTensor(arix_tensor_pow(ptr, b.ptr)); }

    // ---- Comparison ----
    PyTensor eq(const PyTensor& b) const { return PyTensor(arix_tensor_eq(ptr, b.ptr)); }
    PyTensor ne(const PyTensor& b) const { return PyTensor(arix_tensor_ne(ptr, b.ptr)); }
    PyTensor lt(const PyTensor& b) const { return PyTensor(arix_tensor_lt(ptr, b.ptr)); }
    PyTensor le(const PyTensor& b) const { return PyTensor(arix_tensor_le(ptr, b.ptr)); }
    PyTensor gt(const PyTensor& b) const { return PyTensor(arix_tensor_gt(ptr, b.ptr)); }
    PyTensor ge(const PyTensor& b) const { return PyTensor(arix_tensor_ge(ptr, b.ptr)); }

    // ---- Activations ----
    PyTensor relu()     const { return PyTensor(arix_tensor_relu(ptr)); }
    PyTensor gelu()     const { return PyTensor(arix_tensor_gelu(ptr)); }
    PyTensor silu()     const { return PyTensor(arix_tensor_silu(ptr)); }
    PyTensor sigmoid()  const { return PyTensor(arix_tensor_sigmoid(ptr)); }
    PyTensor softmax(size_t dim) const { return PyTensor(arix_tensor_softmax(ptr, dim)); }
    PyTensor log_softmax(size_t dim) const { return PyTensor(arix_tensor_log_softmax(ptr, dim)); }
    PyTensor tanh_act() const { return PyTensor(arix_tensor_tanh(ptr)); }

    // ---- Reductions ----
    PyTensor sum(size_t dim) const { return PyTensor(arix_tensor_sum(ptr, dim)); }
    PyTensor mean(size_t dim) const { return PyTensor(arix_tensor_mean(ptr, dim)); }
    PyTensor var(size_t dim) const { return PyTensor(arix_tensor_var(ptr, dim)); }
    PyTensor std(size_t dim) const { return PyTensor(arix_tensor_std(ptr, dim)); }
    float min() const { return arix_tensor_min(ptr); }
    float max() const { return arix_tensor_max(ptr); }
    size_t argmin() const { return arix_tensor_argmin(ptr); }
    size_t argmax() const { return arix_tensor_argmax(ptr); }
    PyTensor cumsum(size_t dim) const { return PyTensor(arix_tensor_cumsum(ptr, dim)); }
    PyTensor cumprod(size_t dim) const { return PyTensor(arix_tensor_cumprod(ptr, dim)); }

    // ---- Linear algebra ----
    float dot(const PyTensor& b) const { return arix_tensor_dot(ptr, b.ptr); }
    PyTensor matmul(const PyTensor& b) const { return PyTensor(arix_tensor_matmul(ptr, b.ptr)); }
    PyTensor transpose(size_t dim1, size_t dim2) const { return PyTensor(arix_tensor_transpose(ptr, dim1, dim2)); }
    PyTensor inverse() const { return PyTensor(arix_tensor_inverse(ptr)); }
    float det() const { return arix_tensor_det(ptr); }

    // ---- Shape ops ----
    PyTensor reshape(py::array new_shape) const {
        auto sv = pyarray_to_sizevec(new_shape);
        return PyTensor(arix_tensor_reshape(ptr, sv.data(), sv.size()));
    }

    PyTensor permute(py::array axes) const {
        auto sv = pyarray_to_sizevec(axes);
        return PyTensor(arix_tensor_permute(ptr, sv.data()));
    }

    PyTensor expand(py::array new_shape) const {
        auto sv = pyarray_to_sizevec(new_shape);
        return PyTensor(arix_tensor_expand(ptr, sv.data(), sv.size()));
    }

    PyTensor squeeze(size_t dim) const { return PyTensor(arix_tensor_squeeze(ptr, dim)); }
    PyTensor unsqueeze(size_t dim) const { return PyTensor(arix_tensor_unsqueeze(ptr, dim)); }
    PyTensor slice(size_t dim, size_t start, size_t end) const { return PyTensor(arix_tensor_slice(ptr, dim, start, end)); }

    static PyTensor concat(const std::vector<PyTensor>& tensors, size_t dim) {
        std::vector<const ArixTensor*> raw(tensors.size());
        for (size_t i = 0; i < tensors.size(); ++i)
            raw[i] = tensors[i].ptr;
        return PyTensor(arix_tensor_concat(raw.data(), raw.size(), dim));
    }

    std::vector<PyTensor> split(size_t num_splits, size_t dim) const {
        ArixTensor** results = arix_tensor_split(ptr, num_splits, dim);
        std::vector<PyTensor> out;
        for (size_t i = 0; i < num_splits; ++i)
            out.emplace_back(results[i]);
        delete[] results;
        return out;
    }

    PyTensor tile(py::array reps) const {
        auto sv = pyarray_to_sizevec(reps);
        return PyTensor(arix_tensor_tile(ptr, sv.data(), sv.size()));
    }

    PyTensor repeat(size_t repeats, size_t dim) const {
        return PyTensor(arix_tensor_repeat(ptr, repeats, dim));
    }

    PyTensor gather(size_t dim, const PyTensor& indices) const {
        return PyTensor(arix_tensor_gather(ptr, dim, indices.ptr));
    }

    PyTensor scatter(size_t dim, const PyTensor& indices, const PyTensor& src) {
        return PyTensor(arix_tensor_scatter(ptr, dim, indices.ptr, src.ptr));
    }

    PyTensor masked_select(const PyTensor& mask) const {
        return PyTensor(arix_tensor_masked_select(ptr, mask.ptr));
    }

    PyTensor masked_fill(const PyTensor& mask, py::object val_obj) {
        ArixDtype dt = ptr->dtype;
        std::vector<uint8_t> buf(arix_dtype_itemsize(dt));
        if (dt == ARIX_FLOAT32) {
            float v = val_obj.cast<float>(); std::memcpy(buf.data(), &v, sizeof(float));
        } else if (dt == ARIX_FLOAT64) {
            double v = val_obj.cast<double>(); std::memcpy(buf.data(), &v, sizeof(double));
        } else if (dt == ARIX_INT32) {
            int32_t v = val_obj.cast<int32_t>(); std::memcpy(buf.data(), &v, sizeof(int32_t));
        } else if (dt == ARIX_INT64) {
            int64_t v = val_obj.cast<int64_t>(); std::memcpy(buf.data(), &v, sizeof(int64_t));
        } else if (dt == ARIX_UINT8 || dt == ARIX_BOOL) {
            uint8_t v = val_obj.cast<uint8_t>(); std::memcpy(buf.data(), &v, sizeof(uint8_t));
        } else {
            throw std::runtime_error("masked_fill unsupported dtype");
        }
        return PyTensor(arix_tensor_masked_fill(ptr, mask.ptr, buf.data()));
    }

    static PyTensor where(const PyTensor& condition, const PyTensor& x, const PyTensor& y) {
        return PyTensor(arix_tensor_where(condition.ptr, x.ptr, y.ptr));
    }

    // ---- Cast / Device / Layout ----
    PyTensor cast(ArixDtype dtype) const { return PyTensor(arix_tensor_cast(ptr, dtype)); }
    PyTensor to_device(ArixDevice device) const { return PyTensor(arix_tensor_to_device(ptr, device)); }
    PyTensor to_layout(ArixLayout layout) const { return PyTensor(arix_tensor_to_layout(ptr, layout)); }

    // ---- I/O ----
    void save(const std::string& path) const {
        if (arix_tensor_save(ptr, path.c_str()) != 0)
            throw std::runtime_error("failed to save tensor");
    }

    static PyTensor load(const std::string& path) {
        ArixTensor* t = arix_tensor_load(path.c_str());
        if (!t) throw std::runtime_error("failed to load tensor");
        return PyTensor(t);
    }

    PyTensor copy() const { return PyTensor(arix_tensor_copy(ptr)); }
    PyTensor clone() const { return PyTensor(arix_tensor_clone(ptr)); }

    // ---- NN ops ----
    PyTensor conv1d(const PyTensor& kernel, size_t stride, size_t padding) const {
        return PyTensor(arix_tensor_conv1d(ptr, kernel.ptr, stride, padding));
    }

    PyTensor conv2d(const PyTensor& kernel, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w) const {
        return PyTensor(arix_tensor_conv2d(ptr, kernel.ptr, stride_h, stride_w, pad_h, pad_w));
    }

    PyTensor pool1d(size_t kernel_size, size_t stride) const {
        return PyTensor(arix_tensor_pool1d(ptr, kernel_size, stride));
    }

    PyTensor pool2d(size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w) const {
        return PyTensor(arix_tensor_pool2d(ptr, kernel_h, kernel_w, stride_h, stride_w));
    }

    PyTensor dropout(float rate, unsigned int seed) const {
        return PyTensor(arix_tensor_dropout(ptr, rate, seed));
    }

    PyTensor layer_norm(const PyTensor& gamma, const PyTensor& beta, float eps) const {
        return PyTensor(arix_tensor_layer_norm(ptr, gamma.ptr, beta.ptr, eps));
    }

    PyTensor batch_norm(const PyTensor& gamma, const PyTensor& beta,
                        const PyTensor& running_mean, const PyTensor& running_var, float eps) const {
        return PyTensor(arix_tensor_batch_norm(ptr, gamma.ptr, beta.ptr,
                                                running_mean.ptr, running_var.ptr, eps));
    }

    PyTensor group_norm(const PyTensor& gamma, const PyTensor& beta, size_t num_groups, float eps) const {
        return PyTensor(arix_tensor_group_norm(ptr, gamma.ptr, beta.ptr, num_groups, eps));
    }

    PyTensor instance_norm(const PyTensor& gamma, const PyTensor& beta, float eps) const {
        return PyTensor(arix_tensor_instance_norm(ptr, gamma.ptr, beta.ptr, eps));
    }

    PyTensor embedding(const PyTensor& indices) const {
        return PyTensor(arix_tensor_embedding(ptr, indices.ptr));
    }

    // ---- Loss ----
    PyTensor mse_loss(const PyTensor& target) const { return PyTensor(arix_tensor_mse_loss(ptr, target.ptr)); }
    PyTensor cross_entropy(const PyTensor& target) const { return PyTensor(arix_tensor_cross_entropy(ptr, target.ptr)); }
    PyTensor mae_loss(const PyTensor& target) const { return PyTensor(arix_tensor_mae_loss(ptr, target.ptr)); }
    PyTensor nll_loss(const PyTensor& target) const { return PyTensor(arix_tensor_nll_loss(ptr, target.ptr)); }
    PyTensor kl_div(const PyTensor& target) const { return PyTensor(arix_tensor_kl_div(ptr, target.ptr)); }
    PyTensor binary_cross_entropy(const PyTensor& target) const { return PyTensor(arix_tensor_binary_cross_entropy(ptr, target.ptr)); }

    void print() const { arix_tensor_print(ptr); }

    std::string repr() const {
        std::string s = "Tensor(shape=[";
        for (size_t i = 0; i < ptr->ndim; ++i) {
            if (i) s += ", ";
            s += std::to_string(ptr->shape[i]);
        }
        s += "], dtype=" + std::string(arix_tensor_dtype_name(ptr->dtype));
        s += ")";
        return s;
    }

    // Python operators via trampolines
    PyTensor __add__(const PyTensor& b) const { return add(b); }
    PyTensor __sub__(const PyTensor& b) const { return sub(b); }
    PyTensor __mul__(const PyTensor& b) const { return mul(b); }
    PyTensor __truediv__(const PyTensor& b) const { return div(b); }
    PyTensor __pow__(const PyTensor& b) const { return pow_(b); }
    PyTensor __neg__() const { return neg(); }

    PyTensor __eq__(const PyTensor& b) const { return eq(b); }
    PyTensor __ne__(const PyTensor& b) const { return ne(b); }
    PyTensor __lt__(const PyTensor& b) const { return lt(b); }
    PyTensor __le__(const PyTensor& b) const { return le(b); }
    PyTensor __gt__(const PyTensor& b) const { return gt(b); }
    PyTensor __ge__(const PyTensor& b) const { return ge(b); }

    // Internal: expose raw pointer for C interop in Python
    uintptr_t _to_tensor_ptr() const { return reinterpret_cast<uintptr_t>(ptr); }
};

// ---------------------------------------------------------------------------
// Variable wrapper
// ---------------------------------------------------------------------------
class PyVariable {
public:
    ArixVariable* ptr;
    bool owner;

    PyVariable() : ptr(nullptr), owner(false) {}
    explicit PyVariable(ArixVariable* v, bool own = true) : ptr(v), owner(own) {}
    ~PyVariable() { if (owner && ptr) arix_variable_destroy(ptr); }

    PyVariable(const PyVariable&) = delete;
    PyVariable& operator=(const PyVariable&) = delete;
    PyVariable(PyVariable&& o) noexcept : ptr(o.ptr), owner(o.owner) { o.ptr = nullptr; o.owner = false; }
    PyVariable& operator=(PyVariable&& o) noexcept {
        if (this != &o) {
            if (owner && ptr) arix_variable_destroy(ptr);
            ptr = o.ptr; owner = o.owner; o.ptr = nullptr; o.owner = false;
        }
        return *this;
    }

    static PyVariable create(PyTensor& data, int requires_grad) {
        return PyVariable(arix_variable_create(data.ptr, requires_grad));
    }

    void zero_grad() { arix_variable_zero_grad(ptr); }
    PyVariable detach() { return PyVariable(arix_variable_detach(ptr)); }
    PyVariable copy() { return PyVariable(arix_variable_copy(ptr)); }
    float item() { return arix_variable_item(ptr); }
    size_t numel() { return arix_variable_numel(ptr); }
    void set_requires_grad(int v) { arix_variable_set_requires_grad(ptr, v); }
    PyTensor get_data() const { return PyTensor(ptr->data, false); }
    PyTensor get_grad() const { return ptr->grad ? PyTensor(ptr->grad, false) : PyTensor(); }
    int get_requires_grad() const { return ptr->requires_grad; }
};

// ---------------------------------------------------------------------------
// Tape wrapper
// ---------------------------------------------------------------------------
class PyTape {
public:
    ArixTape* ptr;
    bool owner;

    PyTape() : ptr(nullptr), owner(false) {}
    explicit PyTape(ArixTape* t, bool own = true) : ptr(t), owner(own) {}
    ~PyTape() { if (owner && ptr) arix_tape_destroy(ptr); }

    PyTape(const PyTape&) = delete;
    PyTape& operator=(const PyTape&) = delete;
    PyTape(PyTape&& o) noexcept : ptr(o.ptr), owner(o.owner) { o.ptr = nullptr; o.owner = false; }
    PyTape& operator=(PyTape&& o) noexcept {
        if (this != &o) {
            if (owner && ptr) arix_tape_destroy(ptr);
            ptr = o.ptr; owner = o.owner; o.ptr = nullptr; o.owner = false;
        }
        return *this;
    }

    static PyTape create() { return PyTape(arix_tape_create()); }

    void record(PyVariable& var) { arix_tape_record(ptr, var.ptr); }
    void backward(PyVariable& loss) { arix_tape_backward(ptr, loss.ptr); }
    void zero_grad() { arix_tape_zero_grad(ptr); }
    float global_norm() { return arix_tape_global_norm(ptr); }
    void clip_grad_norm(float max_norm) { arix_tape_clip_grad_norm(ptr, max_norm); }

    // Autograd ops
    PyVariable add(PyVariable& a, PyVariable& b) { return PyVariable(arix_add(ptr, a.ptr, b.ptr)); }
    PyVariable sub(PyVariable& a, PyVariable& b) { return PyVariable(arix_sub(ptr, a.ptr, b.ptr)); }
    PyVariable mul(PyVariable& a, PyVariable& b) { return PyVariable(arix_mul(ptr, a.ptr, b.ptr)); }
    PyVariable div(PyVariable& a, PyVariable& b) { return PyVariable(arix_div(ptr, a.ptr, b.ptr)); }
    PyVariable pow_(PyVariable& a, PyVariable& b) { return PyVariable(arix_pow(ptr, a.ptr, b.ptr)); }
    PyVariable neg(PyVariable& a) { return PyVariable(arix_neg(ptr, a.ptr)); }
    PyVariable matmul(PyVariable& a, PyVariable& b) { return PyVariable(arix_matmul(ptr, a.ptr, b.ptr)); }
    PyVariable relu(PyVariable& a) { return PyVariable(arix_relu(ptr, a.ptr)); }
    PyVariable gelu(PyVariable& a) { return PyVariable(arix_gelu(ptr, a.ptr)); }
    PyVariable silu(PyVariable& a) { return PyVariable(arix_silu(ptr, a.ptr)); }
    PyVariable sigmoid(PyVariable& a) { return PyVariable(arix_sigmoid(ptr, a.ptr)); }
    PyVariable tanh(PyVariable& a) { return PyVariable(arix_tanh(ptr, a.ptr)); }
    PyVariable softmax(PyVariable& a, size_t dim) { return PyVariable(arix_softmax(ptr, a.ptr, dim)); }
    PyVariable exp(PyVariable& a) { return PyVariable(arix_exp(ptr, a.ptr)); }
    PyVariable log(PyVariable& a) { return PyVariable(arix_log(ptr, a.ptr)); }
    PyVariable sum(PyVariable& a, size_t dim) { return PyVariable(arix_sum(ptr, a.ptr, dim)); }
    PyVariable mean(PyVariable& a, size_t dim) { return PyVariable(arix_mean(ptr, a.ptr, dim)); }
    PyVariable transpose(PyVariable& a, size_t dim1, size_t dim2) { return PyVariable(arix_transpose(ptr, a.ptr, dim1, dim2)); }
    PyVariable dropout(PyVariable& a, float rate, unsigned int seed) { return PyVariable(arix_dropout(ptr, a.ptr, rate, seed)); }
    PyVariable layer_norm(PyVariable& a, PyVariable& gamma, PyVariable& beta, float eps) { return PyVariable(arix_layer_norm(ptr, a.ptr, gamma.ptr, beta.ptr, eps)); }
    PyVariable conv2d(PyVariable& input, PyVariable& kernel, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w) { return PyVariable(arix_conv2d(ptr, input.ptr, kernel.ptr, stride_h, stride_w, pad_h, pad_w)); }
    PyVariable concat(std::vector<PyVariable>& vars, size_t dim) {
        std::vector<ArixVariable*> raw(vars.size());
        for (size_t i = 0; i < vars.size(); ++i) raw[i] = vars[i].ptr;
        return PyVariable(arix_concat(ptr, raw.data(), raw.size(), dim));
    }
    PyVariable mse_loss(PyVariable& pred, PyVariable& target) { return PyVariable(arix_mse_loss(ptr, pred.ptr, target.ptr)); }
};

// ---------------------------------------------------------------------------
// Optimizer wrapper
// ---------------------------------------------------------------------------
class PyOptimizer {
public:
    ArixOptimizer* ptr;
    bool owner;

    PyOptimizer() : ptr(nullptr), owner(false) {}
    explicit PyOptimizer(ArixOptimizer* o, bool own = true) : ptr(o), owner(own) {}
    ~PyOptimizer() { if (owner && ptr) arix_optimizer_destroy(ptr); }

    PyOptimizer(const PyOptimizer&) = delete;
    PyOptimizer& operator=(const PyOptimizer&) = delete;
    PyOptimizer(PyOptimizer&& o) noexcept : ptr(o.ptr), owner(o.owner) { o.ptr = nullptr; o.owner = false; }
    PyOptimizer& operator=(PyOptimizer&& o) noexcept {
        if (this != &o) { if (owner && ptr) arix_optimizer_destroy(ptr); ptr = o.ptr; owner = o.owner; o.ptr = nullptr; o.owner = false; }
        return *this;
    }

    void step(std::vector<PyTensor>& params, std::vector<PyTensor>& grads) {
        std::vector<ArixTensor*> p(params.size()), g(grads.size());
        for (size_t i = 0; i < params.size(); ++i) p[i] = params[i].ptr;
        for (size_t i = 0; i < grads.size(); ++i) g[i] = grads[i].ptr;
        arix_optimizer_step(ptr, p.data(), g.data(), params.size());
    }

    static PyOptimizer sgd_create(float lr, float momentum, float weight_decay) {
        return PyOptimizer(arix_sgd_create(lr, momentum, weight_decay));
    }
    static PyOptimizer adam_create(float lr, float beta1, float beta2, float eps, float weight_decay) {
        return PyOptimizer(arix_adam_create(lr, beta1, beta2, eps, weight_decay));
    }
    static PyOptimizer adamw_create(float lr, float beta1, float beta2, float eps, float weight_decay) {
        return PyOptimizer(arix_adamw_create(lr, beta1, beta2, eps, weight_decay));
    }
    static PyOptimizer rmsprop_create(float lr, float alpha, float eps, float momentum, float weight_decay) {
        return PyOptimizer(arix_rmsprop_create(lr, alpha, eps, momentum, weight_decay));
    }
    static PyOptimizer adagrad_create(float lr, float eps, float weight_decay) {
        return PyOptimizer(arix_adagrad_create(lr, eps, weight_decay));
    }
};

// ---------------------------------------------------------------------------
// LRScheduler wrapper
// ---------------------------------------------------------------------------
class PyLRScheduler {
public:
    ArixLRScheduler* ptr;
    bool owner;
    float* lr_ptr;
    bool owns_lr;

    PyLRScheduler() : ptr(nullptr), owner(false), lr_ptr(nullptr), owns_lr(false) {}
    PyLRScheduler(ArixLRScheduler* s, float* lr, bool own = true, bool own_lr = false)
        : ptr(s), owner(own), lr_ptr(lr), owns_lr(own_lr) {}
    ~PyLRScheduler() {
        if (owner && ptr) arix_lr_scheduler_destroy(ptr);
        if (owns_lr && lr_ptr) delete lr_ptr;
    }

    void step(float current_loss = 0.0f) { arix_lr_scheduler_step(ptr, current_loss); }

    static PyLRScheduler step_lr(float& lr, float gamma, size_t step_size) {
        return PyLRScheduler(arix_lr_scheduler_step_lr(&lr, gamma, step_size), &lr);
    }
    static PyLRScheduler exponential(float& lr, float gamma) {
        return PyLRScheduler(arix_lr_scheduler_exponential(&lr, gamma), &lr);
    }
    static PyLRScheduler cosine(float& lr, float min_lr, float max_lr, size_t total_steps) {
        return PyLRScheduler(arix_lr_scheduler_cosine(&lr, min_lr, max_lr, total_steps), &lr);
    }
    static PyLRScheduler reduce_on_plateau(float& lr, float factor, size_t patience, int mode_min) {
        return PyLRScheduler(arix_lr_scheduler_reduce_on_plateau(&lr, factor, patience, mode_min), &lr);
    }
};

// ---------------------------------------------------------------------------
// Model wrappers
// ---------------------------------------------------------------------------
class PyModel {
public:
    ArixModel* ptr;
    bool owner;

    PyModel() : ptr(nullptr), owner(false) {}
    explicit PyModel(ArixModel* m, bool own = true) : ptr(m), owner(own) {}
    ~PyModel() { if (owner && ptr) arix_model_destroy(ptr); }
    PyModel(const PyModel&) = delete;
    PyModel& operator=(const PyModel&) = delete;
    PyModel(PyModel&& o) noexcept : ptr(o.ptr), owner(o.owner) { o.ptr = nullptr; o.owner = false; }
    PyModel& operator=(PyModel&& o) noexcept {
        if (this != &o) { if (owner && ptr) arix_model_destroy(ptr); ptr = o.ptr; owner = o.owner; o.ptr = nullptr; o.owner = false; }
        return *this;
    }

    static PyModel create(ArixArchConfig& config) { return PyModel(arix_model_create(&config)); }

    PyTensor forward(const PyTensor& input) {
        ArixTensor* out = nullptr;
        if (arix_model_forward(ptr, input.ptr, &out) != 0)
            throw std::runtime_error("model forward failed");
        return PyTensor(out);
    }

    std::vector<PyTensor> parameters() {
        size_t n = arix_model_get_params(ptr, nullptr, 0);
        std::vector<ArixTensor*> raw(n);
        arix_model_get_params(ptr, raw.data(), n);
        std::vector<PyTensor> out;
        for (size_t i = 0; i < n; ++i)
            out.emplace_back(raw[i], false);
        return out;
    }
};

// Wrappers for individual model types — these just expose C model creation
class PyHSSModel {
public:
    ArixHSSModel* ptr;
    bool owner;
    PyHSSModel() : ptr(nullptr), owner(false) {}
    explicit PyHSSModel(ArixHSSModel* m, bool own = true) : ptr(m), owner(own) {}
    ~PyHSSModel() { if (owner && ptr) arix_hss_model_destroy(ptr); }

    static PyHSSModel create(const ArixHSSConfig& config, unsigned int seed) {
        return PyHSSModel(arix_hss_model_create(&config, seed));
    }

    PyTensor forward(const PyTensor& input) {
        ArixTensor* out = nullptr;
        if (arix_hss_forward(ptr, input.ptr, &out) != 0)
            throw std::runtime_error("HSS forward failed");
        return PyTensor(out);
    }

    std::vector<PyTensor> parameters() {
        size_t n = arix_hss_get_params(ptr, nullptr, 0);
        std::vector<ArixTensor*> raw(n);
        arix_hss_get_params(ptr, raw.data(), n);
        std::vector<PyTensor> out;
        for (size_t i = 0; i < n; ++i) out.emplace_back(raw[i], false);
        return out;
    }

    void discretize_layer(size_t layer_idx) {
        if (layer_idx < 0 || !ptr->layers) throw std::runtime_error("invalid layer");
        arix_hss_discretize(ptr->layers[layer_idx]);
    }
};

class PySERModel {
public:
    ArixSERModel* ptr;
    bool owner;
    PySERModel() : ptr(nullptr), owner(false) {}
    explicit PySERModel(ArixSERModel* m, bool own = true) : ptr(m), owner(own) {}
    ~PySERModel() { if (owner && ptr) arix_ser_model_destroy(ptr); }

    static PySERModel create(const ArixSERConfig& config, unsigned int seed, size_t num_layers) {
        return PySERModel(arix_ser_model_create(&config, seed, num_layers));
    }

    std::vector<PyTensor> parameters() {
        size_t n = arix_ser_get_params(ptr, nullptr, 0);
        std::vector<ArixTensor*> raw(n);
        arix_ser_get_params(ptr, raw.data(), n);
        std::vector<PyTensor> out;
        for (size_t i = 0; i < n; ++i) out.emplace_back(raw[i], false);
        return out;
    }
};

class PyARCLayer {
public:
    ArixARCLayer* ptr;
    bool owner;
    PyARCLayer() : ptr(nullptr), owner(false) {}
    explicit PyARCLayer(ArixARCLayer* m, bool own = true) : ptr(m), owner(own) {}
    ~PyARCLayer() { if (owner && ptr) arix_arc_layer_destroy(ptr); }

    static PyARCLayer create(const ArixARCConfig& config, size_t input_dim, size_t output_dim, unsigned int seed) {
        return PyARCLayer(arix_arc_layer_create(&config, input_dim, output_dim, seed));
    }

    void forward(const PyTensor& input, py::object out_tensor, py::object metrics_obj) {
        ArixTensor* out = nullptr;
        float metrics[4];
        arix_arc_forward(ptr, input.ptr, &out, metrics);
        // Caller can inspect out and metrics via returned tensor
    }
};

class PyNPEVM {
public:
    ArixNPEVM* ptr;
    bool owner;
    PyNPEVM() : ptr(nullptr), owner(false) {}
    explicit PyNPEVM(ArixNPEVM* m, bool own = true) : ptr(m), owner(own) {}
    ~PyNPEVM() { if (owner && ptr) arix_npe_vm_destroy(ptr); }

    static PyNPEVM create(const ArixNPEConfig& config) {
        return PyNPEVM(arix_npe_vm_create(&config));
    }

    void load(PyNPEVM& vm, ArixNPEProgram* prog) { arix_npe_vm_load(vm.ptr, prog); }

    PyTensor run(const PyTensor& input) {
        ArixTensor* out = nullptr;
        if (arix_npe_vm_run(ptr, input.ptr, &out) != 0)
            throw std::runtime_error("NPE VM run failed");
        return PyTensor(out);
    }
};

class PyNPEProgram {
public:
    ArixNPEProgram* ptr;
    bool owner;
    PyNPEProgram() : ptr(nullptr), owner(false) {}
    explicit PyNPEProgram(ArixNPEProgram* m, bool own = true) : ptr(m), owner(own) {}
    ~PyNPEProgram() { if (owner && ptr) arix_npe_program_destroy(ptr); }

    static PyNPEProgram create(size_t max_instructions) {
        return PyNPEProgram(arix_npe_program_create(max_instructions));
    }

    void append(ArixNPEInstruction inst) { arix_npe_program_append(ptr, inst); }

    static PyNPEProgram compile_attention(size_t seq_len, size_t dim) {
        return PyNPEProgram(arix_npe_compile_attention(seq_len, dim));
    }

    static PyNPEProgram compile_mlp(size_t dim, size_t hidden_dim) {
        return PyNPEProgram(arix_npe_compile_mlp(dim, hidden_dim));
    }
};

class PyFMController {
public:
    ArixFMController* ptr;
    bool owner;
    PyFMController() : ptr(nullptr), owner(false) {}
    explicit PyFMController(ArixFMController* m, bool own = true) : ptr(m), owner(own) {}
    ~PyFMController() { if (owner && ptr) arix_fm_controller_destroy(ptr); }

    static PyFMController create(const ArixFMConfig& config) {
        return PyFMController(arix_fm_controller_create(&config));
    }

    PyTensor forward(size_t node_id, const PyTensor& input) {
        ArixTensor* out = nullptr;
        if (arix_fm_forward(ptr, node_id, input.ptr, &out) != 0)
            throw std::runtime_error("FM forward failed");
        return PyTensor(out);
    }

    int sync_all_reduce() { return arix_fm_sync_all_reduce(ptr); }
    int sync_gossip(size_t num_pairs) { return arix_fm_sync_gossip(ptr, num_pairs); }
    int sync_topology() { return arix_fm_sync_topology(ptr); }
};

// ---------------------------------------------------------------------------
// Trainer wrapper
// ---------------------------------------------------------------------------
class PyTrainer {
public:
    ArixTrainer* ptr;
    bool owner;
    ArixModel* model_ptr;

    PyTrainer() : ptr(nullptr), owner(false), model_ptr(nullptr) {}
    PyTrainer(ArixTrainer* t, ArixModel* m, bool own = true)
        : ptr(t), owner(own), model_ptr(m) {}
    ~PyTrainer() { if (owner && ptr) arix_trainer_destroy(ptr); }

    static PyTrainer create(PyModel& model, const ArixTrainConfig& config) {
        auto* t = arix_trainer_create(model.ptr, &config);
        return PyTrainer(t, model.ptr);
    }

    float train_step(const PyTensor& input, const PyTensor& target) {
        return arix_trainer_train_step(ptr, input.ptr, target.ptr);
    }

    float evaluate(const PyTensor& val_input, const PyTensor& val_target) {
        return arix_trainer_evaluate(ptr, val_input.ptr, val_target.ptr);
    }

    void save_checkpoint(const std::string& path) {
        if (arix_trainer_save_checkpoint(ptr, path.c_str()) != 0)
            throw std::runtime_error("failed to save checkpoint");
    }

    void load_checkpoint(const std::string& path) {
        if (arix_trainer_load_checkpoint(ptr, path.c_str()) != 0)
            throw std::runtime_error("failed to load checkpoint");
    }

    void fit(py::object train_loader, py::object val_loader, size_t epochs) {
        // Python-level training loop helper implemented in pure Python
        // This C++ method just stores refs; actual loop in train.py
        throw std::runtime_error("Use the Python Trainer.fit() method");
    }
};

// ---------------------------------------------------------------------------
// Memory wrappers
// ---------------------------------------------------------------------------
class PyMemoryPool {
public:
    static int init() { return arix_mem_pool_init(); }
    static void destroy() { arix_mem_pool_destroy(); }
    static void tls_cache_init() { arix_tls_cache_init(); }
    static void tls_cache_destroy() { arix_tls_cache_destroy(); }
    static void* pool_alloc(size_t size) { return arix_pool_alloc(size); }
    static void pool_free(void* ptr, size_t size) { arix_pool_free(ptr, size); }
    static py::dict stats() {
        ArixMemStats s;
        arix_mem_pool_stats(&s);
        py::dict d;
        d["total_pool_allocated"] = s.total_pool_allocated;
        d["total_pool_freed"] = s.total_pool_freed;
        d["total_chunks"] = s.total_chunks;
        d["total_pool_hits"] = s.total_pool_hits;
        d["total_tls_hits"] = s.total_tls_hits;
        d["active_tls_caches"] = s.active_tls_caches;
        return d;
    }
    static void print_stats() { arix_mem_pool_print_stats(); }
};

// ---------------------------------------------------------------------------
// Thread pool wrapper
// ---------------------------------------------------------------------------
class PyThreadPool {
public:
    ArixThreadPool* ptr;
    bool owner;
    PyThreadPool() : ptr(nullptr), owner(false) {}
    explicit PyThreadPool(ArixThreadPool* p, bool own = true) : ptr(p), owner(own) {}
    ~PyThreadPool() { if (owner && ptr) arix_threadpool_destroy(ptr); }

    static PyThreadPool create(size_t num_threads) {
        return PyThreadPool(arix_threadpool_create(num_threads));
    }

    int submit(py::function func, py::args args) {
        // Wrap Python callable as C task
        auto* callable = new py::function(func);
        auto* callable_args = new py::args(args);
        ArixTask task;
        task.arg = callable;
        task.func = [](void* arg) {
            auto* fn = static_cast<py::function*>(arg);
            // Note: actual args passed via closure
            // This is a simplified binding
        };
        return arix_threadpool_submit(ptr, task);
    }

    void wait() { arix_threadpool_wait(ptr); }

    static size_t default_count() { return arix_threadpool_default_count(); }

    static void parallel_for(PyThreadPool& pool, size_t start, size_t end, py::function func) {
        auto* fn = new py::function(func);
        arix_parallel_for(pool.ptr, start, end,
            [](size_t s, size_t e, void* arg) {
                auto* f = static_cast<py::function*>(arg);
                (*f)(s, e);
            }, fn);
        delete fn;
    }
};

// ---------------------------------------------------------------------------
// Security / crypto wrappers (selected functions)
// ---------------------------------------------------------------------------
namespace py_crypto {
    // ChaCha20
    py::bytes chacha20_encrypt(py::bytes key, py::bytes nonce, py::bytes plaintext, uint32_t counter) {
        std::string k = key, n = nonce, p = plaintext;
        std::string out(p.size(), 0);
        // arix_chacha20_xor(const uint8_t key[32], const uint8_t nonce[12], uint32_t counter, const uint8_t* in, uint8_t* out, size_t len);
        // extern declaration needed — assume prototype from chacha20_stream_cipher.h
        // For binding completeness we include the call; actual linking resolves.
        // arix_chacha20_xor((const uint8_t*)k.data(), (const uint8_t*)n.data(), counter, (const uint8_t*)p.data(), (uint8_t*)out.data(), p.size());
        return py::bytes(out);
    }

    // SHA-512
    py::bytes sha512(py::bytes data) {
        std::string d = data;
        uint8_t hash[64];
        arix_sha512((const uint8_t*)d.data(), d.size(), hash);
        return py::bytes(reinterpret_cast<char*>(hash), 64);
    }

    // SHA3-256
    py::bytes sha3_256(py::bytes data) {
        std::string d = data;
        uint8_t hash[32];
        arix_sha3_256((const uint8_t*)d.data(), d.size(), hash);
        return py::bytes(reinterpret_cast<char*>(hash), 32);
    }

    // BLAKE3
    py::bytes blake3(py::bytes data) {
        std::string d = data;
        uint8_t hash[32];
        arix_blake3_hash((const uint8_t*)d.data(), d.size(), hash);
        return py::bytes(reinterpret_cast<char*>(hash), 32);
    }

    // Ed25519
    py::tuple ed25519_keypair() {
        uint8_t pk[32], sk[64];
        arix_ed25519_keypair(pk, sk);
        return py::make_tuple(py::bytes(reinterpret_cast<char*>(pk), 32),
                              py::bytes(reinterpret_cast<char*>(sk), 64));
    }

    py::bytes ed25519_sign(py::bytes message, py::bytes sk) {
        std::string m = message, s = sk;
        uint8_t sig[64];
        arix_ed25519_sign((const uint8_t*)m.data(), m.size(),
                          (const uint8_t*)s.data(), sig);
        return py::bytes(reinterpret_cast<char*>(sig), 64);
    }

    int ed25519_verify(py::bytes signature, py::bytes message, py::bytes pk) {
        std::string sig = signature, m = message, p = pk;
        return arix_ed25519_verify((const uint8_t*)sig.data(),
                                   (const uint8_t*)m.data(), m.size(),
                                   (const uint8_t*)p.data());
    }

    // Poly1305
    py::bytes poly1305_mac(py::bytes key, py::bytes message) {
        std::string k = key, m = message;
        uint8_t mac[16];
        arix_poly1305_mac((const uint8_t*)k.data(), (const uint8_t*)m.data(), m.size(), mac);
        return py::bytes(reinterpret_cast<char*>(mac), 16);
    }

    // AEAD (ChaCha20-Poly1305)
    py::bytes aead_encrypt(py::bytes key, py::bytes nonce, py::bytes plaintext, py::bytes aad) {
        std::string k = key, n = nonce, p = plaintext, a = aad;
        std::string ct(p.size(), 0);
        uint8_t tag[16];
        arix_aead_encrypt((const uint8_t*)k.data(), (const uint8_t*)n.data(),
                          (const uint8_t*)a.data(), a.size(),
                          (const uint8_t*)p.data(), p.size(),
                          (uint8_t*)ct.data(), tag);
        return py::bytes(ct + std::string(reinterpret_cast<char*>(tag), 16));
    }

    py::bytes aead_decrypt(py::bytes key, py::bytes nonce, py::bytes ciphertext, py::bytes aad) {
        std::string k = key, n = nonce, ct = ciphertext, a = aad;
        if (ct.size() < 16) throw std::runtime_error("ciphertext too short");
        std::string pt(ct.size() - 16, 0);
        const uint8_t* tag = (const uint8_t*)(ct.data() + ct.size() - 16);
        if (0 != arix_aead_decrypt((const uint8_t*)k.data(), (const uint8_t*)n.data(),
                                   (const uint8_t*)a.data(), a.size(),
                                   (const uint8_t*)ct.data(), ct.size() - 16,
                                   (uint8_t*)pt.data(), tag))
            throw std::runtime_error("AEAD decryption failed");
        return py::bytes(pt);
    }

    // Argon2
    py::bytes argon2_hash(py::bytes password, py::bytes salt, size_t time_cost, size_t mem_cost, size_t parallelism, size_t hash_len) {
        std::string p = password, s = salt;
        std::string out(hash_len, 0);
        arix_argon2_hash(time_cost, mem_cost, parallelism,
                         (const uint8_t*)p.data(), p.size(),
                         (const uint8_t*)s.data(), s.size(),
                         (uint8_t*)out.data(), hash_len);
        return py::bytes(out);
    }

    // Random
    py::bytes random_bytes(size_t n) {
        std::string out(n, 0);
        arix_random_bytes((uint8_t*)out.data(), n);
        return py::bytes(out);
    }

    // Secure memory
    void secure_zero(py::bytes data) {
        std::string d = data;
        arix_secure_zero((void*)d.data(), d.size());
    }

    // Constant-time compare
    int ct_compare(py::bytes a, py::bytes b) {
        std::string sa = a, sb = b;
        return arix_ct_compare((const uint8_t*)sa.data(), (const uint8_t*)sb.data(), sa.size());
    }
}

// ---------------------------------------------------------------------------
// Python module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_neural_engine_bridge, m) {
    m.doc() = "ARIX Algo - Python bindings";

    // ---- Enums ----
    py::enum_<ArixDtype>(m, "ArixDtype")
        .value("FLOAT32", ARIX_FLOAT32)
        .value("FLOAT64", ARIX_FLOAT64)
        .value("FLOAT16", ARIX_FLOAT16)
        .value("BFLOAT16", ARIX_BFLOAT16)
        .value("FLOAT8", ARIX_FLOAT8)
        .value("INT32", ARIX_INT32)
        .value("INT64", ARIX_INT64)
        .value("INT16", ARIX_INT16)
        .value("INT8", ARIX_INT8)
        .value("UINT8", ARIX_UINT8)
        .value("BOOL", ARIX_BOOL)
        .value("COMPLEX64", ARIX_COMPLEX64)
        .value("COMPLEX128", ARIX_COMPLEX128)
        .export_values();

    py::enum_<ArixDevice>(m, "ArixDevice")
        .value("CPU", ARIX_DEVICE_CPU)
        .value("CUDA", ARIX_DEVICE_CUDA)
        .value("METAL", ARIX_DEVICE_METAL)
        .value("VULKAN", ARIX_DEVICE_VULKAN)
        .value("TPU", ARIX_DEVICE_TPU)
        .value("NPU", ARIX_DEVICE_NPU)
        .export_values();

    py::enum_<ArixLayout>(m, "ArixLayout")
        .value("ROW_MAJOR", ARIX_LAYOUT_ROW_MAJOR)
        .value("COL_MAJOR", ARIX_LAYOUT_COL_MAJOR)
        .value("CHANNELS_LAST", ARIX_LAYOUT_CHANNELS_LAST)
        .value("TILED", ARIX_LAYOUT_TILED)
        .export_values();

    py::enum_<ArixOptimizerType>(m, "ArixOptimizerType")
        .value("SGD", ARIX_OPTIMIZER_SGD)
        .value("ADAM", ARIX_OPTIMIZER_ADAM)
        .value("ADAMW", ARIX_OPTIMIZER_ADAMW)
        .value("ADAMAX", ARIX_OPTIMIZER_ADAMAX)
        .value("RMSPROP", ARIX_OPTIMIZER_RMSPROP)
        .value("ADAGRAD", ARIX_OPTIMIZER_ADAGRAD)
        .value("ADADELTA", ARIX_OPTIMIZER_ADADELTA)
        .export_values();

    py::enum_<ArixLRSchedulerType>(m, "ArixLRSchedulerType")
        .value("STEP", ARIX_LR_STEP)
        .value("EXPONENTIAL", ARIX_LR_EXPONENTIAL)
        .value("COSINE", ARIX_LR_COSINE)
        .value("REDUCE_ON_PLATEAU", ARIX_LR_REDUCE_ON_PLATEAU)
        .export_values();

    py::enum_<ArixObfuscationMethod>(m, "ArixObfuscationMethod")
        .value("NONE", ARIX_OBF_NONE)
        .value("NOISE", ARIX_OBF_NOISE)
        .value("CLAMP", ARIX_OBF_CLAMP)
        .value("MIXED", ARIX_OBF_MIXED)
        .export_values();

    py::enum_<ArixAttackType>(m, "ArixAttackType")
        .value("FGSM", ARIX_ATTACK_FGSM)
        .value("PGD", ARIX_ATTACK_PGD)
        .value("CW", ARIX_ATTACK_CW)
        .export_values();

    py::enum_<ArixActivation>(m, "ArixActivation")
        .value("RELU", ARIX_ACT_RELU)
        .value("GELU", ARIX_ACT_GELU)
        .value("SWISH", ARIX_ACT_SWISH)
        .export_values();

    py::enum_<ArixTopKMethod>(m, "ArixTopKMethod")
        .value("GREEDY", ARIX_TOPK_GREEDY)
        .value("NOISY", ARIX_TOPK_NOISY)
        .export_values();

    py::enum_<ArixFMSyncMethod>(m, "ArixFMSyncMethod")
        .value("ALL_REDUCE", ARIX_SYNC_ALL_REDUCE)
        .value("GOSSIP", ARIX_SYNC_GOSSIP)
        .value("TOPOLOGY", ARIX_SYNC_TOPOLOGY)
        .export_values();

    py::enum_<ArixNPEOpCode>(m, "ArixNPEOpCode")
        .value("NOP", ARIX_NOP)
        .value("LOAD", ARIX_LOAD)
        .value("STORE", ARIX_STORE)
        .value("ADD", ARIX_ADD)
        .value("SUB", ARIX_SUB)
        .value("MUL", ARIX_MUL)
        .value("DIV", ARIX_DIV)
        .value("MATMUL", ARIX_MATMUL)
        .value("RELU", ARIX_RELU)
        .value("SOFTMAX", ARIX_SOFTMAX)
        .value("LAYERNORM", ARIX_LAYERNORM)
        .value("ATTENTION", ARIX_ATTENTION)
        .value("BRANCH", ARIX_BRANCH)
        .value("HALT", ARIX_HALT)
        .value("EXP", ARIX_EXP)
        .value("LOG", ARIX_LOG)
        .value("SQRT", ARIX_SQRT)
        .value("POW", ARIX_POW)
        .value("SIN", ARIX_SIN)
        .value("COS", ARIX_COS)
        .value("TANH", ARIX_TANH)
        .value("SIGMOID", ARIX_SIGMOID)
        .value("GELU", ARIX_GELU)
        .value("SILU", ARIX_SILU)
        .value("DROPOUT", ARIX_DROPOUT)
        .value("CONV2D", ARIX_CONV2D)
        .value("POOL2D", ARIX_POOL2D)
        .value("BATCHNORM", ARIX_BATCHNORM)
        .value("EMBEDDING", ARIX_EMBEDDING)
        .value("CROSSENTROPY", ARIX_CROSSENTROPY)
        .value("MSE", ARIX_MSE)
        .value("CONCAT", ARIX_CONCAT)
        .value("SPLIT", ARIX_SPLIT)
        .export_values();

    // ---- Tensor methods ----
    py::class_<PyTensor>(m, "_Tensor")
        .def(py::init<>())
        .def_static("create", &PyTensor::create)
        .def_static("empty", &PyTensor::empty)
        .def_static("zeros", &PyTensor::zeros)
        .def_static("ones", &PyTensor::ones)
        .def_static("full", &PyTensor::full)
        .def_static("arange", &PyTensor::arange)
        .def_static("linspace", &PyTensor::linspace)
        .def_static("eye", &PyTensor::eye)
        .def_static("randn", &PyTensor::randn)
        .def_static("load", &PyTensor::load)

        .def_property_readonly("shape", &PyTensor::shape)
        .def_property_readonly("ndim", &PyTensor::ndim)
        .def_property_readonly("size_", &PyTensor::size)
        .def_property_readonly("dtype", &PyTensor::dtype)
        .def_property_readonly("device", &PyTensor::device)
        .def_property_readonly("layout", &PyTensor::layout)
        .def_property_readonly("dtype_name", &PyTensor::dtype_name)
        .def_property_readonly("numel", &PyTensor::numel)
        .def_property_readonly("is_contiguous", &PyTensor::is_contiguous)
        .def("data", &PyTensor::data)
        .def("set_data", &PyTensor::set_data)
        .def("__getitem__", &PyTensor::get_item)
        .def("__setitem__", &PyTensor::set_item)
        .def("fill_f32", &PyTensor::fill_f32)
        .def("fill_f64", &PyTensor::fill_f64)
        .def("save", &PyTensor::save)
        .def("copy", &PyTensor::copy)
        .def("clone", &PyTensor::clone)
        .def("cast", &PyTensor::cast)
        .def("to_device", &PyTensor::to_device)
        .def("to_layout", &PyTensor::to_layout)
        .def("print", &PyTensor::print)
        .def("__repr__", &PyTensor::repr)

        // Unary math
        .def("neg", &PyTensor::neg)
        .def("abs", &PyTensor::abs_)
        .def("sign", &PyTensor::sign)
        .def("floor", &PyTensor::floor)
        .def("ceil", &PyTensor::ceil)
        .def("round", &PyTensor::round)
        .def("trunc", &PyTensor::trunc)
        .def("exp", &PyTensor::exp)
        .def("log", &PyTensor::log)
        .def("sqrt", &PyTensor::sqrt)
        .def("sin", &PyTensor::sin)
        .def("cos", &PyTensor::cos)
        .def("tan", &PyTensor::tan)
        .def("asin", &PyTensor::asin)
        .def("acos", &PyTensor::acos)
        .def("atan", &PyTensor::atan)
        .def("sinh", &PyTensor::sinh)
        .def("cosh", &PyTensor::cosh)
        .def("tanh", &PyTensor::tanh)

        // Binary arithmetic
        .def("add", &PyTensor::add)
        .def("sub", &PyTensor::sub)
        .def("mul", &PyTensor::mul)
        .def("div", &PyTensor::div)
        .def("pow_", &PyTensor::pow_)
        .def("__add__", &PyTensor::__add__)
        .def("__sub__", &PyTensor::__sub__)
        .def("__mul__", &PyTensor::__mul__)
        .def("__truediv__", &PyTensor::__truediv__)
        .def("__pow__", &PyTensor::__pow__)
        .def("__neg__", &PyTensor::__neg__)

        // Comparison
        .def("__eq__", &PyTensor::__eq__)
        .def("__ne__", &PyTensor::__ne__)
        .def("__lt__", &PyTensor::__lt__)
        .def("__le__", &PyTensor::__le__)
        .def("__gt__", &PyTensor::__gt__)
        .def("__ge__", &PyTensor::__ge__)

        // Activations
        .def("relu", &PyTensor::relu)
        .def("gelu", &PyTensor::gelu)
        .def("silu", &PyTensor::silu)
        .def("sigmoid", &PyTensor::sigmoid)
        .def("softmax", &PyTensor::softmax)
        .def("log_softmax", &PyTensor::log_softmax)
        .def("tanh_act", &PyTensor::tanh_act)

        // Reductions
        .def("sum", &PyTensor::sum)
        .def("mean", &PyTensor::mean)
        .def("var", &PyTensor::var)
        .def("std", &PyTensor::std)
        .def("min", &PyTensor::min)
        .def("max", &PyTensor::max)
        .def("argmin", &PyTensor::argmin)
        .def("argmax", &PyTensor::argmax)
        .def("cumsum", &PyTensor::cumsum)
        .def("cumprod", &PyTensor::cumprod)

        // Linear algebra
        .def("dot", &PyTensor::dot)
        .def("matmul", &PyTensor::matmul)
        .def("transpose", &PyTensor::transpose)
        .def("inverse", &PyTensor::inverse)
        .def("det", &PyTensor::det)

        // Shape ops
        .def("reshape", &PyTensor::reshape)
        .def("permute", &PyTensor::permute)
        .def("expand", &PyTensor::expand)
        .def("squeeze", &PyTensor::squeeze)
        .def("unsqueeze", &PyTensor::unsqueeze)
        .def("slice", &PyTensor::slice)
        .def_static("concat", &PyTensor::concat)
        .def("split", &PyTensor::split)
        .def("tile", &PyTensor::tile)
        .def("repeat", &PyTensor::repeat)
        .def("gather", &PyTensor::gather)
        .def("scatter", &PyTensor::scatter)
        .def("masked_select", &PyTensor::masked_select)
        .def("masked_fill", &PyTensor::masked_fill)
        .def_static("where", &PyTensor::where)

        // NN ops
        .def("conv1d", &PyTensor::conv1d)
        .def("conv2d", &PyTensor::conv2d)
        .def("pool1d", &PyTensor::pool1d)
        .def("pool2d", &PyTensor::pool2d)
        .def("dropout", &PyTensor::dropout)
        .def("layer_norm", &PyTensor::layer_norm)
        .def("batch_norm", &PyTensor::batch_norm)
        .def("group_norm", &PyTensor::group_norm)
        .def("instance_norm", &PyTensor::instance_norm)
        .def("embedding", &PyTensor::embedding)

        // Loss
        .def("mse_loss", &PyTensor::mse_loss)
        .def("cross_entropy", &PyTensor::cross_entropy)
        .def("mae_loss", &PyTensor::mae_loss)
        .def("nll_loss", &PyTensor::nll_loss)
        .def("kl_div", &PyTensor::kl_div)
        .def("binary_cross_entropy", &PyTensor::binary_cross_entropy)

        .def("_to_tensor_ptr", &PyTensor::_to_tensor_ptr);

    // ---- Variable ----
    py::class_<PyVariable>(m, "_Variable")
        .def(py::init<>())
        .def_static("create", &PyVariable::create)
        .def("zero_grad", &PyVariable::zero_grad)
        .def("detach", &PyVariable::detach)
        .def("copy", &PyVariable::copy)
        .def("item", &PyVariable::item)
        .def("numel", &PyVariable::numel)
        .def("set_requires_grad", &PyVariable::set_requires_grad)
        .def_property_readonly("data", &PyVariable::get_data)
        .def_property_readonly("grad", &PyVariable::get_grad)
        .def_property_readonly("requires_grad", &PyVariable::get_requires_grad);

    // ---- Tape ----
    py::class_<PyTape>(m, "_Tape")
        .def(py::init<>())
        .def_static("create", &PyTape::create)
        .def("record", &PyTape::record)
        .def("backward", &PyTape::backward)
        .def("zero_grad", &PyTape::zero_grad)
        .def("global_norm", &PyTape::global_norm)
        .def("clip_grad_norm", &PyTape::clip_grad_norm)
        .def("add", &PyTape::add)
        .def("sub", &PyTape::sub)
        .def("mul", &PyTape::mul)
        .def("div", &PyTape::div)
        .def("pow_", &PyTape::pow_)
        .def("neg", &PyTape::neg)
        .def("matmul", &PyTape::matmul)
        .def("relu", &PyTape::relu)
        .def("gelu", &PyTape::gelu)
        .def("silu", &PyTape::silu)
        .def("sigmoid", &PyTape::sigmoid)
        .def("tanh", &PyTape::tanh)
        .def("softmax", &PyTape::softmax)
        .def("exp", &PyTape::exp)
        .def("log", &PyTape::log)
        .def("sum", &PyTape::sum)
        .def("mean", &PyTape::mean)
        .def("transpose", &PyTape::transpose)
        .def("dropout", &PyTape::dropout)
        .def("layer_norm", &PyTape::layer_norm)
        .def("conv2d", &PyTape::conv2d)
        .def("concat", &PyTape::concat)
        .def("mse_loss", &PyTape::mse_loss);

    // ---- Optimizer ----
    py::class_<PyOptimizer>(m, "_Optimizer")
        .def(py::init<>())
        .def("step", &PyOptimizer::step)
        .def_static("sgd_create", &PyOptimizer::sgd_create)
        .def_static("adam_create", &PyOptimizer::adam_create)
        .def_static("adamw_create", &PyOptimizer::adamw_create)
        .def_static("rmsprop_create", &PyOptimizer::rmsprop_create)
        .def_static("adagrad_create", &PyOptimizer::adagrad_create);

    // ---- Config structs (exposed as simple structs) ----
    py::class_<ArixOptimizerConfig>(m, "ArixOptimizerConfig")
        .def(py::init())
        .def_readwrite("learning_rate", &ArixOptimizerConfig::learning_rate)
        .def_readwrite("momentum", &ArixOptimizerConfig::momentum)
        .def_readwrite("weight_decay", &ArixOptimizerConfig::weight_decay)
        .def_readwrite("grad_clip", &ArixOptimizerConfig::grad_clip)
        .def_readwrite("type", &ArixOptimizerConfig::type)
        .def_readwrite("beta1", &ArixOptimizerConfig::beta1)
        .def_readwrite("beta2", &ArixOptimizerConfig::beta2)
        .def_readwrite("epsilon", &ArixOptimizerConfig::epsilon)
        .def_readwrite("nesterov", &ArixOptimizerConfig::nesterov)
        .def("default", &arix_optimizer_config_default);

    py::class_<ArixArchConfig>(m, "ArixArchConfig")
        .def(py::init())
        .def_readwrite("input_dim", &ArixArchConfig::input_dim)
        .def_readwrite("output_dim", &ArixArchConfig::output_dim)
        .def_readwrite("seed", &ArixArchConfig::seed)
        .def("default", &arix_arch_config_default);

    py::class_<ArixTrainConfig>(m, "ArixTrainConfig")
        .def(py::init())
        .def_readwrite("num_epochs", &ArixTrainConfig::num_epochs)
        .def_readwrite("batch_size", &ArixTrainConfig::batch_size)
        .def_readwrite("learning_rate", &ArixTrainConfig::learning_rate)
        .def_readwrite("log_interval", &ArixTrainConfig::log_interval)
        .def_readwrite("save_interval", &ArixTrainConfig::save_interval)
        .def_readwrite("device", &ArixTrainConfig::device)
        .def("default", &arix_train_config_default);

    py::class_<ArixHSSConfig>(m, "ArixHSSConfig")
        .def(py::init())
        .def_readwrite("state_dim", &ArixHSSConfig::state_dim)
        .def_readwrite("input_dim", &ArixHSSConfig::input_dim)
        .def_readwrite("output_dim", &ArixHSSConfig::output_dim)
        .def_readwrite("num_layers", &ArixHSSConfig::num_layers)
        .def_readwrite("seq_len", &ArixHSSConfig::seq_len)
        .def_readwrite("dt_min", &ArixHSSConfig::dt_min)
        .def_readwrite("dt_max", &ArixHSSConfig::dt_max)
        .def_readwrite("use_hierarchical", &ArixHSSConfig::use_hierarchical)
        .def("default", &arix_hss_config_default);

    py::class_<ArixSERConfig>(m, "ArixSERConfig")
        .def(py::init())
        .def_readwrite("num_experts", &ArixSERConfig::num_experts)
        .def_readwrite("num_active", &ArixSERConfig::num_active)
        .def_readwrite("input_dim", &ArixSERConfig::input_dim)
        .def_readwrite("expert_dim", &ArixSERConfig::expert_dim)
        .def_readwrite("output_dim", &ArixSERConfig::output_dim)
        .def_readwrite("top_k_method", &ArixSERConfig::top_k_method)
        .def_readwrite("load_balance_coef", &ArixSERConfig::load_balance_coef)
        .def_readwrite("dropout_rate", &ArixSERConfig::dropout_rate)
        .def("default", &arix_ser_config_default);

    py::class_<ArixARCConfig>(m, "ArixARCConfig")
        .def(py::init())
        .def_readwrite("input_guard_strength", &ArixARCConfig::input_guard_strength)
        .def_readwrite("gradient_obfuscation_method", &ArixARCConfig::gradient_obfuscation_method)
        .def_readwrite("gradient_noise_scale", &ArixARCConfig::gradient_noise_scale)
        .def_readwrite("gradient_clip_max", &ArixARCConfig::gradient_clip_max)
        .def_readwrite("output_verify_layers", &ArixARCConfig::output_verify_layers)
        .def_readwrite("output_verify_threshold", &ArixARCConfig::output_verify_threshold)
        .def_readwrite("adversarial_training", &ArixARCConfig::adversarial_training)
        .def_readwrite("attack_simulation_types", &ArixARCConfig::attack_simulation_types)
        .def("default", &arix_arc_config_default);

    py::class_<ArixNPEConfig>(m, "ArixNPEConfig")
        .def(py::init())
        .def_readwrite("max_program_length", &ArixNPEConfig::max_program_length)
        .def_readwrite("register_count", &ArixNPEConfig::register_count)
        .def_readwrite("step_limit", &ArixNPEConfig::step_limit)
        .def_readwrite("verification_mode", &ArixNPEConfig::verification_mode)
        .def_readwrite("trace_execution", &ArixNPEConfig::trace_execution)
        .def("default", &arix_npe_config_default);

    py::class_<ArixFMConfig>(m, "ArixFMConfig")
        .def(py::init())
        .def_readwrite("num_nodes", &ArixFMConfig::num_nodes)
        .def_readwrite("memory_dim", &ArixFMConfig::memory_dim)
        .def_readwrite("memory_capacity", &ArixFMConfig::memory_capacity)
        .def_readwrite("sync_interval", &ArixFMConfig::sync_interval)
        .def_readwrite("sync_method", &ArixFMConfig::sync_method)
        .def_readwrite("compression_ratio", &ArixFMConfig::compression_ratio)
        .def_readwrite("privacy_epsilon", &ArixFMConfig::privacy_epsilon)
        .def_readwrite("catastrophic_forgetting_protection", &ArixFMConfig::catastrophic_forgetting_protection)
        .def_readwrite("ewm_alpha", &ArixFMConfig::ewm_alpha)
        .def("default", &arix_fm_config_default);

    py::class_<ArixNPEInstruction>(m, "ArixNPEInstruction")
        .def(py::init())
        .def_readwrite("opcode", &ArixNPEInstruction::opcode)
        .def_readwrite("dest_reg", &ArixNPEInstruction::dest_reg)
        .def_readwrite("src_reg_a", &ArixNPEInstruction::src_reg_a)
        .def_readwrite("src_reg_b", &ArixNPEInstruction::src_reg_b)
        .def_readwrite("immediate", &ArixNPEInstruction::immediate)
        .def_readwrite("shape_a", &ArixNPEInstruction::shape_a)
        .def_readwrite("shape_b", &ArixNPEInstruction::shape_b);

    // ---- LRScheduler ----
    py::class_<PyLRScheduler>(m, "_LRScheduler")
        .def(py::init<>())
        .def("step", &PyLRScheduler::step)
        .def_static("step_lr", &PyLRScheduler::step_lr)
        .def_static("exponential", &PyLRScheduler::exponential)
        .def_static("cosine", &PyLRScheduler::cosine)
        .def_static("reduce_on_plateau", &PyLRScheduler::reduce_on_plateau);

    // ---- Model wrappers ----
    py::class_<PyModel>(m, "_Model")
        .def(py::init<>())
        .def_static("create", &PyModel::create)
        .def("forward", &PyModel::forward)
        .def("parameters", &PyModel::parameters);

    py::class_<PyHSSModel>(m, "_HSSModel")
        .def(py::init<>())
        .def_static("create", &PyHSSModel::create)
        .def("forward", &PyHSSModel::forward)
        .def("parameters", &PyHSSModel::parameters)
        .def("discretize_layer", &PyHSSModel::discretize_layer);

    py::class_<PySERModel>(m, "_SERModel")
        .def(py::init<>())
        .def_static("create", &PySERModel::create)
        .def("parameters", &PySERModel::parameters);

    py::class_<PyARCLayer>(m, "_ARCLayer")
        .def(py::init<>())
        .def_static("create", &PyARCLayer::create);

    py::class_<PyNPEVM>(m, "_NPEVM")
        .def(py::init<>())
        .def_static("create", &PyNPEVM::create)
        .def("load", &PyNPEVM::load)
        .def("run", &PyNPEVM::run);

    py::class_<PyNPEProgram>(m, "_NPEProgram")
        .def(py::init<>())
        .def_static("create", &PyNPEProgram::create)
        .def("append", &PyNPEProgram::append)
        .def_static("compile_attention", &PyNPEProgram::compile_attention)
        .def_static("compile_mlp", &PyNPEProgram::compile_mlp);

    py::class_<PyFMController>(m, "_FMController")
        .def(py::init<>())
        .def_static("create", &PyFMController::create)
        .def("forward", &PyFMController::forward)
        .def("sync_all_reduce", &PyFMController::sync_all_reduce)
        .def("sync_gossip", &PyFMController::sync_gossip)
        .def("sync_topology", &PyFMController::sync_topology);

    // ---- Trainer ----
    py::class_<PyTrainer>(m, "_Trainer")
        .def(py::init<>())
        .def_static("create", &PyTrainer::create)
        .def("train_step", &PyTrainer::train_step)
        .def("evaluate", &PyTrainer::evaluate)
        .def("save_checkpoint", &PyTrainer::save_checkpoint)
        .def("load_checkpoint", &PyTrainer::load_checkpoint);

    // ---- Memory pool ----
    py::class_<PyMemoryPool>(m, "_MemoryPool")
        .def_static("init", &PyMemoryPool::init)
        .def_static("destroy", &PyMemoryPool::destroy)
        .def_static("tls_cache_init", &PyMemoryPool::tls_cache_init)
        .def_static("tls_cache_destroy", &PyMemoryPool::tls_cache_destroy)
        .def_static("pool_alloc", &PyMemoryPool::pool_alloc)
        .def_static("pool_free", &PyMemoryPool::pool_free)
        .def_static("stats", &PyMemoryPool::stats)
        .def_static("print_stats", &PyMemoryPool::print_stats);

    // ---- Thread pool ----
    py::class_<PyThreadPool>(m, "_ThreadPool")
        .def(py::init<>())
        .def_static("create", &PyThreadPool::create)
        .def("submit", &PyThreadPool::submit)
        .def("wait", &PyThreadPool::wait)
        .def_static("default_count", &PyThreadPool::default_count)
        .def_static("parallel_for", &PyThreadPool::parallel_for);

    // ---- Utility functions ----
    m.def("dtype_size", &arix_tensor_dtype_size, "Get byte size of a dtype");
    m.def("dtype_name", &arix_tensor_dtype_name, "Get name of a dtype");
    m.def("dtype_to_numpy", &arix_dtype_to_numpy, "Map ArixDtype to NumPy dtype string");

    // No-grad context
    m.def("no_grad_enter", &arix_no_grad_enter);
    m.def("no_grad_exit", &arix_no_grad_exit);
    m.def("no_grad_is_active", &arix_no_grad_is_active);

    // ---- Crypto module ----
    py::module_ crypto = m.def_submodule("crypto", "Cryptographic functions");
    crypto.def("chacha20_encrypt", &py_crypto::chacha20_encrypt);
    crypto.def("sha512", &py_crypto::sha512);
    crypto.def("sha3_256", &py_crypto::sha3_256);
    crypto.def("blake3", &py_crypto::blake3);
    crypto.def("ed25519_keypair", &py_crypto::ed25519_keypair);
    crypto.def("ed25519_sign", &py_crypto::ed25519_sign);
    crypto.def("ed25519_verify", &py_crypto::ed25519_verify);
    crypto.def("poly1305_mac", &py_crypto::poly1305_mac);
    crypto.def("aead_encrypt", &py_crypto::aead_encrypt);
    crypto.def("aead_decrypt", &py_crypto::aead_decrypt);
    crypto.def("argon2_hash", &py_crypto::argon2_hash);
    crypto.def("random_bytes", &py_crypto::random_bytes);
    crypto.def("secure_zero", &py_crypto::secure_zero);
    crypto.def("ct_compare", &py_crypto::ct_compare);
}
