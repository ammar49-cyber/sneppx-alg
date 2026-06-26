#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif
#include "arix_arch.h"
#include "arix_train.h"
#include "arix_optimizer.h"
#include "arix_hss.h"
#ifdef __cplusplus
}
#endif

namespace py = pybind11;

static py::array tensor_to_numpy(const ArixTensor* t) {
    if (!t) return py::none();
    std::vector<size_t> shape(t->shape, t->shape + t->ndim);
    py::array result(py::dtype("float32"), shape);
    memcpy(result.mutable_data(), t->data, t->size * sizeof(float));
    return result;
}

static ArixTensor* numpy_to_tensor(py::array_t<float> arr) {
    py::buffer_info buf = arr.request();
    std::vector<size_t> shape(buf.shape.begin(), buf.shape.end());
    ArixTensor* t = arix_tensor_create(shape.data(), shape.size(), ARIX_FLOAT32);
    if (!t) return NULL;
    memcpy(t->data, buf.ptr, t->size * sizeof(float));
    return t;
}

class PyModel {
public:
    ArixModel* model;
    PyModel(const ArixArchConfig& config) {
        model = arix_model_create(&config);
    }
    ~PyModel() {
        if (model) arix_model_destroy(model);
    }
    py::array forward(py::array_t<float> input) {
        if (!model) return py::none();
        ArixTensor* t = numpy_to_tensor(input);
        if (!t) return py::none();

        ArixTensor* output = NULL;
        int ret = arix_model_forward(model, t, &output);
        arix_tensor_destroy(t);
        if (ret != 0 || !output) return py::none();

        py::array result = tensor_to_numpy(output);
        arix_tensor_destroy(output);
        return result;
    }

    std::vector<py::array> parameters() {
        std::vector<py::array> result;
        if (!model) return result;
        size_t n = arix_hss_get_params(model->hss_model, NULL, 0);
        if (n == 0) return result;
        std::vector<ArixTensor*> params(n);
        arix_hss_get_params(model->hss_model, params.data(), n);
        for (size_t i = 0; i < n; i++) {
            result.push_back(tensor_to_numpy(params[i]));
        }
        return result;
    }

    size_t num_params() {
        if (!model) return 0;
        return arix_hss_get_params(model->hss_model, NULL, 0);
    }
};

class PyTrainer {
public:
    ArixTrainer* trainer;
    PyTrainer(PyModel* py_model, const ArixTrainConfig& config) {
        if (!py_model || !py_model->model) throw std::runtime_error("Model is null");
        trainer = arix_trainer_create(py_model->model, &config);
    }
    ~PyTrainer() {
        if (trainer) arix_trainer_destroy(trainer);
    }
    float train_step(py::array_t<float> input, py::array_t<float> target) {
        if (!trainer) return -1.0f;
        ArixTensor* inp = numpy_to_tensor(input);
        ArixTensor* tgt = numpy_to_tensor(target);
        if (!inp || !tgt) { arix_tensor_destroy(inp); arix_tensor_destroy(tgt); return -1.0f; }
        float loss = arix_trainer_train_step(trainer, inp, tgt);
        arix_tensor_destroy(inp);
        arix_tensor_destroy(tgt);
        return loss;
    }
    float evaluate(py::array_t<float> input, py::array_t<float> target) {
        if (!trainer) return -1.0f;
        ArixTensor* inp = numpy_to_tensor(input);
        ArixTensor* tgt = numpy_to_tensor(target);
        if (!inp || !tgt) { arix_tensor_destroy(inp); arix_tensor_destroy(tgt); return -1.0f; }
        float loss = arix_trainer_evaluate(trainer, inp, tgt);
        arix_tensor_destroy(inp);
        arix_tensor_destroy(tgt);
        return loss;
    }
    bool save(const std::string& path) { return trainer && arix_trainer_save_checkpoint(trainer, path.c_str()) == 0; }
    bool load(const std::string& path) { return trainer && arix_trainer_load_checkpoint(trainer, path.c_str()) == 0; }
    float get_lr() { return trainer ? trainer->optimizer->learning_rate : 0.0f; }
};

class PyOptimizer {
public:
    ArixOptimizer* opt;
    PyOptimizer(const ArixOptimizerConfig& config) {
        opt = arix_optimizer_create(&config);
    }
    ~PyOptimizer() {
        if (opt) arix_optimizer_destroy(opt);
    }
    void step(std::vector<py::array> params, std::vector<py::array> grads) {
        if (!opt) return;
        size_t n = params.size();
        if (n != grads.size()) return;
        std::vector<ArixTensor*> pt(n);
        std::vector<ArixTensor*> gt(n);
        for (size_t i = 0; i < n; i++) {
            pt[i] = numpy_to_tensor(params[i]);
            gt[i] = numpy_to_tensor(grads[i]);
        }
        arix_optimizer_step(opt, pt.data(), gt.data(), n);
        for (size_t i = 0; i < n; i++) {
            memcpy(params[i].mutable_data(), pt[i]->data, pt[i]->size * sizeof(float));
            arix_tensor_destroy(pt[i]);
            arix_tensor_destroy(gt[i]);
        }
    }
};

PYBIND11_MODULE(arix_algo_core, m) {
    m.doc() = "ARIX-Algo core C++ bindings";

    py::class_<ArixHSSConfig>(m, "_HSSConfig")
        .def(py::init<>())
        .def_readwrite("state_dim", &ArixHSSConfig::state_dim)
        .def_readwrite("input_dim", &ArixHSSConfig::input_dim)
        .def_readwrite("output_dim", &ArixHSSConfig::output_dim)
        .def_readwrite("num_layers", &ArixHSSConfig::num_layers)
        .def_readwrite("seq_len", &ArixHSSConfig::seq_len)
        .def_readwrite("dt_min", &ArixHSSConfig::dt_min)
        .def_readwrite("dt_max", &ArixHSSConfig::dt_max)
        .def_readwrite("use_hierarchical", &ArixHSSConfig::use_hierarchical);

    py::class_<ArixArchConfig>(m, "_ArchConfig")
        .def(py::init<>())
        .def_readwrite("input_dim", &ArixArchConfig::input_dim)
        .def_readwrite("output_dim", &ArixArchConfig::output_dim)
        .def_readwrite("seed", &ArixArchConfig::seed)
        .def_property("hss_state_dim",
            [](const ArixArchConfig& c) { return c.hss_config.state_dim; },
            [](ArixArchConfig& c, size_t v) { c.hss_config.state_dim = v; })
        .def_property("hss_num_layers",
            [](const ArixArchConfig& c) { return c.hss_config.num_layers; },
            [](ArixArchConfig& c, size_t v) { c.hss_config.num_layers = v; })
        .def_property("hss_input_dim",
            [](const ArixArchConfig& c) { return c.hss_config.input_dim; },
            [](ArixArchConfig& c, size_t v) { c.hss_config.input_dim = v; })
        .def_property("hss_output_dim",
            [](const ArixArchConfig& c) { return c.hss_config.output_dim; },
            [](ArixArchConfig& c, size_t v) { c.hss_config.output_dim = v; });

    py::enum_<ArixOptimizerType>(m, "_OptimizerType")
        .value("SGD", ARIX_OPTIMIZER_SGD)
        .value("ADAM", ARIX_OPTIMIZER_ADAM)
        .value("ADAMW", ARIX_OPTIMIZER_ADAMW)
        .value("ADAMAX", ARIX_OPTIMIZER_ADAMAX)
        .value("RMSPROP", ARIX_OPTIMIZER_RMSPROP)
        .value("ADAGRAD", ARIX_OPTIMIZER_ADAGRAD)
        .value("ADADELTA", ARIX_OPTIMIZER_ADADELTA)
        .export_values();

    py::enum_<ArixLRSchedulerType>(m, "_LRSchedulerType")
        .value("STEP", ARIX_LR_STEP)
        .value("EXPONENTIAL", ARIX_LR_EXPONENTIAL)
        .value("COSINE", ARIX_LR_COSINE)
        .value("REDUCE_ON_PLATEAU", ARIX_LR_REDUCE_ON_PLATEAU)
        .export_values();

    py::class_<ArixOptimizerConfig>(m, "_OptimizerConfig")
        .def(py::init<>())
        .def_readwrite("learning_rate", &ArixOptimizerConfig::learning_rate)
        .def_readwrite("type", &ArixOptimizerConfig::type)
        .def_readwrite("weight_decay", &ArixOptimizerConfig::weight_decay)
        .def_readwrite("momentum", &ArixOptimizerConfig::momentum)
        .def_readwrite("beta1", &ArixOptimizerConfig::beta1)
        .def_readwrite("beta2", &ArixOptimizerConfig::beta2)
        .def_readwrite("epsilon", &ArixOptimizerConfig::epsilon)
        .def_readwrite("dampening", &ArixOptimizerConfig::dampening)
        .def_readwrite("nesterov", &ArixOptimizerConfig::nesterov)
        .def_readwrite("rho", &ArixOptimizerConfig::rho)
        .def_readwrite("grad_clip", &ArixOptimizerConfig::grad_clip);

    py::class_<PyOptimizer>(m, "_Optimizer")
        .def(py::init<const ArixOptimizerConfig&>())
        .def("step", &PyOptimizer::step);

    py::class_<PyModel>(m, "_Model")
        .def(py::init<const ArixArchConfig&>(), py::arg("config") = arix_arch_config_default())
        .def("forward", &PyModel::forward)
        .def("parameters", &PyModel::parameters)
        .def("num_params", &PyModel::num_params);

    py::class_<ArixTrainConfig>(m, "_TrainConfig")
        .def(py::init<>())
        .def_readwrite("learning_rate", &ArixTrainConfig::learning_rate)
        .def_readwrite("num_epochs", &ArixTrainConfig::num_epochs)
        .def_readwrite("batch_size", &ArixTrainConfig::batch_size)
        .def_readwrite("log_interval", &ArixTrainConfig::log_interval)
        .def_readwrite("save_interval", &ArixTrainConfig::save_interval);

    py::class_<PyTrainer>(m, "_Trainer")
        .def(py::init<PyModel*, const ArixTrainConfig&>(), py::arg("model"), py::arg("config") = arix_train_config_default())
        .def("train_step", &PyTrainer::train_step)
        .def("evaluate", &PyTrainer::evaluate)
        .def("save", &PyTrainer::save)
        .def("load", &PyTrainer::load)
        .def_property_readonly("learning_rate", &PyTrainer::get_lr);

    m.def("_hss_config_default", []() { return arix_hss_config_default(); });
    m.def("_optimizer_config_default", []() { return arix_optimizer_config_default(); });
    m.def("_arch_config_default", []() { return arix_arch_config_default(); });
}
