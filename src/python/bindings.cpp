#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif
#include "arix_arch.h"
#include "arix_train.h"
#ifdef __cplusplus
}
#endif

namespace py = pybind11;

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
        py::buffer_info buf = input.request();
        std::vector<size_t> shape(buf.shape.begin(), buf.shape.end());
        ArixTensor* t = arix_tensor_create(shape.data(), shape.size(), ARIX_FLOAT32);
        if (!t) return py::none();
        memcpy(t->data, buf.ptr, t->size * sizeof(float));

        ArixTensor* output = NULL;
        int ret = arix_model_forward(model, t, &output);
        arix_tensor_destroy(t);
        if (ret != 0 || !output) return py::none();

        std::vector<size_t> out_shape(output->shape, output->shape + output->ndim);
        py::array result(py::dtype("float32"), out_shape);
        memcpy(result.mutable_data(), output->data, output->size * sizeof(float));
        arix_tensor_destroy(output);
        return result;
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
        py::buffer_info ibuf = input.request();
        py::buffer_info tbuf = target.request();
        std::vector<size_t> ishape(ibuf.shape.begin(), ibuf.shape.end());
        std::vector<size_t> tshape(tbuf.shape.begin(), tbuf.shape.end());

        ArixTensor* inp = arix_tensor_create(ishape.data(), ishape.size(), ARIX_FLOAT32);
        ArixTensor* tgt = arix_tensor_create(tshape.data(), tshape.size(), ARIX_FLOAT32);
        if (!inp || !tgt) { arix_tensor_destroy(inp); arix_tensor_destroy(tgt); return -1.0f; }
        memcpy(inp->data, ibuf.ptr, inp->size * sizeof(float));
        memcpy(tgt->data, tbuf.ptr, tgt->size * sizeof(float));

        float loss = arix_trainer_train_step(trainer, inp, tgt);
        arix_tensor_destroy(inp);
        arix_tensor_destroy(tgt);
        return loss;
    }
    float evaluate(py::array_t<float> input, py::array_t<float> target) {
        if (!trainer) return -1.0f;
        py::buffer_info ibuf = input.request();
        py::buffer_info tbuf = target.request();
        std::vector<size_t> ishape(ibuf.shape.begin(), ibuf.shape.end());
        std::vector<size_t> tshape(tbuf.shape.begin(), tbuf.shape.end());

        ArixTensor* inp = arix_tensor_create(ishape.data(), ishape.size(), ARIX_FLOAT32);
        ArixTensor* tgt = arix_tensor_create(tshape.data(), tshape.size(), ARIX_FLOAT32);
        if (!inp || !tgt) { arix_tensor_destroy(inp); arix_tensor_destroy(tgt); return -1.0f; }
        memcpy(inp->data, ibuf.ptr, inp->size * sizeof(float));
        memcpy(tgt->data, tbuf.ptr, tgt->size * sizeof(float));

        float loss = arix_trainer_evaluate(trainer, inp, tgt);
        arix_tensor_destroy(inp);
        arix_tensor_destroy(tgt);
        return loss;
    }
    bool save(const std::string& path) { return trainer && arix_trainer_save_checkpoint(trainer, path.c_str()) == 0; }
    bool load(const std::string& path) { return trainer && arix_trainer_load_checkpoint(trainer, path.c_str()) == 0; }
};

PYBIND11_MODULE(arix_algo_core, m) {
    m.doc() = "ARIX-Algo core C++ bindings";

    py::class_<ArixArchConfig>(m, "_ArchConfig")
        .def(py::init<>())
        .def_readwrite("input_dim", &ArixArchConfig::input_dim)
        .def_readwrite("output_dim", &ArixArchConfig::output_dim);

    py::class_<PyModel>(m, "_Model")
        .def(py::init<const ArixArchConfig&>(), py::arg("config") = arix_arch_config_default())
        .def("forward", &PyModel::forward);

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
        .def("load", &PyTrainer::load);
}
