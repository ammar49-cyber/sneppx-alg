import os
from typing import List, Optional
from . import _neural_engine_bridge
from .tensor import Tensor, _to_shape


class ModelConfig:
    def __init__(self):
        self._c = _neural_engine_bridge.SNEPPXArchConfig()
        self._c.default()

    @property
    def input_dim(self):
        return self._c.input_dim

    @input_dim.setter
    def input_dim(self, v):
        self._c.input_dim = v

    @property
    def output_dim(self):
        return self._c.output_dim

    @output_dim.setter
    def output_dim(self, v):
        self._c.output_dim = v

    @property
    def seed(self):
        return self._c.seed

    @seed.setter
    def seed(self, v):
        self._c.seed = v


class Model:
    def __init__(self, config: Optional[ModelConfig] = None):
        if config is None:
            config = ModelConfig()
        self._m = _neural_engine_bridge._Model.create(config._c)
        self._config = config

    def forward(self, input: Tensor) -> Tensor:
        return Tensor._from_ptr(self._m.forward(input._t))

    def parameters(self) -> List[Tensor]:
        return [Tensor._from_ptr(p) for p in self._m.parameters()]

    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)

    def save_checkpoint(self, path: str):
        import pickle
        params = self.parameters()
        data = {
            'param_data': [p.data.copy() for p in params],
            'config': {
                'input_dim': self._config.input_dim,
                'output_dim': self._config.output_dim,
                'seed': self._config.seed,
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_checkpoint(self, path: str):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        params = self.parameters()
        for p, d in zip(params, data['param_data']):
            p.data = d


# ---- HSS Model ----
class HSSConfig:
    def __init__(self):
        self._c = _neural_engine_bridge.SNEPPXHSSConfig()
        self._c.default()

    @property
    def state_dim(self):
        return self._c.state_dim
    @state_dim.setter
    def state_dim(self, v):
        self._c.state_dim = v

    @property
    def input_dim(self):
        return self._c.input_dim
    @input_dim.setter
    def input_dim(self, v):
        self._c.input_dim = v

    @property
    def output_dim(self):
        return self._c.output_dim
    @output_dim.setter
    def output_dim(self, v):
        self._c.output_dim = v

    @property
    def num_layers(self):
        return self._c.num_layers
    @num_layers.setter
    def num_layers(self, v):
        self._c.num_layers = v

    @property
    def seq_len(self):
        return self._c.seq_len
    @seq_len.setter
    def seq_len(self, v):
        self._c.seq_len = v

    @property
    def dt_min(self):
        return self._c.dt_min
    @dt_min.setter
    def dt_min(self, v):
        self._c.dt_min = v

    @property
    def dt_max(self):
        return self._c.dt_max
    @dt_max.setter
    def dt_max(self, v):
        self._c.dt_max = v

    @property
    def use_hierarchical(self):
        return bool(self._c.use_hierarchical)
    @use_hierarchical.setter
    def use_hierarchical(self, v):
        self._c.use_hierarchical = int(v)


class HSSModel:
    def __init__(self, config: HSSConfig, seed: int = 42):
        self._m = _neural_engine_bridge._HSSModel.create(config._c, seed)
        self._config = config

    def forward(self, input: Tensor) -> Tensor:
        return Tensor._from_ptr(self._m.forward(input._t))

    def parameters(self) -> List[Tensor]:
        return [Tensor._from_ptr(p) for p in self._m.parameters()]

    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)

    def discretize_layer(self, layer_idx: int = 0):
        self._m.discretize_layer(layer_idx)


# ---- SER Model ----
class SERConfig:
    def __init__(self):
        self._c = _neural_engine_bridge.SNEPPXSERConfig()
        self._c.default()

    @property
    def num_experts(self):
        return self._c.num_experts
    @num_experts.setter
    def num_experts(self, v):
        self._c.num_experts = v

    @property
    def num_active(self):
        return self._c.num_active
    @num_active.setter
    def num_active(self, v):
        self._c.num_active = v

    @property
    def input_dim(self):
        return self._c.input_dim
    @input_dim.setter
    def input_dim(self, v):
        self._c.input_dim = v

    @property
    def expert_dim(self):
        return self._c.expert_dim
    @expert_dim.setter
    def expert_dim(self, v):
        self._c.expert_dim = v

    @property
    def output_dim(self):
        return self._c.output_dim
    @output_dim.setter
    def output_dim(self, v):
        self._c.output_dim = v

    @property
    def top_k_method(self):
        return self._c.top_k_method
    @top_k_method.setter
    def top_k_method(self, v):
        self._c.top_k_method = v

    @property
    def load_balance_coef(self):
        return self._c.load_balance_coef
    @load_balance_coef.setter
    def load_balance_coef(self, v):
        self._c.load_balance_coef = v

    @property
    def dropout_rate(self):
        return self._c.dropout_rate
    @dropout_rate.setter
    def dropout_rate(self, v):
        self._c.dropout_rate = v


class SERModel:
    def __init__(self, config: SERConfig, seed: int = 42, num_layers: int = 1):
        self._m = _neural_engine_bridge._SERModel.create(config._c, seed, num_layers)
        self._config = config

    def parameters(self) -> List[Tensor]:
        return [Tensor._from_ptr(p) for p in self._m.parameters()]

    def forward(self, input: Tensor) -> Tensor:
        return Tensor._from_ptr(self._m.forward(input._t))

    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)


# ---- ARC Model ----
class ARCConfig:
    def __init__(self):
        self._c = _neural_engine_bridge.SNEPPXARCConfig()
        self._c.default()

    @property
    def input_guard_strength(self):
        return self._c.input_guard_strength
    @input_guard_strength.setter
    def input_guard_strength(self, v):
        self._c.input_guard_strength = v

    @property
    def gradient_obfuscation_method(self):
        return self._c.gradient_obfuscation_method
    @gradient_obfuscation_method.setter
    def gradient_obfuscation_method(self, v):
        self._c.gradient_obfuscation_method = v

    @property
    def gradient_noise_scale(self):
        return self._c.gradient_noise_scale
    @gradient_noise_scale.setter
    def gradient_noise_scale(self, v):
        self._c.gradient_noise_scale = v

    @property
    def gradient_clip_max(self):
        return self._c.gradient_clip_max
    @gradient_clip_max.setter
    def gradient_clip_max(self, v):
        self._c.gradient_clip_max = v

    @property
    def output_verify_layers(self):
        return self._c.output_verify_layers
    @output_verify_layers.setter
    def output_verify_layers(self, v):
        self._c.output_verify_layers = v

    @property
    def output_verify_threshold(self):
        return self._c.output_verify_threshold
    @output_verify_threshold.setter
    def output_verify_threshold(self, v):
        self._c.output_verify_threshold = v

    @property
    def adversarial_training(self):
        return bool(self._c.adversarial_training)
    @adversarial_training.setter
    def adversarial_training(self, v):
        self._c.adversarial_training = int(v)

    @property
    def attack_simulation_types(self):
        return self._c.attack_simulation_types
    @attack_simulation_types.setter
    def attack_simulation_types(self, v):
        self._c.attack_simulation_types = v


class ARCModel:
    def __init__(self, config: ARCConfig, input_dim: int, output_dim: int, seed: int = 42):
        self._m = _neural_engine_bridge._ARCLayer.create(config._c, input_dim, output_dim, seed)
        self._config = config

    def forward(self, input: Tensor) -> Tensor:
        from . import _neural_engine_bridge as c
        out_ptr = c._Tensor.create((1, 1), c.SNEPPXDtype.FLOAT32)
        metrics = [0.0, 0.0, 0.0, 0.0]
        self._m.forward(input._t, out_ptr, metrics)
        return Tensor._from_ptr(out_ptr)

    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)


# ---- NPE Model ----
class NPEConfig:
    def __init__(self):
        self._c = _neural_engine_bridge.SNEPPXNPEConfig()
        self._c.default()

    @property
    def max_program_length(self):
        return self._c.max_program_length
    @max_program_length.setter
    def max_program_length(self, v):
        self._c.max_program_length = v

    @property
    def register_count(self):
        return self._c.register_count
    @register_count.setter
    def register_count(self, v):
        self._c.register_count = v

    @property
    def step_limit(self):
        return self._c.step_limit
    @step_limit.setter
    def step_limit(self, v):
        self._c.step_limit = v

    @property
    def verification_mode(self):
        return bool(self._c.verification_mode)
    @verification_mode.setter
    def verification_mode(self, v):
        self._c.verification_mode = int(v)

    @property
    def trace_execution(self):
        return bool(self._c.trace_execution)
    @trace_execution.setter
    def trace_execution(self, v):
        self._c.trace_execution = int(v)


class NPEModel:
    def __init__(self, config: NPEConfig):
        self._vm = _neural_engine_bridge._NPEVM.create(config._c)
        self._config = config

    def load_program(self, program):
        self._vm.load(self._vm, program._prog if hasattr(program, '_prog') else program)

    def run(self, input: Tensor) -> Tensor:
        return Tensor._from_ptr(self._vm.run(input._t))

    def forward(self, input: Tensor) -> Tensor:
        return self.run(input)

    def __call__(self, input: Tensor) -> Tensor:
        return self.run(input)


# ---- FM Model ----
class FMConfig:
    def __init__(self):
        self._c = _neural_engine_bridge.SNEPPXFMConfig()
        self._c.default()

    @property
    def num_nodes(self):
        return self._c.num_nodes
    @num_nodes.setter
    def num_nodes(self, v):
        self._c.num_nodes = v

    @property
    def memory_dim(self):
        return self._c.memory_dim
    @memory_dim.setter
    def memory_dim(self, v):
        self._c.memory_dim = v

    @property
    def memory_capacity(self):
        return self._c.memory_capacity
    @memory_capacity.setter
    def memory_capacity(self, v):
        self._c.memory_capacity = v

    @property
    def sync_interval(self):
        return self._c.sync_interval
    @sync_interval.setter
    def sync_interval(self, v):
        self._c.sync_interval = v

    @property
    def sync_method(self):
        return self._c.sync_method
    @sync_method.setter
    def sync_method(self, v):
        self._c.sync_method = v

    @property
    def compression_ratio(self):
        return self._c.compression_ratio
    @compression_ratio.setter
    def compression_ratio(self, v):
        self._c.compression_ratio = v

    @property
    def privacy_epsilon(self):
        return self._c.privacy_epsilon
    @privacy_epsilon.setter
    def privacy_epsilon(self, v):
        self._c.privacy_epsilon = v


class FMModel:
    def __init__(self, config: FMConfig):
        self._ctrl = _neural_engine_bridge._FMController.create(config._c)
        self._config = config

    def forward(self, node_id: int, input: Tensor) -> Tensor:
        return Tensor._from_ptr(self._ctrl.forward(node_id, input._t))

    def sync_all_reduce(self) -> int:
        return self._ctrl.sync_all_reduce()

    def sync_gossip(self, num_pairs: int) -> int:
        return self._ctrl.sync_gossip(num_pairs)

    def sync_topology(self) -> int:
        return self._ctrl.sync_topology()
