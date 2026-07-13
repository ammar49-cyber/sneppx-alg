"""Graph Compiler - Fusion patterns, constant folding, dead code elimination."""

from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from .tensor import Tensor


class OpType(Enum):
    """Supported operation types for fusion."""

    MATMUL = "matmul"
    ADD = "add"
    MUL = "mul"
    RELU = "relu"
    GELU = "gelu"
    SIGMOID = "sigmoid"
    SILU = "silu"
    TANH = "tanh"
    SOFTMAX = "softmax"
    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"
    DROPOUT = "dropout"
    BIAS_ADD = "bias_add"
    CONV2D = "conv2d"
    TRANSPOSE = "transpose"
    RESHAPE = "reshape"
    SLICE = "slice"
    CONCAT = "concat"
    SPLIT = "split"
    REDUCE_SUM = "reduce_sum"
    REDUCE_MEAN = "reduce_mean"


@dataclass
class GraphNode:
    """Node in the computation graph."""

    op_type: OpType
    inputs: List[str]  # input tensor names
    outputs: List[str]  # output tensor names
    attributes: Dict[str, Any] = field(default_factory=dict)
    name: str = ""
    constant_inputs: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Graph:
    """Computation graph."""

    nodes: List[GraphNode] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    constants: Dict[str, np.ndarray] = field(default_factory=dict)
    shapes: Dict[str, List[int]] = field(default_factory=dict)
    dtypes: Dict[str, str] = field(default_factory=dict)


class FusionPattern:
    """Represents a fusion pattern."""

    def __init__(
        self,
        name: str,
        pattern: List[OpType],
        replacement: OpType,
        condition: Optional[Callable[[Graph, int], bool]] = None,
        transform: Optional[Callable[[Graph, int], List[GraphNode]]] = None,
    ):
        self.name = name
        self.pattern = pattern
        self.replacement = replacement
        self.condition = condition or (lambda g, i: True)
        self.transform = transform


class GraphCompiler:
    """Graph compiler with fusion, constant folding, and DCE."""

    def __init__(self):
        self.fusion_patterns: List[FusionPattern] = []
        self._register_default_patterns()

    def _register_default_patterns(self):
        """Register default fusion patterns."""

        # MatMul + Bias + ReLU -> FusedMatMulBiasReLU
        self.add_pattern(
            FusionPattern(
                name="matmul_bias_relu",
                pattern=[OpType.MATMUL, OpType.ADD, OpType.RELU],
                replacement=OpType.MATMUL,  # Fused kernel handles all
                condition=lambda g, i: self._check_fusion_bias_relu(g, i),
                transform=self._transform_matmul_bias_relu,
            )
        )

        # MatMul + Bias + GELU -> FusedMatMulBiasGELU
        self.add_pattern(
            FusionPattern(
                name="matmul_bias_gelu",
                pattern=[OpType.MATMUL, OpType.ADD, OpType.GELU],
                replacement=OpType.MATMUL,
                condition=lambda g, i: self._check_fusion_bias_act(g, i, OpType.GELU),
                transform=self._transform_matmul_bias_act,
            )
        )

        # MatMul + Bias + SiLU -> FusedMatMulBiasSiLU
        self.add_pattern(
            FusionPattern(
                name="matmul_bias_silu",
                pattern=[OpType.MATMUL, OpType.ADD, OpType.SILU],
                replacement=OpType.MATMUL,
                condition=lambda g, i: self._check_fusion_bias_act(g, i, OpType.SILU),
                transform=self._transform_matmul_bias_act,
            )
        )

        # Conv2D + Bias + ReLU -> FusedConvBiasReLU
        self.add_pattern(
            FusionPattern(
                name="conv_bias_relu",
                pattern=[OpType.CONV2D, OpType.ADD, OpType.RELU],
                replacement=OpType.CONV2D,
                condition=lambda g, i: self._check_fusion_bias_act(g, i, OpType.RELU),
                transform=self._transform_conv_bias_act,
            )
        )

        # LayerNorm + Linear -> FusedLayerNormLinear
        self.add_pattern(
            FusionPattern(
                name="layernorm_linear",
                pattern=[OpType.LAYERNORM, OpType.MATMUL],
                replacement=OpType.LAYERNORM,
                condition=lambda g, i: self._check_fusion_ln_linear(g, i),
                transform=self._transform_ln_linear,
            )
        )

        # Add + Mul (scale) -> FusedAddMul
        self.add_pattern(
            FusionPattern(
                name="add_mul",
                pattern=[OpType.ADD, OpType.MUL],
                replacement=OpType.ADD,
                condition=lambda g, i: self._check_fusion_add_mul(g, i),
                transform=self._transform_add_mul,
            )
        )

        # ReduceMean + Mul (variance) -> FusedReduceMeanMul
        self.add_pattern(
            FusionPattern(
                name="reduce_mean_mul",
                pattern=[OpType.REDUCE_MEAN, OpType.MUL],
                replacement=OpType.REDUCE_MEAN,
                condition=lambda g, i: self._check_fusion_reduce_mean_mul(g, i),
                transform=self._transform_reduce_mean_mul,
            )
        )

    def add_pattern(self, pattern: FusionPattern):
        """Add a fusion pattern."""
        self.fusion_patterns.append(pattern)

    def compile(self, graph: Graph) -> Graph:
        """Compile the graph with all optimizations."""
        # Constant folding
        graph = self._constant_folding(graph)

        # Dead code elimination
        graph = self._dead_code_elimination(graph)

        # Fusion
        graph = self._fusion(graph)

        # Shape/dtype inference
        graph = self._infer_shapes_dtypes(graph)

        return graph

    def _constant_folding(self, graph: Graph) -> Graph:
        """Fold constant expressions."""
        changed = True
        while changed:
            changed = False
            new_nodes = []

            for node in graph.nodes:
                if self._is_constant_node(node, graph):
                    # Evaluate and replace with constant
                    const_val = self._evaluate_node(node, graph)
                    if const_val is not None:
                        graph.constants[node.outputs[0]] = const_val
                        graph.shapes[node.outputs[0]] = list(const_val.shape)
                        graph.dtypes[node.outputs[0]] = str(const_val.dtype)
                        changed = True
                    else:
                        new_nodes.append(node)
                else:
                    new_nodes.append(node)

            if changed:
                graph.nodes = new_nodes

        return graph

    def _is_constant_node(self, node: GraphNode, graph: Graph) -> bool:
        """Check if node can be evaluated at compile time."""
        # All inputs must be constants
        for inp in node.inputs:
            if inp not in graph.constants:
                return False
        # Operation must be foldable
        return node.op_type in [
            OpType.ADD,
            OpType.MUL,
            OpType.RELU,
            OpType.GELU,
            OpType.SIGMOID,
            OpType.TANH,
            OpType.SIGMOID,
            OpType.TRANSPOSE,
            OpType.RESHAPE,
            OpType.SLICE,
            OpType.CONCAT,
            OpType.SPLIT,
            OpType.REDUCE_SUM,
            OpType.REDUCE_MEAN,
        ]

    def _evaluate_node(self, node: GraphNode, graph: Graph) -> Optional[np.ndarray]:
        """Evaluate a constant node."""
        inputs = [graph.constants[inp] for inp in node.inputs]

        if node.op_type == OpType.ADD:
            return inputs[0] + inputs[1] if len(inputs) >= 2 else inputs[0]
        elif node.op_type == OpType.MUL:
            return inputs[0] * inputs[1] if len(inputs) >= 2 else inputs[0]
        elif node.op_type == OpType.RELU:
            return np.maximum(inputs[0], 0)
        elif node.op_type == OpType.GELU:
            x = inputs[0]
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        elif node.op_type == OpType.SIGMOID:
            return 1 / (1 + np.exp(-inputs[0]))
        elif node.op_type == OpType.TANH:
            return np.tanh(inputs[0])
        elif node.op_type == OpType.TRANSPOSE:
            perm = node.attributes.get("perm", list(reversed(range(inputs[0].ndim))))
            return inputs[0].transpose(perm)
        elif node.op_type == OpType.RESHAPE:
            shape = node.attributes.get("shape", list(inputs[0].shape))
            return inputs[0].reshape(shape)
        elif node.op_type == OpType.CONCAT:
            axis = node.attributes.get("axis", 0)
            return np.concatenate(inputs, axis=axis)
        elif node.op_type == OpType.REDUCE_SUM:
            axes = node.attributes.get("axes", None)
            keepdims = node.attributes.get("keepdims", 0)
            return np.sum(inputs[0], axis=axes, keepdims=bool(keepdims))
        elif node.op_type == OpType.REDUCE_MEAN:
            axes = node.attributes.get("axes", None)
            keepdims = node.attributes.get("keepdims", 0)
            return np.mean(inputs[0], axis=axes, keepdims=bool(keepdims))

        return None

    def _dead_code_elimination(self, graph: Graph) -> Graph:
        """Remove unused nodes."""
        # Find all reachable outputs
        needed = set(graph.outputs)

        # Work backwards from outputs
        changed = True
        while changed:
            changed = False
            for node in reversed(graph.nodes):
                if any(out in needed for out in node.outputs):
                    for inp in node.inputs:
                        if inp not in needed:
                            needed.add(inp)
                            changed = True

        # Keep only needed nodes
        new_nodes = [n for n in graph.nodes if any(o in needed for o in n.outputs)]
        graph.nodes = new_nodes
        return graph

    def _fusion(self, graph: Graph) -> Graph:
        """Apply fusion patterns."""
        i = 0
        while i < len(graph.nodes):
            matched = False
            for pattern in self.fusion_patterns:
                if self._match_pattern(graph, i, pattern):
                    if pattern.condition(graph, i):
                        new_nodes = pattern.transform(graph, i)
                        if new_nodes:
                            # Replace matched nodes with fused node
                            pattern_len = len(pattern.pattern)
                            graph.nodes = (
                                graph.nodes[:i]
                                + new_nodes
                                + graph.nodes[i + pattern_len :]
                            )
                            matched = True
                            break
            if not matched:
                i += 1
        return graph

    def _match_pattern(
        self, graph: Graph, start_idx: int, pattern: FusionPattern
    ) -> bool:
        """Check if pattern matches at start_idx."""
        if start_idx + len(pattern.pattern) > len(graph.nodes):
            return False

        for j, op_type in enumerate(pattern.pattern):
            if graph.nodes[start_idx + j].op_type != op_type:
                return False
            # Check data flow connectivity
            if j > 0:
                prev_outputs = graph.nodes[start_idx + j - 1].outputs
                curr_inputs = graph.nodes[start_idx + j].inputs
                if not any(out in curr_inputs for out in prev_outputs):
                    return False
        return True

    def _infer_shapes_dtypes(self, graph: Graph) -> Graph:
        """Infer shapes and dtypes for all tensors."""
        # This would propagate shapes through the graph
        return graph

    # ==================== Pattern Condition Checks ====================

    def _check_fusion_bias_relu(self, graph: Graph, idx: int) -> bool:
        """Check if matmul+bias+relu can be fused."""
        if idx + 2 >= len(graph.nodes):
            return False
        add_node = graph.nodes[idx + 1]
        relu_node = graph.nodes[idx + 2]
        # Check if ADD is bias addition (one input is 1D constant)
        return self._is_bias_add(add_node, graph)

    def _check_fusion_bias_act(self, graph: Graph, idx: int, act_type: OpType) -> bool:
        """Check if matmul+bias+activation can be fused."""
        if idx + 2 >= len(graph.nodes):
            return False
        add_node = graph.nodes[idx + 1]
        act_node = graph.nodes[idx + 2]
        return act_node.op_type == act_type and self._is_bias_add(add_node, graph)

    def _is_bias_add(self, node: GraphNode, graph: Graph) -> bool:
        """Check if ADD is a bias addition (one input is 1D constant)."""
        if node.op_type != OpType.ADD or len(node.inputs) != 2:
            return False
        for inp in node.inputs:
            if inp in graph.constants:
                shape = graph.shapes.get(inp, [])
                if len(shape) == 1:
                    return True
        return False

    def _check_fusion_bias_act_conv(
        self, graph: Graph, idx: int, act_type: OpType
    ) -> bool:
        """Check if conv+bias+activation can be fused."""
        if idx + 2 >= len(graph.nodes):
            return False
        add_node = graph.nodes[idx + 1]
        act_node = graph.nodes[idx + 2]
        return act_node.op_type == act_type and self._is_bias_add(add_node, graph)

    def _check_fusion_ln_linear(self, graph: Graph, idx: int) -> bool:
        """Check if LayerNorm + Linear can be fused."""
        if idx + 1 >= len(graph.nodes):
            return False
        ln_node = graph.nodes[idx]
        matmul_node = graph.nodes[idx + 1]
        return (
            ln_node.op_type == OpType.LAYERNORM and matmul_node.op_type == OpType.MATMUL
        )

    def _check_fusion_add_mul(self, graph: Graph, idx: int) -> bool:
        """Check if Add + Mul can be fused (scale after bias)."""
        if idx + 1 >= len(graph.nodes):
            return False
        add_node = graph.nodes[idx]
        mul_node = graph.nodes[idx + 1]
        return (
            add_node.op_type == OpType.ADD
            and mul_node.op_type == OpType.MUL
            and self._is_bias_add(add_node, graph)
            and self._is_scale_mul(mul_node, graph)
        )

    def _is_scale_mul(self, node: GraphNode, graph: Graph) -> bool:
        """Check if MUL is scaling (one input is constant scalar or 1D)."""
        if node.op_type != OpType.MUL or len(node.inputs) != 2:
            return False
        for inp in node.inputs:
            if inp in graph.constants:
                shape = graph.shapes.get(inp, [])
                if len(shape) <= 1:
                    return True
        return False

    def _check_fusion_reduce_mean_mul(self, graph: Graph, idx: int) -> bool:
        """Check if ReduceMean + Mul can be fused."""
        if idx + 1 >= len(graph.nodes):
            return False
        return graph.nodes[idx + 1].op_type == OpType.MUL

    # ==================== Pattern Transforms ====================

    def _transform_matmul_bias_relu(self, graph: Graph, idx: int) -> List[GraphNode]:
        """Transform MatMul + Bias + ReLU -> FusedMatMulBiasReLU."""
        matmul_node = graph.nodes[idx]
        add_node = graph.nodes[idx + 1]
        relu_node = graph.nodes[idx + 2]

        # Find bias input
        bias_input = None
        for inp in add_node.inputs:
            if inp in graph.constants and len(graph.shapes.get(inp, [])) == 1:
                bias_input = inp
                break

        fused = GraphNode(
            op_type=OpType.MATMUL,
            inputs=matmul_node.inputs + ([bias_input] if bias_input else []),
            outputs=relu_node.outputs,
            attributes={"fused_bias": True, "fused_activation": "relu"},
            name=f"fused_matmul_bias_relu_{idx}",
        )
        return [fused]

    def _transform_matmul_bias_act(self, graph: Graph, idx: int) -> List[GraphNode]:
        """Transform MatMul + Bias + Activation -> FusedMatMulBiasAct."""
        matmul_node = graph.nodes[idx]
        add_node = graph.nodes[idx + 1]
        act_node = graph.nodes[idx + 2]

        bias_input = None
        for inp in add_node.inputs:
            if inp in graph.constants and len(graph.shapes.get(inp, [])) == 1:
                bias_input = inp
                break

        act_name = act_node.op_type.name.lower()

        fused = GraphNode(
            op_type=OpType.MATMUL,
            inputs=matmul_node.inputs + ([bias_input] if bias_input else []),
            outputs=act_node.outputs,
            attributes={"fused_bias": True, "fused_activation": act_name},
            name=f"fused_matmul_bias_{act_node.op_type.name.lower()}_{idx}",
        )
        return [fused]

    def _transform_conv_bias_act(self, graph: Graph, idx: int) -> List[GraphNode]:
        """Transform Conv2D + Bias + Activation -> FusedConvBiasAct."""
        conv_node = graph.nodes[idx]
        add_node = graph.nodes[idx + 1]
        act_node = graph.nodes[idx + 2]

        bias_input = None
        for inp in add_node.inputs:
            if inp in graph.constants and len(graph.shapes.get(inp, [])) == 1:
                bias_input = inp
                break

        act_name = act_node.op_type.name.lower()

        fused = GraphNode(
            op_type=OpType.CONV2D,
            inputs=conv_node.inputs + ([bias_input] if bias_input else []),
            outputs=act_node.outputs,
            attributes={
                **conv_node.attributes,
                "fused_bias": True,
                "fused_activation": act_name,
            },
            name=f"fused_conv_bias_{act_name}_{idx}",
        )
        return [fused]

    def _transform_ln_linear(self, graph: Graph, idx: int) -> List[GraphNode]:
        """Transform LayerNorm + Linear -> FusedLayerNormLinear."""
        ln_node = graph.nodes[idx]
        matmul_node = graph.nodes[idx + 1]

        fused = GraphNode(
            op_type=OpType.LAYERNORM,
            inputs=ln_node.inputs + matmul_node.inputs[1:],  # ln_input + weight + bias
            outputs=matmul_node.outputs,
            attributes={**ln_node.attributes, "fused_linear": True},
            name=f"fused_ln_linear_{idx}",
        )
        return [fused]

    def _transform_add_mul(self, graph: Graph, idx: int) -> List[GraphNode]:
        """Transform Add + Mul -> FusedAddMul."""
        add_node = graph.nodes[idx]
        mul_node = graph.nodes[idx + 1]

        fused = GraphNode(
            op_type=OpType.ADD,
            inputs=add_node.inputs
            + [inp for inp in mul_node.inputs if inp not in add_node.inputs],
            outputs=mul_node.outputs,
            attributes={"fused_scale": True},
            name=f"fused_add_mul_{idx}",
        )
        return [fused]

    def _transform_reduce_mean_mul(self, graph: Graph, idx: int) -> List[GraphNode]:
        """Transform ReduceMean + Mul -> FusedReduceMeanMul."""
        rm_node = graph.nodes[idx]
        mul_node = graph.nodes[idx + 1]

        fused = GraphNode(
            op_type=OpType.REDUCE_MEAN,
            inputs=rm_node.inputs
            + [inp for inp in mul_node.inputs if inp not in rm_node.inputs],
            outputs=mul_node.outputs,
            attributes={**rm_node.attributes, "fused_scale": True},
            name=f"fused_reduce_mean_mul_{idx}",
        )
        return [fused]


# Convenience function
def compile_graph(graph: Graph) -> Graph:
    """Compile a graph with all optimizations."""
    compiler = GraphCompiler()
    return compiler.compile(graph)


# Example usage
def create_example_graph() -> Graph:
    """Create an example computation graph."""
    graph = Graph()

    # Input
    graph.inputs = ["input"]
    graph.shapes["input"] = [1, 512]
    graph.dtypes["input"] = "float32"

    # Weight and bias constants
    graph.constants["weight"] = np.random.randn(512, 1024).astype(np.float32)
    graph.constants["bias"] = np.random.randn(1024).astype(np.float32)
    graph.shapes["weight"] = [512, 1024]
    graph.shapes["bias"] = [1024]
    graph.dtypes["weight"] = "float32"
    graph.dtypes["bias"] = "float32"

    # MatMul
    graph.nodes.append(
        GraphNode(
            op_type=OpType.MATMUL,
            inputs=["input", "weight"],
            outputs=["matmul_out"],
            name="matmul_1",
        )
    )
    graph.shapes["matmul_out"] = [1, 1024]
    graph.dtypes["matmul_out"] = "float32"

    # Bias Add
    graph.nodes.append(
        GraphNode(
            op_type=OpType.ADD,
            inputs=["matmul_out", "bias"],
            outputs=["add_out"],
            name="add_1",
        )
    )
    graph.shapes["add_out"] = [1, 1024]
    graph.dtypes["add_out"] = "float32"

    # ReLU
    graph.nodes.append(
        GraphNode(
            op_type=OpType.RELU, inputs=["add_out"], outputs=["output"], name="relu_1"
        )
    )
    graph.outputs = ["output"]
    graph.shapes["output"] = [1, 1024]
    graph.dtypes["output"] = "float32"

    return graph


if __name__ == "__main__":
    # Test compilation
    graph = create_example_graph()
    print(f"Original nodes: {len(graph.nodes)}")

    compiled = compile_graph(graph)
    print(f"Compiled nodes: {len(compiled.nodes)}")
    for node in compiled.nodes:
        print(f"  {node.op_type.name}: {node.inputs} -> {node.outputs}")
