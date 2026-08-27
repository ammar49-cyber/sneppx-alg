"""Gradient checkpointing (activation recomputation).

Wraps a forward sub-function so that only the inputs to each checkpointed
segment are stored; activations inside the segment are recomputed during
backward. This trades compute for memory, enabling training of very deep
models that would otherwise OOM. The API mirrors
``torch.utils.checkpoint.checkpoint``.
"""

from typing import Callable, List, Tuple, Any, Optional
import numpy as np

from .tensor import Tensor
from .autograd import Function, _backward


class CheckpointFunction(Function):
    """Recomputes its wrapped ``fn`` during the backward pass.

    The forward output is disconnected from the inner autograd graph (the
    inner ``_GradFn`` nodes are dropped), so intermediate activations are not
    retained. On backward we re-run ``fn`` on detached clones of the inputs
    and backprop through the freshly built graph, returning gradients for the
    original inputs. Using clones keeps the originals' ``.grad`` correct even
    when an input tensor is shared by several checkpointed segments.
    """

    @staticmethod
    def forward(ctx, fn: Callable, *inputs: Tensor) -> Tensor:
        ctx.fn = fn
        ctx.inputs = inputs
        return fn(*inputs)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        fn = ctx.fn
        inputs = ctx.inputs
        clones = []
        for inp in inputs:
            c = Tensor.from_numpy(np.array(inp.data, copy=True), dtype=inp.dtype_name)
            c.requires_grad_(inp.requires_grad)
            clones.append(c)
        inner = fn(*clones)
        _backward(inner, grad_output)
        grads = [c.grad if inp.requires_grad else None for c, inp in zip(clones, inputs)]
        return grads


class CheckpointSegment:
    """Compatibility shim: records a recomputable segment for stats only.

    The actual recompute-on-backward behaviour lives in
    :class:`CheckpointFunction`; this keeps the previously exported name
    available for callers that inspected ``CheckpointSegment`` instances.
    """

    def __init__(self, fn: Callable, inputs: Tuple[Tensor, ...]):
        self.fn = fn
        self.inputs = inputs
        self.output: Optional[Tensor] = None

    def forward(self) -> Tensor:
        self.output = self.fn(*self.inputs)
        return self.output

    def recompute(self) -> Tensor:
        self.output = self.fn(*self.inputs)
        return self.output


def checkpoint(fn: Callable, *inputs: Tensor, use_reentrant: bool = True) -> Tensor:
    """Run ``fn(*inputs)`` but discard intermediate activations.

    During the backward pass the segment is recomputed. ``use_reentrant`` is
    accepted for API compatibility (the pure-NumPy engine is single-stream).
    """
    return CheckpointFunction.apply(fn, *inputs)


class GradientCheckpointer:
    """Manages a stack of checkpointed segments across a forward pass.

    Usage:
        with checkpointer.context():
            h = checkpointer.checkpoint(block1, x)
            h = checkpointer.checkpoint(block2, h)
        # ... compute loss, backward ...
    """

    def __init__(self):
        self.segments: List[Tuple[Callable, Tuple[Tensor, ...]]] = []
        self._active = False

    def context(self):
        return _CheckpointContext(self)

    def checkpoint(self, fn: Callable, *inputs: Tensor) -> Tensor:
        self.segments.append((fn, inputs))
        return CheckpointFunction.apply(fn, *inputs)

    def num_segments(self) -> int:
        return len(self.segments)

    def memory_saved_bytes(self, per_elem_bytes: int = 4) -> int:
        """Estimate memory saved by not storing intermediate activations.

        This is a heuristic: sum of input sizes (the only thing kept).
        """
        total = 0
        for _, inputs in self.segments:
            for t in inputs:
                total += int(np.prod(t.shape)) * per_elem_bytes
        return total


class _CheckpointContext:
    def __init__(self, owner: GradientCheckpointer):
        self.owner = owner

    def __enter__(self):
        self.owner._active = True
        self.owner.segments = []
        return self.owner

    def __exit__(self, *exc):
        self.owner._active = False
        return False


def checkpoint_sequential(
    layers: List[Callable], inputs: Tensor, segments: int = 1
) -> Tensor:
    """Checkpoint a sequential stack of ``layers`` in ``segments`` chunks.

    ``layers`` is a list of callables each taking one Tensor and returning
    one Tensor. With ``segments=1`` the whole stack is one checkpoint.
    """
    if segments < 1:
        segments = 1
    bounds = np.linspace(0, len(layers), segments + 1).astype(int)
    x = inputs
    for s in range(segments):
        lo, hi = bounds[s], bounds[s + 1]
        if hi <= lo:
            continue
        chunk = layers[lo:hi]

        def make_fn(chunk):
            def fn(t):
                h = t
                for layer in chunk:
                    h = layer(h)
                return h

            return fn

        x = checkpoint(make_fn(chunk), x)
    return x
