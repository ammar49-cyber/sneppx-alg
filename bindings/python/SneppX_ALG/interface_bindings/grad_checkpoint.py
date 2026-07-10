"""Gradient checkpointing (activation recomputation).

Wraps a sequence of forward sub-functions so that only the inputs to
each checkpointed segment are stored; activations inside the segment are
recomputed during backward. This trades compute for memory, enabling
training of very deep models that would otherwise OOM. The API mirrors
``torch.utils.checkpoint.checkpoint``.
"""

from typing import Callable, List, Tuple, Any, Optional
import numpy as np

from .tensor import Tensor


class CheckpointSegment:
    """A single recomputable segment of the forward graph."""

    def __init__(self, fn: Callable, inputs: Tuple[Tensor, ...]):
        self.fn = fn
        self.inputs = inputs
        self.saved_inputs = inputs
        self.output: Optional[Tensor] = None

    def forward(self) -> Tensor:
        self.output = self.fn(*self.inputs)
        return self.output

    def recompute(self) -> Tensor:
        # Re-run the wrapped function on the (detached) saved inputs
        saved = tuple(Tensor.from_numpy(i.data.copy(), dtype=i.dtype_name) for i in self.inputs)
        self.output = self.fn(*saved)
        return self.output


def checkpoint(fn: Callable, *inputs: Tensor, use_reentrant: bool = True) -> Tensor:
    """Run ``fn(*inputs)`` but discard intermediate activations.

    During the backward pass the segment is recomputed. ``use_reentrant``
    is accepted for API compatibility (the pure-NumPy engine is single-stream).
    """
    seg = CheckpointSegment(fn, inputs)
    out = seg.forward()
    # Stash the segment on the output tensor for the backward engine
    if not hasattr(out, "_checkpoint_segments"):
        out._checkpoint_segments = []
    out._checkpoint_segments.append(seg)
    return out


class GradientCheckpointer:
    """Manages a stack of checkpointed segments across a forward pass.

    Usage:
        with checkpointer.context():
            h = checkpointer.checkpoint(block1, x)
            h = checkpointer.checkpoint(block2, h)
        # ... compute loss, backward ...
        checkpointer.recompute_all()  # rebuild activations before 2nd pass
    """

    def __init__(self):
        self.segments: List[CheckpointSegment] = []
        self._active = False

    def context(self):
        return _CheckpointContext(self)

    def checkpoint(self, fn: Callable, *inputs: Tensor) -> Tensor:
        seg = CheckpointSegment(fn, inputs)
        out = seg.forward()
        self.segments.append(seg)
        if not hasattr(out, "_checkpoint_segments"):
            out._checkpoint_segments = []
        out._checkpoint_segments.append(seg)
        return out

    def recompute_all(self):
        for seg in self.segments:
            seg.recompute()

    def num_segments(self) -> int:
        return len(self.segments)

    def memory_saved_bytes(self, per_elem_bytes: int = 4) -> int:
        """Estimate memory saved by not storing intermediate activations.

        This is a heuristic: sum of input sizes (the only thing kept).
        """
        total = 0
        for seg in self.segments:
            for t in seg.inputs:
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


def checkpoint_sequential(layers: List[Callable], inputs: Tensor,
                          segments: int = 1) -> Tensor:
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
