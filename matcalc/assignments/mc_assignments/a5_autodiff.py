"""
Assignment 5 -- Automatic differentiation from scratch.

Textbook: Chapters 8-9.   Lab Manual: Part 5.

Part A: forward mode with dual numbers.
Part B: a reverse-mode Tensor class (a tiny autograd).
Rules: NumPy only.  No JAX / PyTorch / autograd.
"""
import numpy as np


# =====================================================================
# Part A -- forward mode
# =====================================================================

class Dual:
    """A dual number val + der * eps with eps^2 = 0.

    Must support +, -, *, / and ** (real exponent) with Duals AND plain
    numbers on either side (2 * d, d + 1, 1 / d, ...), and unary minus.
    """

    def __init__(self, val, der=0.0):
        self.val = float(val)
        self.der = float(der)

    @staticmethod
    def _lift(o):
        return o if isinstance(o, Dual) else Dual(o, 0.0)

    def __add__(self, o):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def __radd__(self, o):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def __sub__(self, o):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def __rsub__(self, o):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def __mul__(self, o):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def __rmul__(self, o):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def __truediv__(self, o):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def __rtruediv__(self, o):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def __neg__(self):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def __pow__(self, p):
        """d ** p for a plain real number p."""
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError


def d_sin(x):
    """sin for Duals (and plain floats)."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def d_cos(x):
    """cos for Duals (and plain floats)."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def d_exp(x):
    """exp for Duals (and plain floats)."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def d_log(x):
    """Natural log for Duals (and plain floats)."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def derivative(f, x):
    """f'(x) for a scalar function f, by one evaluation on the dual number x + 1 eps."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def gradient_forward(f, x):
    """Gradient of f: R^n -> R in forward mode: n evaluations of f.

    f receives a LIST of n Duals and returns a Dual.  Returns a 1-D array.
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


# =====================================================================
# Part B -- reverse mode
# =====================================================================

def unbroadcast(grad, shape):
    """Sum `grad` over the axes that NumPy broadcasting added or stretched, to get `shape`."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


class Tensor:
    """A float array that records the operations applied to it.

    Attributes: data (np.ndarray, float), grad (same shape, starts at 0).
    Required operations (each returns a NEW Tensor that knows its parents
    and how to send gradients back to them):
        a + b, a - b, a * b, a / b, -a        (Tensors or numbers, with broadcasting)
        a ** p                                 (p a real number)
        a @ b                                  (2-D matrices)
        a.T                                    (transpose)
        a.sum(axis=None), a.mean(axis=None)
        a.exp(), a.log(), a.tanh(), a.relu(), a.sigmoid()
    and t.backward() for a scalar (size-1) Tensor t, which fills .grad of
    every Tensor t depends on, ADDING to existing .grad values.

    Design (given in __init__): out._parents is the tuple of input Tensors and
    out._back is a function of no arguments that, when called, ADDS out.grad's
    contribution to each parent's .grad.  backward() orders the graph
    topologically and calls each node's _back once, from the output down.
    __add__ is done for you as a worked pattern (it needs unbroadcast).
    """

    def __init__(self, data, parents=()):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._parents = parents
        self._back = lambda: None

    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def _lift(o):
        return o if isinstance(o, Tensor) else Tensor(o)

    def __add__(self, o):
        o = Tensor._lift(o)
        out = Tensor(self.data + o.data, (self, o))
        def back():
            self.grad += unbroadcast(out.grad, self.shape)
            o.grad += unbroadcast(out.grad, o.shape)
        out._back = back
        return out

    def __radd__(self, o):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def __mul__(self, o):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def __rmul__(self, o):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def __neg__(self):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def __sub__(self, o):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def __rsub__(self, o):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def __pow__(self, p):
        """a ** p for a plain real number p."""
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def __truediv__(self, o):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def __rtruediv__(self, o):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def __matmul__(self, o):
        """Matrix product of 2-D Tensors."""
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    @property
    def T(self):
        """The transpose (a property: write a.T, not a.T())."""
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def sum(self, axis=None):
        """Sum of all entries (axis=None) or over one integer axis, like ndarray.sum."""
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def mean(self, axis=None):
        """Mean of all entries (axis=None) or over one integer axis."""
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def exp(self):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def log(self):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def tanh(self):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def relu(self):
        """max(a, 0), with relu'(0) = 0."""
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def sigmoid(self):
        """1 / (1 + exp(-a)), without overflow for large |a|."""
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def backward(self):
        """For a size-1 Tensor: ADD ones to self.grad, then call _back of every node
        in reverse topological order (each node exactly once)."""
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError


def cross_entropy(logits, y):
    """Mean softmax cross-entropy of a 2-D logits Tensor with integer labels y, as ONE
    primitive operation: stable forward pass, backward pass (S - T) / N."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError
