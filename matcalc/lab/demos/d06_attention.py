import numpy as np
from mcalc import attention_forward, attention_backward, softmax_rows_backward, softmax, num_grad, rel_error

np.set_printoptions(precision=4, suppress=True)
# Worked Example 12.1: two tokens, d = 1, all weights 1, loss L = y_1
X = np.array([[0.0], [1.0]]); I1 = np.eye(1)
Y, cache = attention_forward(X, I1, I1, I1)
print("Y =", Y.ravel(), "  A =\n", cache[7])
dX, dWq, dWk, dWv = attention_backward(np.array([[1.0], [0.0]]), cache)
print("dX =", dX.ravel(), " (expected 0.75, 0.5)")

# softmax-row backward vs finite differences
rng = np.random.default_rng(0)
S = rng.standard_normal((4, 5)); G = rng.standard_normal((4, 5))
print("softmax rows: %.1e" % rel_error(softmax_rows_backward(softmax(S, axis=1), G),
                                       num_grad(lambda S: np.sum(G * softmax(S, axis=1)), S)))

# full gradient check of one head, with and without a causal mask
N, D, dk = 6, 5, 3
X = rng.standard_normal((N, D)); Wq, Wk, Wv = (rng.standard_normal((D, dk)) for _ in range(3))
G = rng.standard_normal((N, dk))
for causal in (False, True):
    def L(X=X, Wq=Wq, Wk=Wk, Wv=Wv):
        return np.sum(G * attention_forward(X, Wq, Wk, Wv, causal)[0])
    _, cache = attention_forward(X, Wq, Wk, Wv, causal)
    dX, dWq, dWk, dWv = attention_backward(G, cache)
    print("causal=%-5s dX %.1e  dWq %.1e  dWk %.1e  dWv %.1e" % (causal,
          rel_error(dX, num_grad(lambda Z: L(X=Z), X)), rel_error(dWq, num_grad(lambda W: L(Wq=W), Wq)),
          rel_error(dWk, num_grad(lambda W: L(Wk=W), Wk)), rel_error(dWv, num_grad(lambda W: L(Wv=W), Wv))))

# why divide by sqrt(d_k): the size of the softmax Jacobian with and without scaling
for d in (4, 64, 512):
    q, K = rng.standard_normal(d), rng.standard_normal((16, d))
    for scaled in (False, True):
        s = softmax(K @ q / (np.sqrt(d) if scaled else 1.0))
        J = np.diag(s) - np.outer(s, s)
        print("d=%3d scaled=%-5s max attention %.3f, |J|_F = %.3f" % (d, scaled, s.max(), np.linalg.norm(J)))
