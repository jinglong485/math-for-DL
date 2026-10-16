import numpy as np
import matplotlib.pyplot as plt
from plotstyle import C1, C2, C3, savefig
from linalg_scratch import vandermonde, lstsq_normal, lstsq_qr, lstsq_svd

t = np.linspace(0, 1, 50)
degrees = range(1, 17)
errs = {"normal equations": [], "Householder QR": [], "SVD": []}
for d in degrees:
    A = vandermonde(t, d)
    c_true = np.ones(d + 1)                    # exact data from a known polynomial
    b = A @ c_true
    for (name, f) in zip(errs, (lstsq_normal, lstsq_qr, lstsq_svd)):
        try:
            c = f(A, b)
            errs[name].append(np.abs(c - c_true).max())
        except ValueError:                     # Cholesky breaks down
            errs[name].append(np.nan)
for (name, e), col, mk in zip(errs.items(), (C1, C2, C3), ("o", "s", "^")):
    plt.semilogy(list(degrees), e, marker=mk, color=col, label=name)
plt.xlabel("polynomial degree"); plt.ylabel("max error in coefficients")
plt.legend(); savefig("polyfit_methods")
print("degree  kappa(A)    normal eq.   QR        SVD")
for d in (4, 8, 12, 16):
    s = np.linalg.svd(vandermonde(t, d), compute_uv=False)
    print("%4d   %9.1e   %9.1e  %9.1e  %9.1e" % (d, s[0] / s[-1],
          errs["normal equations"][d - 1], errs["Householder QR"][d - 1], errs["SVD"][d - 1]))
