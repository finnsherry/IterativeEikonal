"""
    W2
    ==

    Solve the Eikonal PDE on W2.

    Contains three submodules for different controller types on W2, which
    each contain methods for solving the corresponding Eikonal PDE and computing
    geodesics:
      1. Riemannian.
      2. subRiemannian.
      3. plus.

    Additionally, we have the following "internal" submodules:
      1. derivatives: compute various derivatives of functions on W2.
      2. utils
"""

# Access entire backend
import eikivp.W2.derivatives
import eikivp.W2.costfunction
import eikivp.W2.utils
import eikivp.W2.Riemannian
import eikivp.W2.subRiemannian
import eikivp.W2.plus