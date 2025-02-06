"""
    M2
    ==

    Solve the Eikonal PDE on M2.

    Contains three submodules for different controller types on M2, which
    each contain methods for solving the corresponding Eikonal PDE and computing
    geodesics:
      1. Riemannian.
      2. subRiemannian.
      3. plus.

    Moreover provides the following "top level" submodule:
      1. vesselness: compute the M2 vesselness of an image, which can be put
      into a cost function and subsequently into a data-driven metric. 

    Additionally, we have the following "internal" submodules:
      1. derivatives: compute various derivatives of functions on M2.
      2. utils
"""

# Access entire backend
import eikivp.M2.derivatives
import eikivp.M2.vesselness
import eikivp.M2.utils
import eikivp.M2.Riemannian
import eikivp.M2.subRiemannian
import eikivp.M2.plus