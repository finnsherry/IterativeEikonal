"""
    Sub-Riemannian
    ==============

    Solve the Eikonal PDE on W2 using a sub-Riemannian controller.

    Provides the following "top level" submodules:
      1. distancemap: compute the distance map (as solution of the Eikonal PDE)
      with respect to a data-driven sub-Riemannian metric. The metric must be
      diagonal with respect to the left invariant frame.
      2. backtracking: compute the geodesic, with respect to the distance map,
      connecting two points. 

    Additionally, we have the following "internal" submodules:
      1. interpolate: interpolate scalar and vector fields between grid points
      with trilinear interpolation.
      2. metric: compute the norm of vectors given some data-driven metric.
"""

# Access entire backend
import eikivp.W2.subRiemannian.distancemap
import eikivp.W2.subRiemannian.backtracking
import eikivp.W2.subRiemannian.interpolate
import eikivp.W2.subRiemannian.metric