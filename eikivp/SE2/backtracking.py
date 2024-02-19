"""
    backtracking
    ============

    Provides methods to compute the geodesic, with respect to some distance map,
    connecting two points in SE(2). The primary methods are:
      1. `geodesic_back_tracking`: compute the geodesic using gradient descent.
      The gradient must be provided; it is computed along with the distance map
      by the corresponding methods in the distancemap module.
      2. TODO `geodesic_back_tracking_plus`: compute the geodesic using gradient
      descent when the distance map was computed using the plus controller. The
      gradient must be provided; it is computed along with the distance map by
      the corresponding methods in the distancemap module.
"""

import numpy as np
import taichi as ti
from eikivp.SE2.interpolate import (
    vectorfield_trilinear_interpolate_LI,
    scalar_trilinear_interpolate
)
from eikivp.SE2.metric import (
    vector_LI_to_static, 
    align_to_real_axis_point,
    align_to_real_axis_scalar_field,
    align_to_real_axis_vector_field,
)
from eikivp.utils import sparse_to_dense


def geodesic_back_tracking(grad_W_np, source_point, target_point, cost_np, xs_np, ys_np, θs_np, G_np, dt=None, β=0., n_max=10000):
    """
    Find the geodesic connecting `target_point` to `source_point`, using 
    gradient descent back tracking, as described in Bekkers et al. "A PDE 
    approach to Data-Driven Sub-Riemannian Geodesics in SE(2)" (2015).

    Args:
        `grad_W_np`: np.ndarray of upwind gradient with respect to some cost of 
          the approximate distance map.
        `source_point`: Tuple[int] describing index of source point in `W_np`.
        `target_point`: Tuple[int] describing index of target point in `W_np`.
        `cost_np`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1.
        `G_np`: np.ndarray(shape=(3, 3), dtype=[float]) of matrix of left 
          invariant metric tensor field with respect to left invariant basis.
        `dt`: Step size, taking values greater than 0. Defaults to the minimum
          of `cost_np`.
      Optional:
        `dt`: Step size, taking values greater than 0. Defaults to 1.
        `β`: Momentum parameter in gradient descent, taking values between 0 and 
          1. Defaults to 0.
        `n_max`: Maximum number of points in geodesic, taking positive integral
          values. Defaults to 10000.

    Returns:
        np.ndarray of geodesic connecting `target_point` to `source_point`.
    """
    # Align with (x, y, θ)-frame
    grad_W_np = align_to_real_axis_vector_field(grad_W_np)
    shape = grad_W_np.shape[0:-1]
    cost_np = align_to_real_axis_scalar_field(cost_np)
    source_point = align_to_real_axis_point(source_point, shape)
    target_point = align_to_real_axis_point(target_point, shape)
    xs_np = align_to_real_axis_scalar_field(xs_np)
    ys_np = align_to_real_axis_scalar_field(ys_np)
    θs_np = align_to_real_axis_scalar_field(θs_np)

    # Set hyperparameters
    G = ti.Matrix(G_np, ti.f32)
    if dt is None:
        # It would make sense to also include G somehow, but I am not sure how.
        dt = cost_np.min()

    # Initialise Taichi objects
    grad_W = ti.Vector.field(n=3, dtype=ti.f32, shape=shape)
    grad_W.from_numpy(grad_W_np)
    cost = ti.field(dtype=ti.f32, shape=shape)
    cost.from_numpy(cost_np)
    source_point = ti.Vector(source_point, dt=ti.f32)
    target_point = ti.Vector(target_point, dt=ti.f32)
    θs = ti.field(dtype=ti.f32, shape=θs_np.shape)
    θs.from_numpy(θs_np)

    # Perform backtracking
    γ_list = ti.root.dynamic(ti.i, n_max)
    γ = ti.Vector.field(n=3, dtype=ti.f32)
    γ_list.place(γ)

    γ_len = geodesic_back_tracking_backend(grad_W, source_point, target_point, θs, G, cost, dt, n_max, β, γ)
    γ_dense = ti.Vector.field(n=3, dtype=ti.f32, shape=γ_len)
    print(f"Geodesic consists of {γ_len} points.")
    sparse_to_dense(γ, γ_dense)
    γ_ci = γ_dense.to_numpy()

    # Align with (I, J, K)-frame
    γ_np = convert_continuous_indices_to_real_space_SE2(γ_ci, xs_np, ys_np, θs_np)
    return γ_np

@ti.kernel
def geodesic_back_tracking_backend(
    grad_W: ti.template(),
    source_point: ti.types.vector(3, ti.f32),
    target_point: ti.types.vector(3, ti.f32),
    θs: ti.template(),
    G: ti.types.matrix(3, 3, ti.f32),
    cost: ti.template(),
    dt: ti.f32,
    n_max: ti.i32,
    β: ti.f32,
    γ: ti.template()
) -> ti.i32:
    """
    @taichi.kernel

    Find the geodesic connecting `target_point` to `source_point`, using
    gradient descent backtracking, as described in Bekkers et al. "A PDE 
    approach to Data-Driven Sub-Riemannian Geodesics in SE(2)" (2015).

    Args:
      Static:
        `grad_W`: ti.field(dtype=[float], shape=shape) of upwind gradient with
          respect to some cost of the approximate distance map.
        `dt`: Gradient descent step size, taking values greater than 0.
        `source_point`: ti.types.vector(n=3, dtype=[float]) describing index of 
          source point in `W_np`.
        `target_point`: ti.types.vector(n=3, dtype=[float]) describing index of 
          target point in `W_np`.
        `G`: ti.types.matrix(n=3, m=3, dtype=[float]) of constants of metric 
          tensor with respect to left invariant basis.
        `cost`: ti.field(dtype=[float]) of cost function, taking values between
          0 and 1.
        `n_max`: Maximum number of points in geodesic, taking positive integral
          values. Defaults to 10000.
        `β`: *Currently not used* Momentum parameter in gradient descent, taking 
          values between 0 and 1. Defaults to 0. 
        `*_target`: Indices of the target point.
      Mutated:
        `γ`: ti.Vector.field(n=2, dtype=[float]) of coordinates of points on the
          geodesic. #SNode stuff#

    Returns:
        Number of points in the geodesic.
    """
    point = target_point
    γ.append(point)
    tol = 2 * dt
    n = 0
    while (ti.math.length(point - source_point) >= tol) and (n < n_max - 2):
        gradient_at_point_LI = vectorfield_trilinear_interpolate_LI(grad_W, point, G, cost)
        θ = scalar_trilinear_interpolate(θs, point)
        gradient_at_point = vector_LI_to_static(gradient_at_point_LI, θ)
        new_point = get_next_point(point, gradient_at_point, dt)
        γ.append(new_point)
        point = new_point
        n += 1
    γ.append(source_point)
    return γ.length()

@ti.func
def get_next_point(
    point: ti.types.vector(n=3, dtype=ti.f32),
    gradient_at_point: ti.types.vector(n=3, dtype=ti.f32),
    dt: ti.f32
) -> ti.types.vector(n=3, dtype=ti.f32):
    """
    @taichi.func

    Compute the next point in the gradient descent.

    Args:
        `point`: ti.types.vector(n=2, dtype=[float]) coordinates of current 
          point.
        `gradient_at_point`: ti.types.vector(n=2, dtype=[float]) value of 
          gradient at current point.
        `dt`: Gradient descent step size, taking values greater than 0.

    Returns:
        Next point in the gradient descent.
    """
    new_point = ti.Vector([0., 0., 0.], dt=ti.f32)
    new_point[0] = point[0] - dt * gradient_at_point[0]
    new_point[1] = point[1] - dt * gradient_at_point[1]
    new_point[2] = point[2] - dt * gradient_at_point[2]
    return new_point

def convert_continuous_indices_to_real_space_SE2(γ_ci_np, xs_np, ys_np, θs_np):
    """
    Convert the continuous indices in the geodesic `γ_ci_np` to the 
    corresponding real space coordinates described by `xs_np`, `ys_np`, and
    `θs_np`.
    """
    γ_ci = ti.Vector.field(n=3, dtype=ti.f32, shape=γ_ci_np.shape[0])
    γ_ci.from_numpy(γ_ci_np)
    γ = ti.Vector.field(n=3, dtype=ti.f32, shape=γ_ci.shape)

    xs = ti.field(dtype=ti.f32, shape=xs_np.shape)
    xs.from_numpy(xs_np)
    ys = ti.field(dtype=ti.f32, shape=ys_np.shape)
    ys.from_numpy(ys_np)
    θs = ti.field(dtype=ti.f32, shape=θs_np.shape)
    θs.from_numpy(θs_np)

    continuous_indices_to_real_SE2(γ_ci, xs, ys, θs, γ)

    return γ.to_numpy()

@ti.kernel
def continuous_indices_to_real_SE2(
    γ_ci: ti.template(),
    xs: ti.template(),
    ys: ti.template(),
    θs: ti.template(),
    γ: ti.template()
):
    """
    @taichi.kernel

    Interpolate the real space coordinates described by `xs`, `ys`, and `θs` at 
    the continuous indices in `γ_ci`.
    """
    for I in ti.grouped(γ_ci):
        γ[I][0] = scalar_trilinear_interpolate(xs, γ_ci[I])
        γ[I][1] = scalar_trilinear_interpolate(ys, γ_ci[I])
        γ[I][2] = scalar_trilinear_interpolate(θs, γ_ci[I])