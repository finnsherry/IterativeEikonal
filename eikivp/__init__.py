"""
EikIVP
======

The Python package *eikivp* contains methods to solve the Eikonal PDE on
R^2, M2, and W2 [1] using the iterative Initial Value Problem (IVP)
technique first described in Bekkers et al.[2], and to find geodesics
connecting points with respect to the distance map that solves the Eikonal
PDE.

One application in which we want to solve the Eikonal PDE and subsequently
find geodesics connecting pairs of points is vascular tracking. This package
contains methods to construct data-driven metrics on R^2 and M2, based
on multiscale vesselness filters, that will lead to geodesics that
(hopefully) track vessels.

Summary: compute distance map and geodesics with respect to data-driven
metric on R^2, M2, and W2.

References:
  [1]: N.J. van den Berg, F.M. Sherry, T.T.J.M. Berendschot, and R. Duits.
  "Crossing-Preserving Geodesic Tracking on Spherical Images."
  In: Scale Space and Variational Methods in Computer Vision (2025),
  pp. 192--204.
  DOI:10.1007/978-3-031-92369-2_15.
  [2]: E.J. Bekkers, R. Duits, A. Mashtakov, and G.R. Sanguinetti.
  "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)."
  In: SIAM Journal on Imaging Sciences (2015), pp. 2740--2770.
  DOI:10.1137/15M1018460.
"""

# Access entire backend
import eikivp.utils
import eikivp.visualisations
import eikivp.orientationscore
import eikivp.R2
import eikivp.M2

# Most important functions are available at top level
## R2
from eikivp.R2.distancemap import eikonal_solver as eikonal_solver_R2
from eikivp.R2.distancemap import eikonal_solver_uniform as eikonal_solver_R2_uniform
from eikivp.R2.distancemap import (
    eikonal_solver_multi_source as eikonal_solver_multi_source_R2,
)
from eikivp.R2.distancemap import (
    eikonal_solver_multi_source_uniform as eikonal_solver_multi_source_R2_uniform,
)
from eikivp.R2.backtracking import geodesic_back_tracking as geodesic_back_tracking_R2
from eikivp.R2.backtracking import (
    geodesic_back_tracking_multi_source as geodesic_back_tracking_multi_source_R2,
)

## M2
### Riemannian
from eikivp.M2.Riemannian.distancemap import (
    eikonal_solver as eikonal_solver_M2_Riemannian,
)
from eikivp.M2.Riemannian.distancemap import (
    eikonal_solver_uniform as eikonal_solver_M2_Riemannian_uniform,
)
from eikivp.M2.Riemannian.distancemap import (
    eikonal_solver_multi_source as eikonal_solver_multi_source_M2_Riemannian,
)
from eikivp.M2.Riemannian.distancemap import (
    eikonal_solver_multi_source_uniform as eikonal_solver_multi_source_M2_Riemannian_uniform,
)
from eikivp.M2.Riemannian.backtracking import (
    geodesic_back_tracking as geodesic_back_tracking_M2_Riemannian,
)
from eikivp.M2.Riemannian.backtracking import (
    geodesic_back_tracking_multi_source as geodesic_back_tracking_multi_source_M2_Riemannian,
)

### Sub-Riemannian
from eikivp.M2.subRiemannian.distancemap import (
    eikonal_solver as eikonal_solver_M2_sub_Riemannian,
)
from eikivp.M2.subRiemannian.distancemap import (
    eikonal_solver_uniform as eikonal_solver_M2_sub_Riemannian_uniform,
)
from eikivp.M2.subRiemannian.distancemap import (
    eikonal_solver_multi_source as eikonal_solver_multi_source_M2_sub_Riemannian,
)
from eikivp.M2.subRiemannian.distancemap import (
    eikonal_solver_multi_source_uniform as eikonal_solver_multi_source_M2_sub_Riemannian_uniform,
)
from eikivp.M2.subRiemannian.backtracking import (
    geodesic_back_tracking as geodesic_back_tracking_M2_sub_Riemannian,
)
from eikivp.M2.subRiemannian.backtracking import (
    geodesic_back_tracking_multi_source as geodesic_back_tracking_multi_source_M2_sub_Riemannian,
)

### Plus controller
from eikivp.M2.plus.distancemap import eikonal_solver as eikonal_solver_M2_plus
from eikivp.M2.plus.distancemap import (
    eikonal_solver_uniform as eikonal_solver_M2_plus_uniform,
)
from eikivp.M2.plus.distancemap import (
    eikonal_solver_multi_source as eikonal_solver_multi_source_M2_plus,
)
from eikivp.M2.plus.distancemap import (
    eikonal_solver_multi_source_uniform as eikonal_solver_multi_source_M2_plus_uniform,
)
from eikivp.M2.plus.backtracking import (
    geodesic_back_tracking as geodesic_back_tracking_M2_plus,
)
from eikivp.M2.plus.backtracking import (
    geodesic_back_tracking_multi_source as geodesic_back_tracking_multi_source_M2_plus,
)


### Single top level function to select any controller
def eikonal_solver_M2(
    cost,
    source_point,
    dxy,
    dθ,
    θs,
    controller="sub-Riemannian",
    G=None,
    ξ=None,
    plus_softness=0.0,
    target_point=None,
    n_max=1e5,
    n_max_initialisation=1e4,
    n_check=None,
    n_check_initialisation=None,
    tol=1e-3,
    dε=1.0,
    initial_condition=100.0,
):
    """
    Solve the Eikonal PDE on M2 equipped with a datadriven left invariant
    norm, with source at `source_point`, using the iterative method first
    described by Bekkers et al.[2] and generalised in [1].

    Args:
        `cost`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1, with shape `[Nx, Ny, Nθ]`.
        `source_point`: Tuple[int] describing index of source point in
          `cost`.
        `G_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: Orientation coordinate at every point in the grid on which
          `cost` is sampled.
      Optional:
        `controller`: Type of controller to use for computing the distance map.
          Can choose between "Riemannian", "sub-Riemannian", and "plus". If
          "Riemannian" is chosen, parameter `G`, the diagonal of the left
          invariant metric tensor field, must be provided. If "sub-Riemannian"
          or "plus" is chosen, parameter `ξ`, the stiffness of moving in the A1
          direction compared to the A3 direction, must be provided.
        `G`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis. Defaults
          to `None`
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0. Defaults to `None`.
        `plus_softness`: Strength of the plus controller, taking values between
          0 and 1. As `plus_softness` is decreased, motion in the reverse A1
          direction is increasingly inhibited. For `plus_softness` 0, motion is
          possibly exclusively in the forward A1 direction; for `plus_softness`
          1, we recover the sub-Riemannian metric that is symmetric in the A1
          direction. Defaults to 0.
        `target_point`: Tuple[int] describing index of target point in
          `cost`. Defaults to `None`. If `target_point` is provided, the
          algorithm will terminate when the Hamiltonian has converged at
          `target_point`; otherwise it will terminate when the Hamiltonian has
          converged throughout the domain.
        `n_max`: Maximum number of iterations, taking positive values. Defaults
          to 1e5.
        `n_max_initialisation`: Maximum number of iterations for the
          initialisation, taking positive values. Defaults to 1e4.
        `n_check`: Number of iterations between each convergence check, taking
          positive values. Should be at most `n_max`. Defaults to `None`; if no
          `n_check` is passed, convergence is only checked at `n_max`.
        `n_check_initialisation`: Number of iterations between each convergence
          check in the initialisation, taking positive values. Should be at most
          `n_max_initialisation`. Defaults to `None`; if no
          `n_check_initialisation` is passed, convergence is only checked at
          `n_max_initialisation`.
        `tol`: Tolerance for determining convergence of the Hamiltonian, taking
          positive values. Defaults to 1e-3.
        `dε`: Multiplier for varying the "time" step size, taking positive
          values. Defaults to 1.
        `initial_condition`: Initial value of the approximate distance map.
          Defaults to 100.

    Returns:
        np.ndarray of (approximate) distance map with respect to the datadriven
          left invariant metric tensor field described by `G_np` and `cost_np`.
        np.ndarray of upwind gradient field of (approximate) distance map.

    Notes:
        The base sub-Riemannian metric tensor field (i.e. with uniform cost),
          is given, for a pair of vectors v = v^i A_i and w = w^i A_i at point
          p, by
            G_p(v, w) = ξ^2 v^1 w^1 + v^3 w^3.
        The base Finsler function (i.e. with uniform cost), is given, for vector
          v = v^i A_i at point p, by
            F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
          where (x)_+ := max{x, 0} is the positive part of x.

    References:
        [1]: N.J. van den Berg, F.M. Sherry, T.T.J.M. Berendschot, and R. Duits.
          "Crossing-Preserving Geodesic Tracking on Spherical Images."
          In: Scale Space and Variational Methods in Computer Vision (2025),
          pp. 192--204.
          DOI:10.1007/978-3-031-92369-2_15.
        [2]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    if controller == "Riemannian":
        if G is None:
            raise ValueError(
                f"When using the Riemannian controller you must pass the entire diagonal of the left invariant metric tensor G!"
            )
        W, grad_W = eikonal_solver_M2_Riemannian(
            cost,
            source_point,
            G,
            dxy,
            dθ,
            θs,
            target_point=target_point,
            n_max=n_max,
            n_max_initialisation=n_max_initialisation,
            n_check=n_check,
            n_check_initialisation=n_check_initialisation,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    elif controller == "sub-Riemannian":
        if ξ is None:
            raise ValueError(
                f"When using the sub-Riemannian controller you must pass the the stiffness parameter ξ!"
            )
        W, grad_W = eikonal_solver_M2_sub_Riemannian(
            cost,
            source_point,
            ξ,
            dxy,
            dθ,
            θs,
            target_point=target_point,
            n_max=n_max,
            n_max_initialisation=n_max_initialisation,
            n_check=n_check,
            n_check_initialisation=n_check_initialisation,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    elif controller == "plus":
        if ξ is None:
            raise ValueError(
                f"When using the plus controller you must pass the the stiffness parameter ξ!"
            )
        W, grad_W = eikonal_solver_M2_plus(
            cost,
            source_point,
            ξ,
            dxy,
            dθ,
            θs,
            plus_softness=plus_softness,
            target_point=target_point,
            n_max=n_max,
            n_max_initialisation=n_max_initialisation,
            n_check=n_check,
            n_check_initialisation=n_check_initialisation,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    else:
        raise ValueError(
            f"""Controller "{controller}" is not supported! Choose one of "Riemannian", "sub-Riemannian", or "plus"."""
        )
    return W, grad_W


def eikonal_solver_M2_uniform(
    domain_shape,
    source_point,
    dxy,
    dθ,
    θs,
    controller="sub-Riemannian",
    G=None,
    ξ=None,
    plus_softness=0.0,
    target_point=None,
    n_max=1e5,
    n_check=None,
    tol=1e-3,
    dε=1.0,
    initial_condition=100.0,
):
    """
    Solve the Eikonal PDE on M equipped with a left invariant norm, with
    source at `source_point`, using the iterative method first
    described by Bekkers et al.[2] and generalised in [1].

    Args:
        `domain_shape`: Tuple[int] describing the shape of the domain, namely
          [Nx, Ny, Nθ].
        `source_point`: Tuple[int] describing index of source point in
          `domain_shape`.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: Orientation coordinate at every point in the grid on which
          `cost` is sampled.
      Optional:
        `controller`: Type of controller to use for computing the distance map.
          Can choose between "Riemannian", "sub-Riemannian", and "plus". If
          "Riemannian" is chosen, parameter `G`, the diagonal of the left
          invariant metric tensor field, must be provided. If "sub-Riemannian"
          or "plus" is chosen, parameter `ξ`, the stiffness of moving in the A1
          direction compared to the A3 direction, must be provided.
        `G`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis. Defaults
          to `None`
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0. Defaults to `None`.
        `plus_softness`: Strength of the plus controller, taking values between
          0 and 1. As `plus_softness` is decreased, motion in the reverse A1
          direction is increasingly inhibited. For `plus_softness` 0, motion is
          possibly exclusively in the forward A1 direction; for `plus_softness`
          1, we recover the sub-Riemannian metric that is symmetric in the A1
          direction. Defaults to 0.
        `target_point`: Tuple[int] describing index of target point in
          `domain_shape`. Defaults to `None`. If `target_point` is provided, the
          algorithm will terminate when the Hamiltonian has converged at
          `target_point`; otherwise it will terminate when the Hamiltonian has
          converged throughout the domain.
        `n_max`: Maximum number of iterations, taking positive values. Defaults
          to 1e5.
        `n_check`: Number of iterations between each convergence check, taking
          positive values. Should be at most `n_max` and `n_max_initialisation`.
          Defaults to `None`; if no `n_check` is passed, convergence is only
          checked at `n_max`.
        `tol`: Tolerance for determining convergence of the Hamiltonian, taking
          positive values. Defaults to 1e-3.
        `dε`: Multiplier for varying the "time" step size, taking positive
          values. Defaults to 1.
        `initial_condition`: Initial value of the approximate distance map.
          Defaults to 100.

    Returns:
        np.ndarray of (approximate) distance map with respect to the left
          invariant metric tensor field described by `G_np`.
        np.ndarray of upwind gradient field of (approximate) distance map.

    Notes:
        The base sub-Riemannian metric tensor field (i.e. with uniform cost),
          is given, for a pair of vectors v = v^i A_i and w = w^i A_i at point
          p, by
            G_p(v, w) = ξ^2 v^1 w^1 + v^3 w^3.
        The base Finsler function (i.e. with uniform cost), is given, for vector
          v = v^i A_i at point p, by
            F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
          where (x)_+ := max{x, 0} is the positive part of x.

    References:
        [1]: N.J. van den Berg, F.M. Sherry, T.T.J.M. Berendschot, and R. Duits.
          "Crossing-Preserving Geodesic Tracking on Spherical Images."
          In: Scale Space and Variational Methods in Computer Vision (2025),
          pp. 192--204.
          DOI:10.1007/978-3-031-92369-2_15.
        [2]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    if controller == "Riemannian":
        if G is None:
            raise ValueError(
                f"When using the Riemannian controller you must pass the entire diagonal of the left invariant metric tensor G!"
            )
        W, grad_W = eikonal_solver_M2_Riemannian_uniform(
            domain_shape,
            source_point,
            G,
            dxy,
            dθ,
            θs,
            target_point=target_point,
            n_max=n_max,
            n_check=n_check,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    elif controller == "sub-Riemannian":
        if ξ is None:
            raise ValueError(
                f"When using the sub-Riemannian controller you must pass the the stiffness parameter ξ!"
            )
        W, grad_W = eikonal_solver_M2_sub_Riemannian_uniform(
            domain_shape,
            source_point,
            ξ,
            dxy,
            dθ,
            θs,
            target_point=target_point,
            n_max=n_max,
            n_check=n_check,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    elif controller == "plus":
        if ξ is None:
            raise ValueError(
                f"When using the plus controller you must pass the the stiffness parameter ξ!"
            )
        W, grad_W = eikonal_solver_M2_plus_uniform(
            domain_shape,
            source_point,
            ξ,
            dxy,
            dθ,
            θs,
            plus_softness=plus_softness,
            target_point=target_point,
            n_max=n_max,
            n_check=n_check,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    else:
        raise ValueError(
            f"""Controller "{controller}" is not supported! Choose one of "Riemannian", "sub-Riemannian", or "plus"."""
        )
    return W, grad_W


def geodesic_back_tracking_M2(
    grad_W,
    source_point,
    target_point,
    cost,
    x_min,
    y_min,
    θ_min,
    dxy,
    dθ,
    θs,
    controller="sub-Riemannian",
    G=None,
    ξ=None,
    dt=1.0,
    n_max=10000,
):
    """
    Find the geodesic connecting `target_point` to `source_point`, using
    gradient descent back tracking, as first described by Bekkers et al.[2] and
    generalised in [1].

    Args:
        `grad_W`: np.ndarray of upwind gradient with respect to some cost of the
          approximate distance map, with shape [Nx, Ny, Nθ, 3].
        `source_point`: Tuple[int] describing index of source point in `cost`.
        `target_point`: Tuple[int] describing index of target point in `cost`.
        `cost`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1, with shape [Nx, Ny, Nθ].
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `θ_min`: minimum value of θ-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
        `dθ`: orientational resolution, taking values greater than 0.
        `θs`: Orientation coordinate at every point in the grid on which `cost`
          is sampled.
      Optional:
        `controller`: Type of controller to use for computing the distance map.
          Can choose between "Riemannian", "sub-Riemannian", and "plus". If
          "Riemannian" is chosen, parameter `G`, the diagonal of the left
          invariant metric tensor field, must be provided. If "sub-Riemannian"
          or "plus" is chosen, parameter `ξ`, the stiffness of moving in the A1
          direction compared to the A3 direction, must be provided.
        `G`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis. Defaults
          to `None`
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0. Defaults to `None`.
        `dt`: Step size, taking values greater than 0. Defaults to 1.
        `n_max`: Maximum number of points in geodesic, taking positive integral
          values. Defaults to 10000.

    Returns:
        np.ndarray of geodesic connecting `target_point` to `source_point`.

    Notes:
        The base sub-Riemannian metric tensor field (i.e. with uniform cost),
          is given, for a pair of vectors v = v^i A_i and w = w^i A_i at point
          p, by
            G_p(v, w) = ξ^2 v^1 w^1 + v^3 w^3.
        The base Finsler function (i.e. with uniform cost), is given, for vector
          v = v^i A_i at point p, by
            F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
          where (x)_+ := max{x, 0} is the positive part of x.

    References:
        [1]: N.J. van den Berg, F.M. Sherry, T.T.J.M. Berendschot, and R. Duits.
          "Crossing-Preserving Geodesic Tracking on Spherical Images."
          In: Scale Space and Variational Methods in Computer Vision (2025),
          pp. 192--204.
          DOI:10.1007/978-3-031-92369-2_15.
        [2]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    if controller == "Riemannian":
        if G is None:
            raise ValueError(
                f"When using the Riemannian controller you must pass the entire diagonal of the left invariant metric tensor G!"
            )
        γ = geodesic_back_tracking_M2_Riemannian(
            grad_W,
            source_point,
            target_point,
            cost,
            x_min,
            y_min,
            θ_min,
            dxy,
            dθ,
            θs,
            G,
            dt=dt,
            n_max=n_max,
        )
    elif controller == "sub-Riemannian":
        if ξ is None:
            raise ValueError(
                f"When using the sub-Riemannian controller you must pass the the stiffness parameter ξ!"
            )
        γ = geodesic_back_tracking_M2_sub_Riemannian(
            grad_W,
            source_point,
            target_point,
            cost,
            x_min,
            y_min,
            θ_min,
            dxy,
            dθ,
            θs,
            ξ,
            dt=dt,
            n_max=n_max,
        )
    elif controller == "plus":
        if ξ is None:
            raise ValueError(
                f"When using the plus controller you must pass the the stiffness parameter ξ!"
            )
        γ = geodesic_back_tracking_M2_plus(
            grad_W,
            source_point,
            target_point,
            cost,
            x_min,
            y_min,
            θ_min,
            dxy,
            dθ,
            θs,
            ξ,
            dt=dt,
            n_max=n_max,
        )
    else:
        raise ValueError(
            f"""Controller "{controller}" is not supported! Choose one of "Riemannian", "sub-Riemannian", or "plus"."""
        )
    return γ


### Single top level function to select any controller
def eikonal_solver_multi_source_M2(
    cost,
    source_points,
    dxy,
    dθ,
    θs,
    controller="sub-Riemannian",
    G=None,
    ξ=None,
    plus_softness=0.0,
    target_point=None,
    n_max=1e5,
    n_max_initialisation=1e4,
    n_check=None,
    n_check_initialisation=None,
    tol=1e-3,
    dε=1.0,
    initial_condition=100.0,
):
    """
    Solve the Eikonal PDE on M2 equipped with a datadriven left invariant
    norm, with source at `source_points`, using the iterative method first
    described by Bekkers et al.[2] and generalised in [1].

    Args:
        `cost`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1, with shape `[Nx, Ny, Nθ]`.
        `source_points`: Tuple[Tuple[int]] describing index of source points in
          `cost`.
        `G_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: Orientation coordinate at every point in the grid on which
          `cost` is sampled.
      Optional:
        `controller`: Type of controller to use for computing the distance map.
          Can choose between "Riemannian", "sub-Riemannian", and "plus". If
          "Riemannian" is chosen, parameter `G`, the diagonal of the left
          invariant metric tensor field, must be provided. If "sub-Riemannian"
          or "plus" is chosen, parameter `ξ`, the stiffness of moving in the A1
          direction compared to the A3 direction, must be provided.
        `G`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis. Defaults
          to `None`
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0. Defaults to `None`.
        `plus_softness`: Strength of the plus controller, taking values between
          0 and 1. As `plus_softness` is decreased, motion in the reverse A1
          direction is increasingly inhibited. For `plus_softness` 0, motion is
          possibly exclusively in the forward A1 direction; for `plus_softness`
          1, we recover the sub-Riemannian metric that is symmetric in the A1
          direction. Defaults to 0.
        `target_point`: Tuple[int] describing index of target point in
          `cost`. Defaults to `None`. If `target_point` is provided, the
          algorithm will terminate when the Hamiltonian has converged at
          `target_point`; otherwise it will terminate when the Hamiltonian has
          converged throughout the domain.
        `n_max`: Maximum number of iterations, taking positive values. Defaults
          to 1e5.
        `n_max_initialisation`: Maximum number of iterations for the
          initialisation, taking positive values. Defaults to 1e4.
        `n_check`: Number of iterations between each convergence check, taking
          positive values. Should be at most `n_max`. Defaults to `None`; if no
          `n_check` is passed, convergence is only checked at `n_max`.
        `n_check_initialisation`: Number of iterations between each convergence
          check in the initialisation, taking positive values. Should be at most
          `n_max_initialisation`. Defaults to `None`; if no
          `n_check_initialisation` is passed, convergence is only checked at
          `n_max_initialisation`.
        `tol`: Tolerance for determining convergence of the Hamiltonian, taking
          positive values. Defaults to 1e-3.
        `dε`: Multiplier for varying the "time" step size, taking positive
          values. Defaults to 1.
        `initial_condition`: Initial value of the approximate distance map.
          Defaults to 100.

    Returns:
        np.ndarray of (approximate) distance map with respect to the datadriven
          left invariant metric tensor field described by `G_np` and `cost_np`.
        np.ndarray of upwind gradient field of (approximate) distance map.

    Notes:
        The base sub-Riemannian metric tensor field (i.e. with uniform cost),
          is given, for a pair of vectors v = v^i A_i and w = w^i A_i at point
          p, by
            G_p(v, w) = ξ^2 v^1 w^1 + v^3 w^3.
        The base Finsler function (i.e. with uniform cost), is given, for vector
          v = v^i A_i at point p, by
            F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
          where (x)_+ := max{x, 0} is the positive part of x.

    References:
        [1]: N.J. van den Berg, F.M. Sherry, T.T.J.M. Berendschot, and R. Duits.
          "Crossing-Preserving Geodesic Tracking on Spherical Images."
          In: Scale Space and Variational Methods in Computer Vision (2025),
          pp. 192--204.
          DOI:10.1007/978-3-031-92369-2_15.
        [2]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    if controller == "Riemannian":
        if G is None:
            raise ValueError(
                f"When using the Riemannian controller you must pass the entire diagonal of the left invariant metric tensor G!"
            )
        W, grad_W = eikonal_solver_multi_source_M2_Riemannian(
            cost,
            source_points,
            G,
            dxy,
            dθ,
            θs,
            target_point=target_point,
            n_max=n_max,
            n_max_initialisation=n_max_initialisation,
            n_check=n_check,
            n_check_initialisation=n_check_initialisation,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    elif controller == "sub-Riemannian":
        if ξ is None:
            raise ValueError(
                f"When using the sub-Riemannian controller you must pass the the stiffness parameter ξ!"
            )
        W, grad_W = eikonal_solver_multi_source_M2_sub_Riemannian(
            cost,
            source_points,
            ξ,
            dxy,
            dθ,
            θs,
            target_point=target_point,
            n_max=n_max,
            n_max_initialisation=n_max_initialisation,
            n_check=n_check,
            n_check_initialisation=n_check_initialisation,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    elif controller == "plus":
        if ξ is None:
            raise ValueError(
                f"When using the plus controller you must pass the the stiffness parameter ξ!"
            )
        W, grad_W = eikonal_solver_multi_source_M2_plus(
            cost,
            source_points,
            ξ,
            dxy,
            dθ,
            θs,
            plus_softness=plus_softness,
            target_point=target_point,
            n_max=n_max,
            n_max_initialisation=n_max_initialisation,
            n_check=n_check,
            n_check_initialisation=n_check_initialisation,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    else:
        raise ValueError(
            f"""Controller "{controller}" is not supported! Choose one of "Riemannian", "sub-Riemannian", or "plus"."""
        )
    return W, grad_W


def eikonal_solver_multi_source_M2_uniform(
    domain_shape,
    source_points,
    dxy,
    dθ,
    θs,
    controller="sub-Riemannian",
    G=None,
    ξ=None,
    plus_softness=0.0,
    target_point=None,
    n_max=1e5,
    n_check=None,
    tol=1e-3,
    dε=1.0,
    initial_condition=100.0,
):
    """
    Solve the Eikonal PDE on M2 equipped with a left invariant norm, with
    source at `source_point`, using the iterative method first
    described by Bekkers et al.[2] and generalised in [1].

    Args:
        `domain_shape`: Tuple[int] describing the shape of the domain, namely
          [Nx, Ny, Nθ].
        `source_points`: Tuple[Tuple[int]] describing index of source points in
          `domain_shape`.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: Orientation coordinate at every point in the grid on which
          `cost` is sampled.
      Optional:
        `controller`: Type of controller to use for computing the distance map.
          Can choose between "Riemannian", "sub-Riemannian", and "plus". If
          "Riemannian" is chosen, parameter `G`, the diagonal of the left
          invariant metric tensor field, must be provided. If "sub-Riemannian"
          or "plus" is chosen, parameter `ξ`, the stiffness of moving in the A1
          direction compared to the A3 direction, must be provided.
        `G`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis. Defaults
          to `None`
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0. Defaults to `None`.
        `plus_softness`: Strength of the plus controller, taking values between
          0 and 1. As `plus_softness` is decreased, motion in the reverse A1
          direction is increasingly inhibited. For `plus_softness` 0, motion is
          possibly exclusively in the forward A1 direction; for `plus_softness`
          1, we recover the sub-Riemannian metric that is symmetric in the A1
          direction. Defaults to 0.
        `target_point`: Tuple[int] describing index of target point in
          `domain_shape`. Defaults to `None`. If `target_point` is provided, the
          algorithm will terminate when the Hamiltonian has converged at
          `target_point`; otherwise it will terminate when the Hamiltonian has
          converged throughout the domain.
        `n_max`: Maximum number of iterations, taking positive values. Defaults
          to 1e5.
        `n_check`: Number of iterations between each convergence check, taking
          positive values. Should be at most `n_max` and `n_max_initialisation`.
          Defaults to `None`; if no `n_check` is passed, convergence is only
          checked at `n_max`.
        `tol`: Tolerance for determining convergence of the Hamiltonian, taking
          positive values. Defaults to 1e-3.
        `dε`: Multiplier for varying the "time" step size, taking positive
          values. Defaults to 1.
        `initial_condition`: Initial value of the approximate distance map.
          Defaults to 100.

    Returns:
        np.ndarray of (approximate) distance map with respect to the left
          invariant metric tensor field described by `G_np`.
        np.ndarray of upwind gradient field of (approximate) distance map.

    Notes:
        The base sub-Riemannian metric tensor field (i.e. with uniform cost),
          is given, for a pair of vectors v = v^i A_i and w = w^i A_i at point
          p, by
            G_p(v, w) = ξ^2 v^1 w^1 + v^3 w^3.
        The base Finsler function (i.e. with uniform cost), is given, for vector
          v = v^i A_i at point p, by
            F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
          where (x)_+ := max{x, 0} is the positive part of x.

    References:
        [1]: N.J. van den Berg, F.M. Sherry, T.T.J.M. Berendschot, and R. Duits.
          "Crossing-Preserving Geodesic Tracking on Spherical Images."
          In: Scale Space and Variational Methods in Computer Vision (2025),
          pp. 192--204.
          DOI:10.1007/978-3-031-92369-2_15.
        [2]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    if controller == "Riemannian":
        if G is None:
            raise ValueError(
                f"When using the Riemannian controller you must pass the entire diagonal of the left invariant metric tensor G!"
            )
        W, grad_W = eikonal_solver_multi_source_M2_Riemannian_uniform(
            domain_shape,
            source_points,
            G,
            dxy,
            dθ,
            θs,
            target_point=target_point,
            n_max=n_max,
            n_check=n_check,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    elif controller == "sub-Riemannian":
        if ξ is None:
            raise ValueError(
                f"When using the sub-Riemannian controller you must pass the the stiffness parameter ξ!"
            )
        W, grad_W = eikonal_solver_multi_source_M2_sub_Riemannian_uniform(
            domain_shape,
            source_points,
            ξ,
            dxy,
            dθ,
            θs,
            target_point=target_point,
            n_max=n_max,
            n_check=n_check,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    elif controller == "plus":
        if ξ is None:
            raise ValueError(
                f"When using the plus controller you must pass the the stiffness parameter ξ!"
            )
        W, grad_W = eikonal_solver_multi_source_M2_plus_uniform(
            domain_shape,
            source_points,
            ξ,
            dxy,
            dθ,
            θs,
            plus_softness=plus_softness,
            target_point=target_point,
            n_max=n_max,
            n_check=n_check,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    else:
        raise ValueError(
            f"""Controller "{controller}" is not supported! Choose one of "Riemannian", "sub-Riemannian", or "plus"."""
        )
    return W, grad_W


def geodesic_back_tracking_multi_source_M2(
    grad_W,
    source_points,
    target_point,
    cost,
    x_min,
    y_min,
    θ_min,
    dxy,
    dθ,
    θs,
    controller="sub-Riemannian",
    G=None,
    ξ=None,
    dt=1.0,
    n_max=10000,
):
    """
    Find the geodesic connecting `target_point` to `source_point`, using
    gradient descent back tracking, as first described by Bekkers et al.[2] and
    generalised in [1].

    Args:
        `grad_W`: np.ndarray of upwind gradient with respect to some cost of the
          approximate distance map, with shape [Nx, Ny, Nθ, 3].
        `source_points`: Tuple[Tuple[int]] describing index of source points in
          `cost`.
        `target_point`: Tuple[int] describing index of target point in `cost`.
        `cost`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1, with shape [Nx, Ny, Nθ].
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `θ_min`: minimum value of θ-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
        `dθ`: orientational resolution, taking values greater than 0.
        `θs`: Orientation coordinate at every point in the grid on which `cost`
          is sampled.
      Optional:
        `controller`: Type of controller to use for computing the distance map.
          Can choose between "Riemannian", "sub-Riemannian", and "plus". If
          "Riemannian" is chosen, parameter `G`, the diagonal of the left
          invariant metric tensor field, must be provided. If "sub-Riemannian"
          or "plus" is chosen, parameter `ξ`, the stiffness of moving in the A1
          direction compared to the A3 direction, must be provided.
        `G`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis. Defaults
          to `None`
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0. Defaults to `None`.
        `dt`: Step size, taking values greater than 0. Defaults to 1.
        `n_max`: Maximum number of points in geodesic, taking positive integral
          values. Defaults to 10000.

    Returns:
        np.ndarray of geodesic connecting `target_point` to `source_point`.

    Notes:
        The base sub-Riemannian metric tensor field (i.e. with uniform cost),
          is given, for a pair of vectors v = v^i A_i and w = w^i A_i at point
          p, by
            G_p(v, w) = ξ^2 v^1 w^1 + v^3 w^3.
        The base Finsler function (i.e. with uniform cost), is given, for vector
          v = v^i A_i at point p, by
            F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
          where (x)_+ := max{x, 0} is the positive part of x.

    References:
        [1]: N.J. van den Berg, F.M. Sherry, T.T.J.M. Berendschot, and R. Duits.
          "Crossing-Preserving Geodesic Tracking on Spherical Images."
          In: Scale Space and Variational Methods in Computer Vision (2025),
          pp. 192--204.
          DOI:10.1007/978-3-031-92369-2_15.
        [2]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    if controller == "Riemannian":
        if G is None:
            raise ValueError(
                f"When using the Riemannian controller you must pass the entire diagonal of the left invariant metric tensor G!"
            )
        γ = geodesic_back_tracking_multi_source_M2_Riemannian(
            grad_W,
            source_points,
            target_point,
            cost,
            x_min,
            y_min,
            θ_min,
            dxy,
            dθ,
            θs,
            G,
            dt=dt,
            n_max=n_max,
        )
    elif controller == "sub-Riemannian":
        if ξ is None:
            raise ValueError(
                f"When using the sub-Riemannian controller you must pass the the stiffness parameter ξ!"
            )
        γ = geodesic_back_tracking_multi_source_M2_sub_Riemannian(
            grad_W,
            source_points,
            target_point,
            cost,
            x_min,
            y_min,
            θ_min,
            dxy,
            dθ,
            θs,
            ξ,
            dt=dt,
            n_max=n_max,
        )
    elif controller == "plus":
        if ξ is None:
            raise ValueError(
                f"When using the plus controller you must pass the the stiffness parameter ξ!"
            )
        γ = geodesic_back_tracking_multi_source_M2_plus(
            grad_W,
            source_points,
            target_point,
            cost,
            x_min,
            y_min,
            θ_min,
            dxy,
            dθ,
            θs,
            ξ,
            dt=dt,
            n_max=n_max,
        )
    else:
        raise ValueError(
            f"""Controller "{controller}" is not supported! Choose one of "Riemannian", "sub-Riemannian", or "plus"."""
        )
    return γ


## W2
### Riemannian
from eikivp.W2.Riemannian.distancemap import (
    eikonal_solver as eikonal_solver_W2_Riemannian,
)
from eikivp.W2.Riemannian.distancemap import (
    eikonal_solver_uniform as eikonal_solver_W2_Riemannian_uniform,
)
from eikivp.W2.Riemannian.distancemap import (
    eikonal_solver_multi_source as eikonal_solver_multi_source_W2_Riemannian,
)
from eikivp.W2.Riemannian.distancemap import (
    eikonal_solver_multi_source_uniform as eikonal_solver_multi_source_W2_Riemannian_uniform,
)
from eikivp.W2.Riemannian.backtracking import (
    geodesic_back_tracking as geodesic_back_tracking_W2_Riemannian,
)
from eikivp.W2.Riemannian.backtracking import (
    geodesic_back_tracking_multi_source as geodesic_back_tracking_multi_source_W2_Riemannian,
)

### Sub-Riemannian
from eikivp.W2.subRiemannian.distancemap import (
    eikonal_solver as eikonal_solver_W2_sub_Riemannian,
)
from eikivp.W2.subRiemannian.distancemap import (
    eikonal_solver_uniform as eikonal_solver_W2_sub_Riemannian_uniform,
)
from eikivp.W2.subRiemannian.distancemap import (
    eikonal_solver_multi_source as eikonal_solver_multi_source_W2_sub_Riemannian,
)
from eikivp.W2.subRiemannian.distancemap import (
    eikonal_solver_multi_source_uniform as eikonal_solver_multi_source_W2_sub_Riemannian_uniform,
)
from eikivp.W2.subRiemannian.backtracking import (
    geodesic_back_tracking as geodesic_back_tracking_W2_sub_Riemannian,
)
from eikivp.W2.subRiemannian.backtracking import (
    geodesic_back_tracking_multi_source as geodesic_back_tracking_multi_source_W2_sub_Riemannian,
)

### Plus controller
from eikivp.W2.plus.distancemap import eikonal_solver as eikonal_solver_W2_plus
from eikivp.W2.plus.distancemap import (
    eikonal_solver_uniform as eikonal_solver_W2_plus_uniform,
)
from eikivp.W2.plus.distancemap import (
    eikonal_solver_multi_source as eikonal_solver_multi_source_W2_plus,
)
from eikivp.W2.plus.distancemap import (
    eikonal_solver_multi_source_uniform as eikonal_solver_multi_source_W2_plus_uniform,
)
from eikivp.W2.plus.backtracking import (
    geodesic_back_tracking as geodesic_back_tracking_W2_plus,
)
from eikivp.W2.plus.backtracking import (
    geodesic_back_tracking_multi_source as geodesic_back_tracking_multi_source_W2_plus,
)


### Single top level function to select any controller
def eikonal_solver_W2(
    cost,
    source_point,
    dα,
    dβ,
    dφ,
    αs_np,
    φs_np,
    controller="sub-Riemannian",
    G=None,
    ξ=None,
    plus_softness=0.0,
    target_point=None,
    n_max=1e5,
    n_max_initialisation=1e4,
    n_check=None,
    n_check_initialisation=None,
    tol=1e-3,
    dε=1.0,
    initial_condition=100.0,
):
    """
    Solve the Eikonal PDE on W2 equipped with a datadriven left invariant
    norm, with source at `source_point`, using the iterative method described by
    Bekkers et al.[1]

    Args:
        `cost`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1, with shape [Nα, Nβ, Nφ].
        `source_point`: Tuple[int] describing index of source point in
          `cost`.
        `G_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis.
        `dα`: spatial resolution in the α-direction, taking values greater than
          0.
        `dβ`: spatial resolution in the β-direction, taking values greater than
          0.
        `dφ`: step size in orientational direction, taking values greater than
          0.
        `αs_np`: α-coordinate at every point in the grid on which `cost` is
          sampled.
        `φs_np`: Orientation coordinate at every point in the grid on which
          `cost` is sampled.
      Optional:
        `controller`: Type of controller to use for computing the distance map.
          Can choose between "Riemannian", "sub-Riemannian", and "plus". If
          "Riemannian" is chosen, parameter `G`, the diagonal of the left
          invariant metric tensor field, must be provided. If "sub-Riemannian"
          or "plus" is chosen, parameter `ξ`, the stiffness of moving in the B1
          direction compared to the B3 direction, must be provided.
        `G`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis. Defaults
          to `None`
        `ξ`: Stiffness of moving in the B1 direction compared to the B3
          direction, taking values greater than 0. Defaults to `None`.
        `plus_softness`: Strength of the plus controller, taking values between
          0 and 1. As `plus_softness` is decreased, motion in the reverse B1
          direction is increasingly inhibited. For `plus_softness` 0, motion is
          possibly exclusively in the forward B1 direction; for `plus_softness`
          1, we recover the sub-Riemannian metric that is symmetric in the B1
          direction. Defaults to 0.
        `target_point`: Tuple[int] describing index of target point in
          `cost`. Defaults to `None`. If `target_point` is provided, the
          algorithm will terminate when the Hamiltonian has converged at
          `target_point`; otherwise it will terminate when the Hamiltonian has
          converged throughout the domain.
        `n_max`: Maximum number of iterations, taking positive values. Defaults
          to 1e5.
        `n_max_initialisation`: Maximum number of iterations for the
          initialisation, taking positive values. Defaults to 1e4.
        `n_check`: Number of iterations between each convergence check, taking
          positive values. Should be at most `n_max`. Defaults to `None`; if no
          `n_check` is passed, convergence is only checked at `n_max`.
        `n_check_initialisation`: Number of iterations between each convergence
          check in the initialisation, taking positive values. Should be at most
          `n_max_initialisation`. Defaults to `None`; if no
          `n_check_initialisation` is passed, convergence is only checked at
          `n_max_initialisation`.
        `tol`: Tolerance for determining convergence of the Hamiltonian, taking
          positive values. Defaults to 1e-3.
        `dε`: Multiplier for varying the "time" step size, taking positive
          values. Defaults to 1.
        `initial_condition`: Initial value of the approximate distance map.
          Defaults to 100.

    Returns:
        np.ndarray of (approximate) distance map with respect to the datadriven
          left invariant metric tensor field described by `G_np` and `cost_np`.
        np.ndarray of upwind gradient field of (approximate) distance map.

    Notes:
        The base sub-Riemannian metric tensor field (i.e. with uniform cost),
          is given, for a pair of vectors v = v^i B_i and w = w^i B_i at point
          p, by
            G_p(v, w) = ξ^2 v^1 w^1 + v^3 w^3.
        The base Finsler function (i.e. with uniform cost), is given, for vector
          v = v^i B_i at point p, by
            F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
          where (x)_+ := max{x, 0} is the positive part of x.

    References:
        [1]: N.J. van den Berg, F.M. Sherry, T.T.J.M. Berendschot, and R. Duits.
          "Crossing-Preserving Geodesic Tracking on Spherical Images."
          In: Scale Space and Variational Methods in Computer Vision (2025),
          pp. 192--204.
          DOI:10.1007/978-3-031-92369-2_15.
        [2]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    if controller == "Riemannian":
        if G is None:
            raise ValueError(
                f"When using the Riemannian controller you must pass the entire diagonal of the left invariant metric tensor G!"
            )
        W, grad_W = eikonal_solver_W2_Riemannian(
            cost,
            source_point,
            G,
            dα,
            dβ,
            dφ,
            αs_np,
            φs_np,
            target_point=target_point,
            n_max=n_max,
            n_max_initialisation=n_max_initialisation,
            n_check=n_check,
            n_check_initialisation=n_check_initialisation,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    elif controller == "sub-Riemannian":
        if ξ is None:
            raise ValueError(
                f"When using the sub-Riemannian controller you must pass the the stiffness parameter ξ!"
            )
        W, grad_W = eikonal_solver_W2_sub_Riemannian(
            cost,
            source_point,
            ξ,
            dα,
            dβ,
            dφ,
            αs_np,
            φs_np,
            target_point=target_point,
            n_max=n_max,
            n_max_initialisation=n_max_initialisation,
            n_check=n_check,
            n_check_initialisation=n_check_initialisation,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    elif controller == "plus":
        if ξ is None:
            raise ValueError(
                f"When using the plus controller you must pass the the stiffness parameter ξ!"
            )
        W, grad_W = eikonal_solver_W2_plus(
            cost,
            source_point,
            ξ,
            dα,
            dβ,
            dφ,
            αs_np,
            φs_np,
            plus_softness=plus_softness,
            target_point=target_point,
            n_max=n_max,
            n_max_initialisation=n_max_initialisation,
            n_check=n_check,
            n_check_initialisation=n_check_initialisation,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    else:
        raise ValueError(
            f"""Controller "{controller}" is not supported! Choose one of "Riemannian", "sub-Riemannian", or "plus"."""
        )
    return W, grad_W


def eikonal_solver_W2_uniform(
    domain_shape,
    source_point,
    dα,
    dβ,
    dφ,
    αs_np,
    φs_np,
    controller="sub-Riemannian",
    G=None,
    ξ=None,
    plus_softness=0.0,
    target_point=None,
    n_max=1e5,
    n_check=None,
    tol=1e-3,
    dε=1.0,
    initial_condition=100.0,
):
    """
    Solve the Eikonal PDE on W2 equipped with a left invariant norm, with
    source at `source_point`, using the iterative method first
    described by Bekkers et al.[2] and generalised in [1].

    Args:
        `domain_shape`: Tuple[int] describing the shape of the domain, namely
          [Nα, Nβ, Nφ].
        `source_point`: Tuple[int] describing index of source point in
          `domain_shape`.
        `dα`: spatial resolution in the α-direction, taking values greater than
          0.
        `dβ`: spatial resolution in the β-direction, taking values greater than
          0.
        `dφ`: step size in orientational direction, taking values greater than
          0.
        `αs_np`: α-coordinate at every point in the grid on which `cost` is
          sampled.
        `φs_np`: Orientation coordinate at every point in the grid on which
          `cost` is sampled.
      Optional:
        `controller`: Type of controller to use for computing the distance map.
          Can choose between "Riemannian", "sub-Riemannian", and "plus". If
          "Riemannian" is chosen, parameter `G`, the diagonal of the left
          invariant metric tensor field, must be provided. If "sub-Riemannian"
          or "plus" is chosen, parameter `ξ`, the stiffness of moving in the B1
          direction compared to the B3 direction, must be provided.
        `G`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis. Defaults
          to `None`
        `ξ`: Stiffness of moving in the B1 direction compared to the B3
          direction, taking values greater than 0. Defaults to `None`.
        `plus_softness`: Strength of the plus controller, taking values between
          0 and 1. As `plus_softness` is decreased, motion in the reverse B1
          direction is increasingly inhibited. For `plus_softness` 0, motion is
          possibly exclusively in the forward A1 direction; for `plus_softness`
          1, we recover the sub-Riemannian metric that is symmetric in the B1
          direction. Defaults to 0.
        `target_point`: Tuple[int] describing index of target point in
          `domain_shape`. Defaults to `None`. If `target_point` is provided, the
          algorithm will terminate when the Hamiltonian has converged at
          `target_point`; otherwise it will terminate when the Hamiltonian has
          converged throughout the domain.
        `n_max`: Maximum number of iterations, taking positive values. Defaults
          to 1e5.
        `n_check`: Number of iterations between each convergence check, taking
          positive values. Should be at most `n_max` and `n_max_initialisation`.
          Defaults to `None`; if no `n_check` is passed, convergence is only
          checked at `n_max`.
        `tol`: Tolerance for determining convergence of the Hamiltonian, taking
          positive values. Defaults to 1e-3.
        `dε`: Multiplier for varying the "time" step size, taking positive
          values. Defaults to 1.
        `initial_condition`: Initial value of the approximate distance map.
          Defaults to 100.

    Returns:
        np.ndarray of (approximate) distance map with respect to the left
          invariant metric tensor field described by `G_np`.
        np.ndarray of upwind gradient field of (approximate) distance map.

    Notes:
        The base sub-Riemannian metric tensor field (i.e. with uniform cost),
          is given, for a pair of vectors v = v^i B_i and w = w^i B_i at point
          p, by
            G_p(v, w) = ξ^2 v^1 w^1 + v^3 w^3.
        The base Finsler function (i.e. with uniform cost), is given, for vector
          v = v^i B_i at point p, by
            F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
          where (x)_+ := max{x, 0} is the positive part of x.

    References:
        [1]: N.J. van den Berg, F.M. Sherry, T.T.J.M. Berendschot, and R. Duits.
          "Crossing-Preserving Geodesic Tracking on Spherical Images."
          In: Scale Space and Variational Methods in Computer Vision (2025),
          pp. 192--204.
          DOI:10.1007/978-3-031-92369-2_15.
        [2]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    if controller == "Riemannian":
        if G is None:
            raise ValueError(
                f"When using the Riemannian controller you must pass the entire diagonal of the left invariant metric tensor G!"
            )
        W, grad_W = eikonal_solver_W2_Riemannian_uniform(
            domain_shape,
            source_point,
            G,
            dβ,
            dφ,
            αs_np,
            φs_np,
            target_point=target_point,
            n_max=n_max,
            n_check=n_check,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    elif controller == "sub-Riemannian":
        if ξ is None:
            raise ValueError(
                f"When using the sub-Riemannian controller you must pass the the stiffness parameter ξ!"
            )
        W, grad_W = eikonal_solver_W2_sub_Riemannian_uniform(
            domain_shape,
            source_point,
            ξ,
            dβ,
            dφ,
            αs_np,
            φs_np,
            target_point=target_point,
            n_max=n_max,
            n_check=n_check,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    elif controller == "plus":
        if ξ is None:
            raise ValueError(
                f"When using the plus controller you must pass the the stiffness parameter ξ!"
            )
        W, grad_W = eikonal_solver_W2_plus_uniform(
            domain_shape,
            source_point,
            ξ,
            dβ,
            dφ,
            αs_np,
            φs_np,
            plus_softness=plus_softness,
            target_point=target_point,
            n_max=n_max,
            n_check=n_check,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    else:
        raise ValueError(
            f"""Controller "{controller}" is not supported! Choose one of "Riemannian", "sub-Riemannian", or "plus"."""
        )
    return W, grad_W


def geodesic_back_tracking_W2(
    grad_W,
    source_point,
    target_point,
    cost,
    α_min,
    β_min,
    φ_min,
    dα,
    dβ,
    dφ,
    αs,
    φs,
    controller="sub-Riemannian",
    G=None,
    ξ=None,
    dt=1.0,
    n_max=10000,
):
    """
    Find the geodesic connecting `target_point` to `source_point`, using
    gradient descent back tracking, as first described by Bekkers et al.[2] and
    generalised in [1].

    Args:
        `grad_W`: np.ndarray of upwind gradient with respect to some cost of the
          approximate distance map, with shape [Nα, Nβ, Nφ, 3].
        `source_point`: Tuple[int] describing index of source point in `cost`.
        `target_point`: Tuple[int] describing index of target point in `cost`.
        `cost`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1, with shape [Nα, Nβ, Nφ].
        `α_min`: minimum value of α-coordinates in rectangular domain.
        `β_min`: minimum value of β-coordinates in rectangular domain.
        `φ_min`: minimum value of φ-coordinates in rectangular domain.
        `dα`: spatial resolution in the α-direction, taking values greater than
          0.
        `dβ`: spatial resolution in the β-direction, taking values greater than
          0.
        `dφ`: orientational resolution, taking values greater than 0.
        `αs`: α-coordinate at every point in the grid on which `cost` is
          sampled.
        `φs`: Orientation coordinate at every point in the grid on which
          `cost_np` is sampled.
      Optional:
        `controller`: Type of controller to use for computing the distance map.
          Can choose between "Riemannian", "sub-Riemannian", and "plus". If
          "Riemannian" is chosen, parameter `G`, the diagonal of the left
          invariant metric tensor field, must be provided. If "sub-Riemannian"
          or "plus" is chosen, parameter `ξ`, the stiffness of moving in the B1
          direction compared to the B3 direction, must be provided.
        `G`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis. Defaults
          to `None`
        `ξ`: Stiffness of moving in the B1 direction compared to the B3
          direction, taking values greater than 0. Defaults to `None`.
        `dt`: Step size, taking values greater than 0. Defaults to 1.
        `n_max`: Maximum number of points in geodesic, taking positive integral
          values. Defaults to 10000.

    Returns:
        np.ndarray of geodesic connecting `target_point` to `source_point`.

    Notes:
        The base sub-Riemannian metric tensor field (i.e. with uniform cost),
          is given, for a pair of vectors v = v^i B_i and w = w^i B_i at point
          p, by
            G_p(v, w) = ξ^2 v^1 w^1 + v^3 w^3.
        The base Finsler function (i.e. with uniform cost), is given, for vector
          v = v^i B_i at point p, by
            F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
          where (x)_+ := max{x, 0} is the positive part of x.

    References:
        [1]: N.J. van den Berg, F.M. Sherry, T.T.J.M. Berendschot, and R. Duits.
          "Crossing-Preserving Geodesic Tracking on Spherical Images."
          In: Scale Space and Variational Methods in Computer Vision (2025),
          pp. 192--204.
          DOI:10.1007/978-3-031-92369-2_15.
        [2]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    if controller == "Riemannian":
        if G is None:
            raise ValueError(
                f"When using the Riemannian controller you must pass the entire diagonal of the left invariant metric tensor G!"
            )
        γ = geodesic_back_tracking_W2_Riemannian(
            grad_W,
            source_point,
            target_point,
            cost,
            α_min,
            β_min,
            φ_min,
            dα,
            dβ,
            dφ,
            αs,
            φs,
            G,
            dt=dt,
            n_max=n_max,
        )
    elif controller == "sub-Riemannian":
        if ξ is None:
            raise ValueError(
                f"When using the sub-Riemannian controller you must pass the the stiffness parameter ξ!"
            )
        γ = geodesic_back_tracking_W2_sub_Riemannian(
            grad_W,
            source_point,
            target_point,
            cost,
            α_min,
            β_min,
            φ_min,
            dα,
            dβ,
            dφ,
            αs,
            φs,
            ξ,
            dt=dt,
            n_max=n_max,
        )
    elif controller == "plus":
        if ξ is None:
            raise ValueError(
                f"When using the plus controller you must pass the the stiffness parameter ξ!"
            )
        γ = geodesic_back_tracking_W2_plus(
            grad_W,
            source_point,
            target_point,
            cost,
            α_min,
            β_min,
            φ_min,
            dα,
            dβ,
            dφ,
            αs,
            φs,
            ξ,
            dt=dt,
            n_max=n_max,
        )
    else:
        raise ValueError(
            f"""Controller "{controller}" is not supported! Choose one of "Riemannian", "sub-Riemannian", or "plus"."""
        )
    return γ


def eikonal_solver_multi_source_W2(
    cost,
    source_points,
    dα,
    dβ,
    dφ,
    αs_np,
    φs_np,
    controller="sub-Riemannian",
    G=None,
    ξ=None,
    plus_softness=0.0,
    target_point=None,
    n_max=1e5,
    n_max_initialisation=1e4,
    n_check=None,
    n_check_initialisation=None,
    tol=1e-3,
    dε=1.0,
    initial_condition=100.0,
):
    """
    Solve the Eikonal PDE on W2 equipped with a datadriven left invariant
    norm, with source at `source_points`, using the iterative method first
    described by Bekkers et al.[2] and generalised in [1].

    Args:
        `cost`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1, with shape [Nα, Nβ, Nφ].
        `source_points`: Tuple[Tuple[int]] describing index of source points in
          `cost`.
        `G_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis.
        `dα`: spatial resolution in the α-direction, taking values greater than
          0.
        `dβ`: spatial resolution in the β-direction, taking values greater than
          0.
        `dφ`: step size in orientational direction, taking values greater than
          0.
        `αs_np`: α-coordinate at every point in the grid on which `cost` is
          sampled.
        `φs_np`: Orientation coordinate at every point in the grid on which
          `cost` is sampled.
      Optional:
        `controller`: Type of controller to use for computing the distance map.
          Can choose between "Riemannian", "sub-Riemannian", and "plus". If
          "Riemannian" is chosen, parameter `G`, the diagonal of the left
          invariant metric tensor field, must be provided. If "sub-Riemannian"
          or "plus" is chosen, parameter `ξ`, the stiffness of moving in the B1
          direction compared to the B3 direction, must be provided.
        `G`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis. Defaults
          to `None`
        `ξ`: Stiffness of moving in the B1 direction compared to the B3
          direction, taking values greater than 0. Defaults to `None`.
        `plus_softness`: Strength of the plus controller, taking values between
          0 and 1. As `plus_softness` is decreased, motion in the reverse B1
          direction is increasingly inhibited. For `plus_softness` 0, motion is
          possibly exclusively in the forward B1 direction; for `plus_softness`
          1, we recover the sub-Riemannian metric that is symmetric in the B1
          direction. Defaults to 0.
        `target_point`: Tuple[int] describing index of target point in
          `cost`. Defaults to `None`. If `target_point` is provided, the
          algorithm will terminate when the Hamiltonian has converged at
          `target_point`; otherwise it will terminate when the Hamiltonian has
          converged throughout the domain.
        `n_max`: Maximum number of iterations, taking positive values. Defaults
          to 1e5.
        `n_max_initialisation`: Maximum number of iterations for the
          initialisation, taking positive values. Defaults to 1e4.
        `n_check`: Number of iterations between each convergence check, taking
          positive values. Should be at most `n_max`. Defaults to `None`; if no
          `n_check` is passed, convergence is only checked at `n_max`.
        `n_check_initialisation`: Number of iterations between each convergence
          check in the initialisation, taking positive values. Should be at most
          `n_max_initialisation`. Defaults to `None`; if no
          `n_check_initialisation` is passed, convergence is only checked at
          `n_max_initialisation`.
        `tol`: Tolerance for determining convergence of the Hamiltonian, taking
          positive values. Defaults to 1e-3.
        `dε`: Multiplier for varying the "time" step size, taking positive
          values. Defaults to 1.
        `initial_condition`: Initial value of the approximate distance map.
          Defaults to 100.

    Returns:
        np.ndarray of (approximate) distance map with respect to the datadriven
          left invariant metric tensor field described by `G_np` and `cost_np`.
        np.ndarray of upwind gradient field of (approximate) distance map.

    Notes:
        The base sub-Riemannian metric tensor field (i.e. with uniform cost),
          is given, for a pair of vectors v = v^i B_i and w = w^i B_i at point
          p, by
            G_p(v, w) = ξ^2 v^1 w^1 + v^3 w^3.
        The base Finsler function (i.e. with uniform cost), is given, for vector
          v = v^i B_i at point p, by
            F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
          where (x)_+ := max{x, 0} is the positive part of x.

    References:
        [1]: N.J. van den Berg, F.M. Sherry, T.T.J.M. Berendschot, and R. Duits.
          "Crossing-Preserving Geodesic Tracking on Spherical Images."
          In: Scale Space and Variational Methods in Computer Vision (2025),
          pp. 192--204.
          DOI:10.1007/978-3-031-92369-2_15.
        [2]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    if controller == "Riemannian":
        if G is None:
            raise ValueError(
                f"When using the Riemannian controller you must pass the entire diagonal of the left invariant metric tensor G!"
            )
        W, grad_W = eikonal_solver_multi_source_W2_Riemannian(
            cost,
            source_points,
            G,
            dα,
            dβ,
            dφ,
            αs_np,
            φs_np,
            target_point=target_point,
            n_max=n_max,
            n_max_initialisation=n_max_initialisation,
            n_check=n_check,
            n_check_initialisation=n_check_initialisation,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    elif controller == "sub-Riemannian":
        if ξ is None:
            raise ValueError(
                f"When using the sub-Riemannian controller you must pass the the stiffness parameter ξ!"
            )
        W, grad_W = eikonal_solver_multi_source_W2_sub_Riemannian(
            cost,
            source_points,
            ξ,
            dα,
            dβ,
            dφ,
            αs_np,
            φs_np,
            target_point=target_point,
            n_max=n_max,
            n_max_initialisation=n_max_initialisation,
            n_check=n_check,
            n_check_initialisation=n_check_initialisation,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    elif controller == "plus":
        if ξ is None:
            raise ValueError(
                f"When using the plus controller you must pass the the stiffness parameter ξ!"
            )
        W, grad_W = eikonal_solver_multi_source_W2_plus(
            cost,
            source_points,
            ξ,
            dα,
            dβ,
            dφ,
            αs_np,
            φs_np,
            plus_softness=plus_softness,
            target_point=target_point,
            n_max=n_max,
            n_max_initialisation=n_max_initialisation,
            n_check=n_check,
            n_check_initialisation=n_check_initialisation,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    else:
        raise ValueError(
            f"""Controller "{controller}" is not supported! Choose one of "Riemannian", "sub-Riemannian", or "plus"."""
        )
    return W, grad_W


def eikonal_solver_multi_source_W2_uniform(
    domain_shape,
    source_points,
    dα,
    dβ,
    dφ,
    αs_np,
    φs_np,
    controller="sub-Riemannian",
    G=None,
    ξ=None,
    plus_softness=0.0,
    target_point=None,
    n_max=1e5,
    n_check=None,
    tol=1e-3,
    dε=1.0,
    initial_condition=100.0,
):
    """
    Solve the Eikonal PDE on W2 equipped with a left invariant norm, with
    source at `source_points`, using the iterative method first
    described by Bekkers et al.[2] and generalised in [1].

    Args:
        `domain_shape`: Tuple[int] describing the shape of the domain, namely
          [Nα, Nβ, Nφ].
        `source_points`: Tuple[Tuple[int]] describing index of source points in
          `domain_shape`.
        `dα`: spatial resolution in the α-direction, taking values greater than
          0.
        `dβ`: spatial resolution in the β-direction, taking values greater than
          0.
        `dφ`: step size in orientational direction, taking values greater than
          0.
        `αs_np`: α-coordinate at every point in the grid on which `cost` is
          sampled.
        `φs_np`: Orientation coordinate at every point in the grid on which
          `cost` is sampled.
      Optional:
        `controller`: Type of controller to use for computing the distance map.
          Can choose between "Riemannian", "sub-Riemannian", and "plus". If
          "Riemannian" is chosen, parameter `G`, the diagonal of the left
          invariant metric tensor field, must be provided. If "sub-Riemannian"
          or "plus" is chosen, parameter `ξ`, the stiffness of moving in the B1
          direction compared to the B3 direction, must be provided.
        `G`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis. Defaults
          to `None`
        `ξ`: Stiffness of moving in the B1 direction compared to the B3
          direction, taking values greater than 0. Defaults to `None`.
        `plus_softness`: Strength of the plus controller, taking values between
          0 and 1. As `plus_softness` is decreased, motion in the reverse B1
          direction is increasingly inhibited. For `plus_softness` 0, motion is
          possibly exclusively in the forward A1 direction; for `plus_softness`
          1, we recover the sub-Riemannian metric that is symmetric in the B1
          direction. Defaults to 0.
        `target_point`: Tuple[int] describing index of target point in
          `domain_shape`. Defaults to `None`. If `target_point` is provided, the
          algorithm will terminate when the Hamiltonian has converged at
          `target_point`; otherwise it will terminate when the Hamiltonian has
          converged throughout the domain.
        `n_max`: Maximum number of iterations, taking positive values. Defaults
          to 1e5.
        `n_check`: Number of iterations between each convergence check, taking
          positive values. Should be at most `n_max` and `n_max_initialisation`.
          Defaults to `None`; if no `n_check` is passed, convergence is only
          checked at `n_max`.
        `tol`: Tolerance for determining convergence of the Hamiltonian, taking
          positive values. Defaults to 1e-3.
        `dε`: Multiplier for varying the "time" step size, taking positive
          values. Defaults to 1.
        `initial_condition`: Initial value of the approximate distance map.
          Defaults to 100.

    Returns:
        np.ndarray of (approximate) distance map with respect to the left
          invariant metric tensor field described by `G_np`.
        np.ndarray of upwind gradient field of (approximate) distance map.

    Notes:
        The base sub-Riemannian metric tensor field (i.e. with uniform cost),
          is given, for a pair of vectors v = v^i B_i and w = w^i B_i at point
          p, by
            G_p(v, w) = ξ^2 v^1 w^1 + v^3 w^3.
        The base Finsler function (i.e. with uniform cost), is given, for vector
          v = v^i B_i at point p, by
            F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
          where (x)_+ := max{x, 0} is the positive part of x.

    References:
        [1]: N.J. van den Berg, F.M. Sherry, T.T.J.M. Berendschot, and R. Duits.
          "Crossing-Preserving Geodesic Tracking on Spherical Images."
          In: Scale Space and Variational Methods in Computer Vision (2025),
          pp. 192--204.
          DOI:10.1007/978-3-031-92369-2_15.
        [2]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    if controller == "Riemannian":
        if G is None:
            raise ValueError(
                f"When using the Riemannian controller you must pass the entire diagonal of the left invariant metric tensor G!"
            )
        W, grad_W = eikonal_solver_multi_source_W2_Riemannian_uniform(
            domain_shape,
            source_points,
            G,
            dα,
            dβ,
            dφ,
            αs_np,
            φs_np,
            target_point=target_point,
            n_max=n_max,
            n_check=n_check,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    elif controller == "sub-Riemannian":
        if ξ is None:
            raise ValueError(
                f"When using the sub-Riemannian controller you must pass the the stiffness parameter ξ!"
            )
        W, grad_W = eikonal_solver_multi_source_W2_sub_Riemannian_uniform(
            domain_shape,
            source_points,
            ξ,
            dα,
            dβ,
            dφ,
            αs_np,
            φs_np,
            target_point=target_point,
            n_max=n_max,
            n_check=n_check,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    elif controller == "plus":
        if ξ is None:
            raise ValueError(
                f"When using the plus controller you must pass the the stiffness parameter ξ!"
            )
        W, grad_W = eikonal_solver_multi_source_W2_plus_uniform(
            domain_shape,
            source_points,
            ξ,
            dα,
            dβ,
            dφ,
            αs_np,
            φs_np,
            plus_softness=plus_softness,
            target_point=target_point,
            n_max=n_max,
            n_check=n_check,
            tol=tol,
            dε=dε,
            initial_condition=initial_condition,
        )
    else:
        raise ValueError(
            f"""Controller "{controller}" is not supported! Choose one of "Riemannian", "sub-Riemannian", or "plus"."""
        )
    return W, grad_W


def geodesic_back_tracking_multi_source_W2(
    grad_W,
    source_points,
    target_point,
    cost,
    α_min,
    β_min,
    φ_min,
    dα,
    dβ,
    dφ,
    αs,
    φs,
    controller="sub-Riemannian",
    G=None,
    ξ=None,
    dt=1.0,
    n_max=10000,
):
    """
    Find the geodesic connecting `target_point` to `source_points`, using
    gradient descent back tracking, as first described by Bekkers et al.[2] and
    generalised in [1].

    Args:
        `grad_W`: np.ndarray of upwind gradient with respect to some cost of the
          approximate distance map, with shape [Nα, Nβ, Nφ, 3].
        `source_points`: Tuple[Tuple[int]] describing index of source points in
          `cost`.
        `target_point`: Tuple[int] describing index of target point in `cost`.
        `cost`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1, with shape [Nα, Nβ, Nφ].
        `α_min`: minimum value of α-coordinates in rectangular domain.
        `β_min`: minimum value of β-coordinates in rectangular domain.
        `φ_min`: minimum value of φ-coordinates in rectangular domain.
        `dα`: spatial resolution in the α-direction, taking values greater than
          0.
        `dβ`: spatial resolution in the β-direction, taking values greater than
          0.
        `dφ`: orientational resolution, taking values greater than 0.
        `αs`: α-coordinate at every point in the grid on which `cost` is
          sampled.
        `φs`: Orientation coordinate at every point in the grid on which
          `cost_np` is sampled.
      Optional:
        `controller`: Type of controller to use for computing the distance map.
          Can choose between "Riemannian", "sub-Riemannian", and "plus". If
          "Riemannian" is chosen, parameter `G`, the diagonal of the left
          invariant metric tensor field, must be provided. If "sub-Riemannian"
          or "plus" is chosen, parameter `ξ`, the stiffness of moving in the B1
          direction compared to the B3 direction, must be provided.
        `G`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis. Defaults
          to `None`
        `ξ`: Stiffness of moving in the B1 direction compared to the B3
          direction, taking values greater than 0. Defaults to `None`.
        `dt`: Step size, taking values greater than 0. Defaults to 1.
        `n_max`: Maximum number of points in geodesic, taking positive integral
          values. Defaults to 10000.

    Returns:
        np.ndarray of geodesic connecting `target_point` to `source_point`.

    Notes:
        The base sub-Riemannian metric tensor field (i.e. with uniform cost),
          is given, for a pair of vectors v = v^i B_i and w = w^i B_i at point
          p, by
            G_p(v, w) = ξ^2 v^1 w^1 + v^3 w^3.
        The base Finsler function (i.e. with uniform cost), is given, for vector
          v = v^i B_i at point p, by
            F(p, v)^2 = ξ^2 (v^1)_+^2 + (v^3)^2,
          where (x)_+ := max{x, 0} is the positive part of x.

    References:
        [1]: N.J. van den Berg, F.M. Sherry, T.T.J.M. Berendschot, and R. Duits.
          "Crossing-Preserving Geodesic Tracking on Spherical Images."
          In: Scale Space and Variational Methods in Computer Vision (2025),
          pp. 192--204.
          DOI:10.1007/978-3-031-92369-2_15.
        [2]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    if controller == "Riemannian":
        if G is None:
            raise ValueError(
                f"When using the Riemannian controller you must pass the entire diagonal of the left invariant metric tensor G!"
            )
        γ = geodesic_back_tracking_multi_source_W2_Riemannian(
            grad_W,
            source_points,
            target_point,
            cost,
            α_min,
            β_min,
            φ_min,
            dα,
            dβ,
            dφ,
            αs,
            φs,
            G,
            dt=dt,
            n_max=n_max,
        )
    elif controller == "sub-Riemannian":
        if ξ is None:
            raise ValueError(
                f"When using the sub-Riemannian controller you must pass the the stiffness parameter ξ!"
            )
        γ = geodesic_back_tracking_multi_source_W2_sub_Riemannian(
            grad_W,
            source_points,
            target_point,
            cost,
            α_min,
            β_min,
            φ_min,
            dα,
            dβ,
            dφ,
            αs,
            φs,
            ξ,
            dt=dt,
            n_max=n_max,
        )
    elif controller == "plus":
        if ξ is None:
            raise ValueError(
                f"When using the plus controller you must pass the the stiffness parameter ξ!"
            )
        γ = geodesic_back_tracking_multi_source_W2_plus(
            grad_W,
            source_points,
            target_point,
            cost,
            α_min,
            β_min,
            φ_min,
            dα,
            dβ,
            dφ,
            αs,
            φs,
            ξ,
            dt=dt,
            n_max=n_max,
        )
    else:
        raise ValueError(
            f"""Controller "{controller}" is not supported! Choose one of "Riemannian", "sub-Riemannian", or "plus"."""
        )
    return γ
