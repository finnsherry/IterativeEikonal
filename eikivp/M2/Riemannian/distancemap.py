"""
distancemap
===========

Provides methods to compute the distance map on M2 with respect to a
data-driven left invariant Riemannian metric, by solving the Eikonal PDE
using the iterative Initial Value Problem (IVP) technique first described by
Bekkers et al.[2] and generalised in [1]. The primary methods are:
  1. `eikonal_solver`: solve the Eikonal PDE with respect to some
  data-driven left invariant Riemannian metric, defined by the diagonal
  components of the underlying left invariant metric, with respect to the
  left invariant basis {A1, A2, A3}, and a cost function.
  2. `eikonal_solver_uniform`: solve the Eikonal PDE with respect to some
  left invariant Riemannian metric, defined by its diagonal components, with
  respect to the left invariant basis {A1, A2, A3}.

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

import numpy as np
import h5py
import taichi as ti
from tqdm import tqdm
from eikivp.M2.derivatives import upwind_derivatives
from eikivp.M2.utils import (
    get_boundary_conditions,
    get_boundary_conditions_multi_source,
    check_convergence,
    check_convergence_multi_source,
)
from eikivp.M2.Riemannian.metric import invert_metric
from eikivp.utils import (
    get_initial_W,
    apply_boundary_conditions,
    get_padded_cost,
    unpad_array,
)


def import_W(params, folder):
    """
    Import the distance and its gradient matching `params`.
    """
    image_name = params["image_name"]
    λ = params["λ"]
    p = params["p"]
    G = params["G"]
    if "target_point" in params:
        target_point = params["target_point"]
    else:
        target_point = "default"
    cost_domain = params["cost_domain"]
    if cost_domain == "M2":
        σ_s_list = params["σ_s_list"]
        σ_o = params["σ_o"]
        σ_s_ext = params["σ_s_ext"]
        σ_o_ext = params["σ_o_ext"]
        if "source_point" in params:
            source_point = params["source_point"]
            distance_filename = f"{folder}\\M2_R_ss_s={[s for s in σ_s_list]}_s_o={σ_o}_s_s_ext={σ_s_ext}_s_o_ext={σ_o_ext}_l={λ}_p={p}_G={[g for g in G]}_s={source_point}.hdf5"
            with h5py.File(distance_filename, "r") as distance_file:
                assert (
                    np.all(σ_s_list == distance_file.attrs["σ_s_list"])
                    and σ_o == distance_file.attrs["σ_o"]
                    and σ_s_ext == distance_file.attrs["σ_s_ext"]
                    and σ_o_ext == distance_file.attrs["σ_o_ext"]
                    and image_name == distance_file.attrs["image_name"]
                    and λ == distance_file.attrs["λ"]
                    and p == distance_file.attrs["p"]
                    and G == distance_file.attrs["G"]
                    and np.all(source_point == distance_file.attrs["source_point"])
                    and np.all(target_point == distance_file.attrs["target_point"])
                ), "There is a parameter mismatch!"
                W = distance_file["Distance"][()]
                grad_W = distance_file["Gradient"][()]
        else:
            source_points = params["source_points"]
            distance_filename = f"{folder}\\M2_R_ss_s={[s for s in σ_s_list]}_s_o={σ_o}_s_s_ext={σ_s_ext}_s_o_ext={σ_o_ext}_l={λ}_p={p}_G={[g for g in G]}.hdf5"
            with h5py.File(distance_filename, "r") as distance_file:
                assert (
                    np.all(σ_s_list == distance_file.attrs["σ_s_list"])
                    and σ_o == distance_file.attrs["σ_o"]
                    and σ_s_ext == distance_file.attrs["σ_s_ext"]
                    and σ_o_ext == distance_file.attrs["σ_o_ext"]
                    and image_name == distance_file.attrs["image_name"]
                    and λ == distance_file.attrs["λ"]
                    and p == distance_file.attrs["p"]
                    and G == distance_file.attrs["G"]
                    and np.all(source_points == distance_file.attrs["source_points"])
                    and np.all(target_point == distance_file.attrs["target_point"])
                ), "There is a parameter mismatch!"
                W = distance_file["Distance"][()]
                grad_W = distance_file["Gradient"][()]
    elif cost_domain == "R2":
        scales = params["scales"]
        α = params["α"]
        γ = params["γ"]
        ε = params["ε"]
        if "source_point" in params:
            source_point = params["source_point"]
            distance_filename = f"{folder}\\M2_R_ss={[s for s in scales]}_a={α}_g={γ}_e={ε}_l={λ}_p={p}_G={[g for g in G]}_s={source_point}.hdf5"
            with h5py.File(distance_filename, "r") as distance_file:
                assert (
                    np.all(scales == distance_file.attrs["scales"])
                    and α == distance_file.attrs["α"]
                    and γ == distance_file.attrs["γ"]
                    and ε == distance_file.attrs["ε"]
                    and image_name == distance_file.attrs["image_name"]
                    and λ == distance_file.attrs["λ"]
                    and p == distance_file.attrs["p"]
                    and G == distance_file.attrs["G"]
                    and np.all(source_point == distance_file.attrs["source_point"])
                    and np.all(target_point == distance_file.attrs["target_point"])
                ), "There is a parameter mismatch!"
                W = distance_file["Distance"][()]
                grad_W = distance_file["Gradient"][()]
        else:
            source_points = params["source_points"]
            distance_filename = f"{folder}\\M2_R_ss={[s for s in scales]}_a={α}_g={γ}_e={ε}_l={λ}_p={p}_G={[g for g in G]}.hdf5"
            with h5py.File(distance_filename, "r") as distance_file:
                assert (
                    np.all(scales == distance_file.attrs["scales"])
                    and α == distance_file.attrs["α"]
                    and γ == distance_file.attrs["γ"]
                    and ε == distance_file.attrs["ε"]
                    and image_name == distance_file.attrs["image_name"]
                    and λ == distance_file.attrs["λ"]
                    and p == distance_file.attrs["p"]
                    and G == distance_file.attrs["G"]
                    and np.all(source_points == distance_file.attrs["source_points"])
                    and np.all(target_point == distance_file.attrs["target_point"])
                ), "There is a parameter mismatch!"
                W = distance_file["Distance"][()]
                grad_W = distance_file["Gradient"][()]
    return W, grad_W


def export_W(W, grad_W, params, folder):
    """
    Export the distance and its gradient to hdf5 with `params` stored as metadata.
    """
    image_name = params["image_name"]
    λ = params["λ"]
    p = params["p"]
    G = params["G"]
    if "target_point" in params:
        target_point = params["target_point"]
    else:
        target_point = "default"
    cost_domain = params["cost_domain"]
    if cost_domain == "M2":
        σ_s_list = params["σ_s_list"]
        σ_o = params["σ_o"]
        σ_s_ext = params["σ_s_ext"]
        σ_o_ext = params["σ_o_ext"]
        if "source_point" in params:
            source_point = params["source_point"]
            distance_filename = f"{folder}\\M2_R_ss_s={[s for s in σ_s_list]}_s_o={σ_o}_s_s_ext={σ_s_ext}_s_o_ext={σ_o_ext}_l={λ}_p={p}_G={[g for g in G]}_s={source_point}.hdf5"
            with h5py.File(distance_filename, "w") as distance_file:
                distance_file.create_dataset("Distance", data=W)
                distance_file.create_dataset("Gradient", data=grad_W)
                distance_file.attrs["σ_s_list"] = σ_s_list
                distance_file.attrs["σ_o"] = σ_o
                distance_file.attrs["σ_s_ext"] = σ_s_ext
                distance_file.attrs["σ_o_ext"] = σ_o_ext
                distance_file.attrs["image_name"] = image_name
                distance_file.attrs["λ"] = λ
                distance_file.attrs["p"] = p
                distance_file.attrs["G"] = G
                distance_file.attrs["source_point"] = source_point
                distance_file.attrs["target_point"] = target_point
        else:
            source_points = params["source_points"]
            distance_filename = f"{folder}\\M2_R_ss_s={[s for s in σ_s_list]}_s_o={σ_o}_s_s_ext={σ_s_ext}_s_o_ext={σ_o_ext}_l={λ}_p={p}_G={[g for g in G]}.hdf5"
            with h5py.File(distance_filename, "w") as distance_file:
                distance_file.create_dataset("Distance", data=W)
                distance_file.create_dataset("Gradient", data=grad_W)
                distance_file.attrs["σ_s_list"] = σ_s_list
                distance_file.attrs["σ_o"] = σ_o
                distance_file.attrs["σ_s_ext"] = σ_s_ext
                distance_file.attrs["σ_o_ext"] = σ_o_ext
                distance_file.attrs["image_name"] = image_name
                distance_file.attrs["λ"] = λ
                distance_file.attrs["p"] = p
                distance_file.attrs["G"] = G
                distance_file.attrs["source_points"] = source_points
                distance_file.attrs["target_point"] = target_point
    elif cost_domain == "R2":
        scales = params["scales"]
        α = params["α"]
        γ = params["γ"]
        ε = params["ε"]
        if "source_point" in params:
            source_point = params["source_point"]
            distance_filename = f"{folder}\\M2_R_ss={[s for s in scales]}_a={α}_g={γ}_e={ε}_l={λ}_p={p}_G={[g for g in G]}_s={source_point}.hdf5"
            with h5py.File(distance_filename, "w") as distance_file:
                distance_file.create_dataset("Distance", data=W)
                distance_file.create_dataset("Gradient", data=grad_W)
                distance_file.attrs["scales"] = scales
                distance_file.attrs["α"] = α
                distance_file.attrs["γ"] = γ
                distance_file.attrs["ε"] = ε
                distance_file.attrs["image_name"] = image_name
                distance_file.attrs["λ"] = λ
                distance_file.attrs["p"] = p
                distance_file.attrs["G"] = G
                distance_file.attrs["source_point"] = source_point
                distance_file.attrs["target_point"] = target_point
        else:
            source_points = params["source_points"]
            distance_filename = f"{folder}\\M2_R_ss={[s for s in scales]}_a={α}_g={γ}_e={ε}_l={λ}_p={p}_G={[g for g in G]}.hdf5"
            with h5py.File(distance_filename, "w") as distance_file:
                distance_file.create_dataset("Distance", data=W)
                distance_file.create_dataset("Gradient", data=grad_W)
                distance_file.attrs["scales"] = scales
                distance_file.attrs["α"] = α
                distance_file.attrs["γ"] = γ
                distance_file.attrs["ε"] = ε
                distance_file.attrs["image_name"] = image_name
                distance_file.attrs["λ"] = λ
                distance_file.attrs["p"] = p
                distance_file.attrs["G"] = G
                distance_file.attrs["source_points"] = source_points
                distance_file.attrs["target_point"] = target_point


# Data-driven left invariant


def eikonal_solver(
    cost_np,
    source_point,
    G_np,
    dxy,
    dθ,
    θs_np,
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
    Riemannian metric tensor field defined by `G_np` and `cost_np`, with source
    at `source_point`, using the iterative method first  described by Bekkers et
    al.[2] and generalised in [1].

    Args:
        `cost_np`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1, with shape [Nx, Ny, Nθ].
        `source_point`: Tuple[int] describing index of source point in
          `cost_np`.
        `G_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs_np`: Orientation coordinate at every point in the grid on which
          `cost_np` is sampled.
      Optional:
        `target_point`: Tuple[int] describing index of target point in
          `cost_np`. Defaults to `None`. If `target_point` is provided, the
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
    # First compute for uniform cost to get initial W
    print("Solving Eikonal PDE with left invariant metric to compute initialisation.")
    W_init_np, _ = eikonal_solver_uniform(
        cost_np.shape,
        source_point,
        G_np,
        dxy,
        dθ,
        θs_np,
        target_point=target_point,
        n_max=n_max_initialisation,
        n_check=n_check_initialisation,
        tol=tol,
        dε=dε,
        initial_condition=initial_condition,
    )

    print("Solving Eikonal PDE data-driven left invariant metric.")

    # Set hyperparameters
    G_inv = ti.Vector(invert_metric(G_np), ti.f32)
    # Heuristic, so that W does not become negative.
    # The sqrt(3) comes from the fact that the norm of the gradient consists of
    # 3 terms.
    ε = dε * (dxy / G_inv.max()) / np.sqrt(3)  # * cost_np.min()
    if n_check is None:  # Only check convergence at n_max
        n_check = n_max
    N_check = int(n_max / n_check)

    # Initialise Taichi objects
    cost = get_padded_cost(cost_np, pad_shape=((1,), (1,), (0,)))
    W = get_padded_cost(
        W_init_np, pad_shape=((1,), (1,), (0,)), pad_value=initial_condition
    )
    boundarypoints, boundaryvalues = get_boundary_conditions(source_point)
    apply_boundary_conditions(W, boundarypoints, boundaryvalues)

    θs_np = np.pad(θs_np, ((1,), (1,), (0,)), mode="edge")
    θs = ti.field(dtype=ti.f32, shape=θs_np.shape)
    θs.from_numpy(θs_np)

    A1_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A1_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A2_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A2_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A3_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A3_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A1_W = ti.field(dtype=ti.f32, shape=W.shape)
    A2_W = ti.field(dtype=ti.f32, shape=W.shape)
    A3_W = ti.field(dtype=ti.f32, shape=W.shape)
    dW_dt = ti.field(dtype=ti.f32, shape=W.shape)
    grad_W = ti.Vector.field(n=3, dtype=ti.f32, shape=W.shape)

    # Compute approximate distance map
    is_converged = False
    for n in range(N_check):
        for _ in tqdm(range(int(n_check))):
            step_W(
                W,
                cost,
                G_inv,
                dxy,
                dθ,
                θs,
                ε,
                A1_forward,
                A1_backward,
                A2_forward,
                A2_backward,
                A3_forward,
                A3_backward,
                A1_W,
                A2_W,
                A3_W,
                dW_dt,
            )
            apply_boundary_conditions(W, boundarypoints, boundaryvalues)
        is_converged = check_convergence(
            dW_dt, source_point, tol=tol, target_point=target_point
        )
        if is_converged:  # Hamiltonian throughout domain is sufficiently small
            print(f"Converged after {(n + 1) * n_check} steps!")
            break
    if not is_converged:
        print(f"Hamiltonian did not converge to tolerance {tol}!")

    # Compute gradient field: note that ||grad_cost W|| = 1 by Eikonal PDE.
    distance_gradient_field(
        W,
        cost,
        G_inv,
        dxy,
        dθ,
        θs,
        A1_forward,
        A1_backward,
        A2_forward,
        A2_backward,
        A3_forward,
        A3_backward,
        A1_W,
        A2_W,
        A3_W,
        grad_W,
    )

    # Cleanup
    W_np = W.to_numpy()
    grad_W_np = grad_W.to_numpy()

    return unpad_array(W_np, pad_shape=(1, 1, 0)), unpad_array(
        grad_W_np, pad_shape=(1, 1, 0, 0)
    )


def eikonal_solver_multi_source(
    cost_np,
    source_points,
    G_np,
    dxy,
    dθ,
    θs_np,
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
    Riemannian metric tensor field defined by `G_np` and `cost_np`, with source
    at `source_points`, using the iterative method first described by Bekkers et
    al.[2] and generalised in [1].

    Args:
        `cost_np`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1, with shape [Nx, Ny, Nθ].
        `source_points`: Tuple[Tuple[int]] describing index of source points in
          `cost_np`.
        `G_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs_np`: Orientation coordinate at every point in the grid on which
          `cost_np` is sampled.
      Optional:
        `target_point`: Tuple[int] describing index of target point in
          `cost_np`. Defaults to `None`. If `target_point` is provided, the
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
    # First compute for uniform cost to get initial W
    print("Solving Eikonal PDE with left invariant metric to compute initialisation.")
    W_init_np, _ = eikonal_solver_multi_source_uniform(
        cost_np.shape,
        source_points,
        G_np,
        dxy,
        dθ,
        θs_np,
        target_point=target_point,
        n_max=n_max_initialisation,
        n_check=n_check_initialisation,
        tol=tol,
        dε=dε,
        initial_condition=initial_condition,
    )

    print("Solving Eikonal PDE data-driven left invariant metric.")

    # Set hyperparameters
    G_inv = ti.Vector(invert_metric(G_np), ti.f32)
    # Heuristic, so that W does not become negative.
    # The sqrt(3) comes from the fact that the norm of the gradient consists of
    # 3 terms.
    ε = dε * (dxy / G_inv.max()) / np.sqrt(3)  # * cost_np.min()
    if n_check is None:  # Only check convergence at n_max
        n_check = n_max
    N_check = int(n_max / n_check)

    # Initialise Taichi objects
    cost = get_padded_cost(cost_np, pad_shape=((1,), (1,), (0,)))
    W = get_padded_cost(
        W_init_np, pad_shape=((1,), (1,), (0,)), pad_value=initial_condition
    )
    boundarypoints, boundaryvalues = get_boundary_conditions_multi_source(source_points)
    apply_boundary_conditions(W, boundarypoints, boundaryvalues)

    θs_np = np.pad(θs_np, ((1,), (1,), (0,)), mode="edge")
    θs = ti.field(dtype=ti.f32, shape=θs_np.shape)
    θs.from_numpy(θs_np)

    A1_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A1_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A2_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A2_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A3_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A3_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A1_W = ti.field(dtype=ti.f32, shape=W.shape)
    A2_W = ti.field(dtype=ti.f32, shape=W.shape)
    A3_W = ti.field(dtype=ti.f32, shape=W.shape)
    dW_dt = ti.field(dtype=ti.f32, shape=W.shape)
    grad_W = ti.Vector.field(n=3, dtype=ti.f32, shape=W.shape)

    # Compute approximate distance map
    is_converged = False
    for n in range(N_check):
        for _ in tqdm(range(int(n_check))):
            step_W(
                W,
                cost,
                G_inv,
                dxy,
                dθ,
                θs,
                ε,
                A1_forward,
                A1_backward,
                A2_forward,
                A2_backward,
                A3_forward,
                A3_backward,
                A1_W,
                A2_W,
                A3_W,
                dW_dt,
            )
            apply_boundary_conditions(W, boundarypoints, boundaryvalues)
        is_converged = check_convergence_multi_source(
            dW_dt, source_points, tol=tol, target_point=target_point
        )
        if is_converged:  # Hamiltonian throughout domain is sufficiently small
            print(f"Converged after {(n + 1) * n_check} steps!")
            break
    if not is_converged:
        print(f"Hamiltonian did not converge to tolerance {tol}!")

    # Compute gradient field: note that ||grad_cost W|| = 1 by Eikonal PDE.
    distance_gradient_field(
        W,
        cost,
        G_inv,
        dxy,
        dθ,
        θs,
        A1_forward,
        A1_backward,
        A2_forward,
        A2_backward,
        A3_forward,
        A3_backward,
        A1_W,
        A2_W,
        A3_W,
        grad_W,
    )

    # Cleanup
    W_np = W.to_numpy()
    grad_W_np = grad_W.to_numpy()

    return unpad_array(W_np, pad_shape=(1, 1, 0)), unpad_array(
        grad_W_np, pad_shape=(1, 1, 0, 0)
    )


@ti.kernel
def step_W(
    W: ti.template(),
    cost: ti.template(),
    G_inv: ti.types.vector(3, ti.f32),
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    ε: ti.f32,
    A1_forward: ti.template(),
    A1_backward: ti.template(),
    A2_forward: ti.template(),
    A2_backward: ti.template(),
    A3_forward: ti.template(),
    A3_backward: ti.template(),
    A1_W: ti.template(),
    A2_W: ti.template(),
    A3_W: ti.template(),
    dW_dt: ti.template(),
):
    """
    @taichi.kernel

    Update the (approximate) distance map `W` by a single step of the iterative
    method first described by Bekkers et al.[2] and generalised in [1].

    Args:
      Static:
        `cost`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of cost function.
        `G_inv`: ti.types.vector(n=3, dtype=[float]) of constants of the inverse
          of the diagonal metric tensor with respect to left invariant basis.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
        `ε`: "Time" step size, taking values greater than 0.
        `*_target`: Indices of the target point.
      Mutated:
        `W`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of approximate distance
          map, which is updated in place.
        `A*_*`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of derivatives.
        `A*_W`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of upwind derivative
          of the approximate distance map in the A* direction, which is updated
          in place.
        `dW_dt`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of error of the
          distance map with respect to the Eikonal PDE, which is updated in
          place.

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
    upwind_derivatives(
        W,
        dxy,
        dθ,
        θs,
        A1_forward,
        A1_backward,
        A2_forward,
        A2_backward,
        A3_forward,
        A3_backward,
        A1_W,
        A2_W,
        A3_W,
    )
    for I in ti.grouped(W):
        # It seems like TaiChi does not allow negative exponents.
        dW_dt[I] = (
            1
            - (
                ti.math.sqrt(
                    G_inv[0] * A1_W[I] ** 2
                    + G_inv[1] * A2_W[I] ** 2
                    + G_inv[2] * A3_W[I] ** 2
                )
                / cost[I]
            )
        ) * cost[I]
        W[I] += dW_dt[I] * ε


@ti.kernel
def distance_gradient_field(
    W: ti.template(),
    cost: ti.template(),
    G_inv: ti.types.vector(3, ti.f32),
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    A1_forward: ti.template(),
    A1_backward: ti.template(),
    A2_forward: ti.template(),
    A2_backward: ti.template(),
    A3_forward: ti.template(),
    A3_backward: ti.template(),
    A1_W: ti.template(),
    A2_W: ti.template(),
    A3_W: ti.template(),
    grad_W: ti.template(),
):
    """
    @taichi.kernel

    Compute the gradient with respect to `cost` of the (approximate) distance
    map `W`.

    Args:
      Static:
        `W`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of approximate distance
          map.
        `cost`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of cost function.
        `G_inv`: ti.types.vector(n=3, dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `A*_*`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of derivatives,
          which are updated in place.
        `A*_W`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of upwind derivative
          of the approximate distance map in the A* direction, which is updated
          in place.
        `grad_W`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ, 3]) of upwind
          derivatives of approximate distance map, which is updated inplace.
    """
    upwind_derivatives(
        W,
        dxy,
        dθ,
        θs,
        A1_forward,
        A1_backward,
        A2_forward,
        A2_backward,
        A3_forward,
        A3_backward,
        A1_W,
        A2_W,
        A3_W,
    )
    for I in ti.grouped(A1_W):
        grad_W[I] = (
            ti.Vector([G_inv[0] * A1_W[I], G_inv[1] * A2_W[I], G_inv[2] * A3_W[I]])
            / cost[I] ** 2
        )


# Left invariant


def eikonal_solver_uniform(
    domain_shape,
    source_point,
    G_np,
    dxy,
    dθ,
    θs_np,
    target_point=None,
    n_max=1e5,
    n_check=None,
    tol=1e-3,
    dε=1.0,
    initial_condition=100.0,
):
    """
    Solve the Eikonal PDE on M2 equipped with a datadriven left invariant
    metric tensor field defined by `G_np`, with source at `source_point`, using
    the iterative method first described by Bekkers et al.[2] and generalised in
    [1].

    Args:
        `domain_shape`: Tuple[int] describing the shape of the domain, namely
          [Nx, Ny, Nθ].
        `source_point`: Tuple[int] describing index of source point in
          `domain_shape`.
        `G_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs_np`: Orientation coordinate at every point in the grid on which
          `cost_np` is sampled.
      Optional:
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
    # Set hyperparameters.
    G_inv = ti.Vector(invert_metric(G_np), ti.f32)
    # Heuristic, so that W does not become negative.
    # The sqrt(3) comes from the fact that the norm of the gradient consists of
    # 3 terms.
    print(G_inv)
    ε = dε * (dxy / G_inv.max()) / np.sqrt(3)
    print(f"ε = {ε}")
    if n_check is None:  # Only check convergence at n_max
        n_check = n_max
    N_check = int(n_max / n_check)

    # Initialise Taichi objects
    W = get_initial_W(
        domain_shape, initial_condition=initial_condition, pad_shape=((1,), (1,), (0,))
    )
    boundarypoints, boundaryvalues = get_boundary_conditions(source_point)
    apply_boundary_conditions(W, boundarypoints, boundaryvalues)

    θs_np = np.pad(θs_np, ((1,), (1,), (0,)), mode="edge")
    θs = ti.field(dtype=ti.f32, shape=θs_np.shape)
    θs.from_numpy(θs_np)

    A1_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A1_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A2_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A2_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A3_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A3_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A1_W = ti.field(dtype=ti.f32, shape=W.shape)
    A2_W = ti.field(dtype=ti.f32, shape=W.shape)
    A3_W = ti.field(dtype=ti.f32, shape=W.shape)
    dW_dt = ti.field(dtype=ti.f32, shape=W.shape)
    grad_W = ti.Vector.field(n=3, dtype=ti.f32, shape=W.shape)

    # Compute approximate distance map
    is_converged = False
    for n in range(N_check):
        for _ in tqdm(range(int(n_check))):
            step_W_uniform(
                W,
                G_inv,
                dxy,
                dθ,
                θs,
                ε,
                A1_forward,
                A1_backward,
                A2_forward,
                A2_backward,
                A3_forward,
                A3_backward,
                A1_W,
                A2_W,
                A3_W,
                dW_dt,
            )
            apply_boundary_conditions(W, boundarypoints, boundaryvalues)
        is_converged = check_convergence(
            dW_dt, source_point, tol=tol, target_point=target_point
        )
        if is_converged:  # Hamiltonian throughout domain is sufficiently small
            print(f"Converged after {(n + 1) * n_check} steps!")
            break
    if not is_converged:
        print(f"Hamiltonian did not converge to tolerance {tol}!")

    # Compute gradient field: note that ||grad W|| = 1 by Eikonal PDE.
    distance_gradient_field_uniform(
        W,
        G_inv,
        dxy,
        dθ,
        θs,
        A1_forward,
        A1_backward,
        A2_forward,
        A2_backward,
        A3_forward,
        A3_backward,
        A1_W,
        A2_W,
        A3_W,
        grad_W,
    )

    # Cleanup
    W_np = W.to_numpy()
    grad_W_np = grad_W.to_numpy()

    return unpad_array(W_np, pad_shape=(1, 1, 0)), unpad_array(
        grad_W_np, pad_shape=(1, 1, 0, 0)
    )


def eikonal_solver_multi_source_uniform(
    domain_shape,
    source_points,
    G_np,
    dxy,
    dθ,
    θs_np,
    target_point=None,
    n_max=1e5,
    n_check=None,
    tol=1e-3,
    dε=1.0,
    initial_condition=100.0,
):
    """
    Solve the Eikonal PDE on M2 equipped with a datadriven left invariant
    metric tensor field defined by `G_np`, with source at `source_point`, using
    the iterative method first described by Bekkers et al.[2] and generalised in
    [1].

    Args:
        `domain_shape`: Tuple[int] describing the shape of the domain, namely
          [Nx, Ny, Nθ].
        `source_points`: Tuple[Tuple[int]] describing index of source points in
          `domain_shape`.
        `G_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs_np`: Orientation coordinate at every point in the grid on which
          `cost_np` is sampled.
      Optional:
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
    # Set hyperparameters.
    G_inv = ti.Vector(invert_metric(G_np), ti.f32)
    # Heuristic, so that W does not become negative.
    # The sqrt(3) comes from the fact that the norm of the gradient consists of
    # 3 terms.
    print(G_inv)
    ε = dε * (dxy / G_inv.max()) / np.sqrt(3)
    print(f"ε = {ε}")
    if n_check is None:  # Only check convergence at n_max
        n_check = n_max
    N_check = int(n_max / n_check)

    # Initialise Taichi objects
    W = get_initial_W(
        domain_shape, initial_condition=initial_condition, pad_shape=((1,), (1,), (0,))
    )
    boundarypoints, boundaryvalues = get_boundary_conditions_multi_source(source_points)
    apply_boundary_conditions(W, boundarypoints, boundaryvalues)

    θs_np = np.pad(θs_np, ((1,), (1,), (0,)), mode="edge")
    θs = ti.field(dtype=ti.f32, shape=θs_np.shape)
    θs.from_numpy(θs_np)

    A1_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A1_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A2_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A2_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A3_forward = ti.field(dtype=ti.f32, shape=W.shape)
    A3_backward = ti.field(dtype=ti.f32, shape=W.shape)
    A1_W = ti.field(dtype=ti.f32, shape=W.shape)
    A2_W = ti.field(dtype=ti.f32, shape=W.shape)
    A3_W = ti.field(dtype=ti.f32, shape=W.shape)
    dW_dt = ti.field(dtype=ti.f32, shape=W.shape)
    grad_W = ti.Vector.field(n=3, dtype=ti.f32, shape=W.shape)

    # Compute approximate distance map
    is_converged = False
    for n in range(N_check):
        for _ in tqdm(range(int(n_check))):
            step_W_uniform(
                W,
                G_inv,
                dxy,
                dθ,
                θs,
                ε,
                A1_forward,
                A1_backward,
                A2_forward,
                A2_backward,
                A3_forward,
                A3_backward,
                A1_W,
                A2_W,
                A3_W,
                dW_dt,
            )
            apply_boundary_conditions(W, boundarypoints, boundaryvalues)
        is_converged = check_convergence_multi_source(
            dW_dt, source_points, tol=tol, target_point=target_point
        )
        if is_converged:  # Hamiltonian throughout domain is sufficiently small
            print(f"Converged after {(n + 1) * n_check} steps!")
            break
    if not is_converged:
        print(f"Hamiltonian did not converge to tolerance {tol}!")

    # Compute gradient field: note that ||grad W|| = 1 by Eikonal PDE.
    distance_gradient_field_uniform(
        W,
        G_inv,
        dxy,
        dθ,
        θs,
        A1_forward,
        A1_backward,
        A2_forward,
        A2_backward,
        A3_forward,
        A3_backward,
        A1_W,
        A2_W,
        A3_W,
        grad_W,
    )

    # Cleanup
    W_np = W.to_numpy()
    grad_W_np = grad_W.to_numpy()

    return unpad_array(W_np, pad_shape=(1, 1, 0)), unpad_array(
        grad_W_np, pad_shape=(1, 1, 0, 0)
    )


@ti.kernel
def step_W_uniform(
    W: ti.template(),
    G_inv: ti.types.vector(3, ti.f32),
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    ε: ti.f32,
    A1_forward: ti.template(),
    A1_backward: ti.template(),
    A2_forward: ti.template(),
    A2_backward: ti.template(),
    A3_forward: ti.template(),
    A3_backward: ti.template(),
    A1_W: ti.template(),
    A2_W: ti.template(),
    A3_W: ti.template(),
    dW_dt: ti.template(),
):
    """
    @taichi.kernel

    Update the (approximate) distance map `W` by a single step of the iterative
    method first described by Bekkers et al.[2] and generalised in [1].

    Args:
      Static:
        `G_inv`: ti.types.vector(n=3, dtype=[float]) of constants of the
          diagonal metric tensor with respect to left invariant basis.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
        `ε`: "Time" step size, taking values greater than 0.
        `*_target`: Indices of the target point.
      Mutated:
        `W`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of approximate distance
          map, which is updated in place.
        `A*_*`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of derivatives.
        `A*_W`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of upwind derivative
          of the approximate distance map in the A* direction, which is updated
          in place.
        `dW_dt`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of error of the
          distance map with respect to the Eikonal PDE, which is updated in
          place.

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
    upwind_derivatives(
        W,
        dxy,
        dθ,
        θs,
        A1_forward,
        A1_backward,
        A2_forward,
        A2_backward,
        A3_forward,
        A3_backward,
        A1_W,
        A2_W,
        A3_W,
    )
    for I in ti.grouped(W):
        # It seems like TaiChi does not allow negative exponents.
        dW_dt[I] = 1 - ti.math.sqrt(
            G_inv[0] * A1_W[I] ** 2 + G_inv[1] * A2_W[I] ** 2 + G_inv[2] * A3_W[I] ** 2
        )
        W[I] += dW_dt[I] * ε


@ti.kernel
def distance_gradient_field_uniform(
    W: ti.template(),
    G_inv: ti.types.vector(3, ti.f32),
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    A1_forward: ti.template(),
    A1_backward: ti.template(),
    A2_forward: ti.template(),
    A2_backward: ti.template(),
    A3_forward: ti.template(),
    A3_backward: ti.template(),
    A1_W: ti.template(),
    A2_W: ti.template(),
    A3_W: ti.template(),
    grad_W: ti.template(),
):
    """
    @taichi.kernel

    Compute the gradient of the (approximate) distance map `W`.

    Args:
      Static:
        `W`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of approximate distance
          map.
        `G_inv`: ti.types.vector(n=3, dtype=[float]) of constants of the inverse
          of the diagonal metric tensor with respect to left invariant basis.
        `dxy`: Spatial step size, taking values greater than 0.
        `dθ`: Orientational step size, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `A*_*`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of derivatives,
          which are updated in place.
        `A*_W`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of upwind derivative
          of the approximate distance map in the A* direction, which is updated
          in place.
        `grad_W`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ, 3]) of upwind
          derivatives of approximate distance map, which is updated inplace.
    """
    upwind_derivatives(
        W,
        dxy,
        dθ,
        θs,
        A1_forward,
        A1_backward,
        A2_forward,
        A2_backward,
        A3_forward,
        A3_backward,
        A1_W,
        A2_W,
        A3_W,
    )
    for I in ti.grouped(A1_W):
        grad_W[I] = ti.Vector(
            [G_inv[0] * A1_W[I], G_inv[1] * A2_W[I], G_inv[2] * A3_W[I]]
        )
