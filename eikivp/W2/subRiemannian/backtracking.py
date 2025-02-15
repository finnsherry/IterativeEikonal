"""
    backtracking
    ============

    Provides methods to compute the geodesic, with respect to some distance map,
    connecting two points in W2. The primary methods are:
      1. `geodesic_back_tracking`: compute the geodesic using gradient descent.
      The gradient must be provided; it is computed along with the distance map
      by the corresponding methods in the distancemap module.
"""

import numpy as np
import h5py
import taichi as ti
from eikivp.W2.subRiemannian.interpolate import vectorfield_trilinear_interpolate_LI
from eikivp.W2.utils import (
    get_next_point,
    coordinate_array_to_real,
    coordinate_real_to_array_ti,
    vector_LI_to_static,
    distance_in_pixels,
    distance_in_pixels_multi_source
)

def import_γ_path(params, folder):
    """
    Import the geodesic matching `params`.
    """
    cost_domain = params["cost_domain"]
    image_name = params["image_name"]
    λ = params["λ"]
    p = params["p"]
    ξ = params["ξ"]
    target_point = params["target_point"]
    if "dt" in params:
        dt = params["dt"]
    else:
        dt = "default"
    if cost_domain == "M2":
        σ_s_list = params["σ_s_list"]
        σ_o = params["σ_o"]
        σ_s_ext = params["σ_s_ext"]
        σ_o_ext = params["σ_o_ext"]
        geodesic_filename = f"{folder}\\W2_sR_ss_s={[s for s in σ_s_list]}_s_o={σ_o}_s_s_e={σ_s_ext}_s_o_e={σ_o_ext}_l={λ}_p={p}_x={ξ}_t={target_point}.hdf5"
        with h5py.File(geodesic_filename, "r") as geodesic_file:
            if "source_point" in params:
                source_point = params["source_point"]
                assert (
                    np.all(σ_s_list == geodesic_filename.attrs["σ_s_list"]) and
                    σ_o == geodesic_filename.attrs["σ_o"] and
                    σ_s_ext == geodesic_filename.attrs["σ_s_ext"] and
                    σ_o_ext == geodesic_filename.attrs["σ_o_ext"] and
                    image_name == geodesic_file.attrs["image_name"] and
                    λ == geodesic_file.attrs["λ"] and
                    p == geodesic_file.attrs["p"] and
                    ξ == geodesic_file.attrs["ξ"] and
                    np.all(source_point == geodesic_file.attrs["source_point"]) and
                    np.all(target_point == geodesic_file.attrs["target_point"]) and
                    dt == geodesic_file.attrs["dt"]
                ), "There is a parameter mismatch!"
            elif "source_points" in params:
                source_points = params["source_points"]
                assert (
                    np.all(σ_s_list == geodesic_filename.attrs["σ_s_list"]) and
                    σ_o == geodesic_filename.attrs["σ_o"] and
                    σ_s_ext == geodesic_filename.attrs["σ_s_ext"] and
                    σ_o_ext == geodesic_filename.attrs["σ_o_ext"] and
                    image_name == geodesic_file.attrs["image_name"] and
                    λ == geodesic_file.attrs["λ"] and
                    p == geodesic_file.attrs["p"] and
                    ξ == geodesic_file.attrs["ξ"] and
                    np.all(source_points == geodesic_file.attrs["source_points"]) and
                    np.all(target_point == geodesic_file.attrs["target_point"]) and
                    dt == geodesic_file.attrs["dt"]
                ), "There is a parameter mismatch!"
            γ_path = geodesic_file["Geodesic"][()]
    else:
        scales = params["scales"]
        α = params["α"]
        γ = params["γ"]
        ε = params["ε"]
        geodesic_filename = f"{folder}\\W2_sR_ss={[s for s in scales]}_a={α}_g={γ}_e={ε}_l={λ}_p={p}_x={ξ}_t={target_point}.hdf5"
        with h5py.File(geodesic_filename, "r") as geodesic_file:
            if "source_point" in params:
                source_point = params["source_point"]
                assert (
                    np.all(scales == geodesic_file.attrs["scales"]) and
                    α == geodesic_file.attrs["α"] and
                    γ == geodesic_file.attrs["γ"] and
                    ε == geodesic_file.attrs["ε"] and
                    image_name == geodesic_file.attrs["image_name"] and
                    λ == geodesic_file.attrs["λ"] and
                    p == geodesic_file.attrs["p"] and
                    ξ == geodesic_file.attrs["ξ"] and
                    np.all(source_point == geodesic_file.attrs["source_point"]) and
                    np.all(target_point == geodesic_file.attrs["target_point"]) and
                    dt == geodesic_file.attrs["dt"]
                ), "There is a parameter mismatch!"
            elif "source_points" in params:
                source_points = params["source_points"]
                assert (
                    np.all(scales == geodesic_file.attrs["scales"]) and
                    α == geodesic_file.attrs["α"] and
                    γ == geodesic_file.attrs["γ"] and
                    ε == geodesic_file.attrs["ε"] and
                    image_name == geodesic_file.attrs["image_name"] and
                    λ == geodesic_file.attrs["λ"] and
                    p == geodesic_file.attrs["p"] and
                    ξ == geodesic_file.attrs["ξ"] and
                    np.all(source_points == geodesic_file.attrs["source_points"]) and
                    np.all(target_point == geodesic_file.attrs["target_point"]) and
                    dt == geodesic_file.attrs["dt"]
                ), "There is a parameter mismatch!"
            γ_path = geodesic_file["Geodesic"][()]
    return γ_path

def export_γ_path(γ_path, params, folder):
    """
    Export the geodesic to hdf5 with attributes `params`.
    """
    cost_domain = params["cost_domain"]
    image_name = params["image_name"]
    λ = params["λ"]
    p = params["p"]
    ξ = params["ξ"]
    target_point = params["target_point"]
    if "dt" in params:
        dt = params["dt"]
    else:
        dt = "default"
    if cost_domain == "M2":
        σ_s_list = params["σ_s_list"]
        σ_o = params["σ_o"]
        σ_s_ext = params["σ_s_ext"]
        σ_o_ext = params["σ_o_ext"]
        geodesic_filename = f"{folder}\\W2_sR_ss_s={[s for s in σ_s_list]}_s_o={σ_o}_s_s_e={σ_s_ext}_s_o_e={σ_o_ext}_l={λ}_p={p}_x={ξ}_t={target_point}.hdf5"
        with h5py.File(geodesic_filename, "w") as geodesic_file:
            geodesic_file.create_dataset("Geodesic", data=γ_path)
            geodesic_file.attrs["σ_s_list"] = σ_s_list
            geodesic_file.attrs["σ_o"] = σ_o
            geodesic_file.attrs["σ_s_ext"] = σ_s_ext
            geodesic_file.attrs["σ_o_ext"] = σ_o_ext
            geodesic_file.attrs["image_name"] = image_name
            geodesic_file.attrs["λ"] = λ
            geodesic_file.attrs["p"] = p
            geodesic_file.attrs["ξ"] = ξ
            geodesic_file.attrs["target_point"] = target_point
            geodesic_file.attrs["dt"] = dt
            if "source_point" in params:
                geodesic_file.attrs["source_point"] = params["source_point"]
            else:
                geodesic_file.attrs["source_points"] = params["source_points"]
    else:
        scales = params["scales"]
        α = params["α"]
        γ = params["γ"]
        ε = params["ε"]
        geodesic_filename = f"{folder}\\W2_sR_ss={[s for s in scales]}_a={α}_g={γ}_e={ε}_l={λ}_p={p}_x={ξ}_t={target_point}.hdf5"
        with h5py.File(geodesic_filename, "w") as geodesic_file:
            geodesic_file.create_dataset("Geodesic", data=γ_path)
            geodesic_file.attrs["scales"] = scales
            geodesic_file.attrs["α"] = α
            geodesic_file.attrs["γ"] = γ
            geodesic_file.attrs["ε"] = ε
            geodesic_file.attrs["image_name"] = image_name
            geodesic_file.attrs["λ"] = λ
            geodesic_file.attrs["p"] = p
            geodesic_file.attrs["ξ"] = ξ
            geodesic_file.attrs["target_point"] = target_point
            geodesic_file.attrs["dt"] = dt
            if "source_point" in params:
                geodesic_file.attrs["source_point"] = params["source_point"]
            else:
                geodesic_file.attrs["source_points"] = params["source_points"]

# Sub-Riemannian backtracking

def geodesic_back_tracking(grad_W_np, source_point, target_point, cost_np, α_min, β_min, φ_min, dα, dβ, dφ, αs_np,
                           φs_np, ξ, dt=1., n_max=10000):
    """
    Find the geodesic connecting `target_point` to `source_point`, using 
    gradient descent back tracking, as first described by Bekkers et al.[2] and
    generalised in [1].

    Args:
        `grad_W_np`: np.ndarray of upwind gradient with respect to some cost of 
          the approximate distance map, with shape [Nα, Nβ, Nφ, 3].
        `source_point`: Tuple[int] describing index of source point in `W_np`.
        `target_point`: Tuple[int] describing index of target point in `W_np`.
        `cost_np`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1, with shape [Nα, Nβ, Nφ].
        `α_min`: minimum value of α-coordinates in rectangular domain.
        `β_min`: minimum value of β-coordinates in rectangular domain.
        `φ_min`: minimum value of φ-coordinates in rectangular domain.
        `dα`: spatial resolution in the α-direction, taking values greater than
          0.
        `dβ`: spatial resolution in the β-direction, taking values greater than
          0.
        `dφ`: orientational resolution, taking values greater than 0.
        `αs_np`: α-coordinate at every point in the grid on which `cost_np` is
          sampled.
        `φs_np`: Orientation coordinate at every point in the grid on which
          `cost_np` is sampled.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
      Optional:
        `dt`: Step size, taking values greater than 0. Defaults to 1.
        `n_max`: Maximum number of points in geodesic, taking positive integral
          values. Defaults to 10000.

    Returns:
        np.ndarray of geodesic connecting `target_point` to `source_point`.
    
    References:
        [1]: N.J. van den Berg, F.M. Sherry, T.T.J.M. Berendschot, and R. Duits.
          "Crossing-Preserving Geodesic Tracking on Spherical Images."
          In: Scale Space and Variational Methods in Computer Vision (2025),
          pp. .
          DOI:.
        [2]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    # Set hyperparameters
    shape = grad_W_np.shape[0:-1]

    # Initialise Taichi objects
    grad_W = ti.Vector.field(n=3, dtype=ti.f32, shape=shape)
    grad_W.from_numpy(grad_W_np)
    cost = ti.field(dtype=ti.f32, shape=shape)
    cost.from_numpy(cost_np)
    source_point = coordinate_array_to_real(*source_point, α_min, β_min, φ_min, dα, dβ, dφ)
    target_point = coordinate_array_to_real(*target_point, α_min, β_min, φ_min, dα, dβ, dφ)
    source_point = ti.Vector(source_point, dt=ti.f32)
    target_point = ti.Vector(target_point, dt=ti.f32)
    αs = ti.field(dtype=ti.f32, shape=αs_np.shape)
    αs.from_numpy(αs_np)
    φs = ti.field(dtype=ti.f32, shape=φs_np.shape)
    φs.from_numpy(φs_np)

    # Perform backtracking
    γ = ti.Vector.field(n=3, dtype=ti.f32, shape=n_max)

    point = target_point
    γ[0] = point
    tol = 2. # Stop if we are within two pixels of the source.
    n = 1
    distance = ti.math.inf
    while (distance >= tol) and (n < n_max - 1):
        point = geodesic_back_tracking_step(grad_W, point, αs, φs, ξ, cost, α_min, β_min, φ_min, dα, dβ, dφ, dt)
        distance = distance_in_pixels(point, source_point, dα, dβ, dφ)
        γ[n] = point
        n += 1
    γ_len = n
    print(f"Geodesic consists of {γ_len} points.")
    γ_np = γ.to_numpy()[:γ_len]
    γ_np[-1] = source_point
    return γ_np

def geodesic_back_tracking_multi_source(grad_W_np, source_points, target_point, cost_np, α_min, β_min, φ_min, dα, dβ,
                                        dφ, αs_np, φs_np, ξ, dt=1., n_max=10000):
    """
    Find the geodesic connecting `target_point` to `source_point`, using 
    gradient descent back tracking, as first described by Bekkers et al.[2] and
    generalised in [1].

    Args:
        `grad_W_np`: np.ndarray of upwind gradient with respect to some cost of 
          the approximate distance map, with shape [Nα, Nβ, Nφ, 3].
        `source_points`: Tuple[Tuple[int]] describing index of source point in
          `W_np`.
        `target_point`: Tuple[int] describing index of target point in `W_np`.
        `cost_np`: np.ndarray of cost function throughout domain, taking values
          between 0 and 1, with shape [Nα, Nβ, Nφ].
        `α_min`: minimum value of α-coordinates in rectangular domain.
        `β_min`: minimum value of β-coordinates in rectangular domain.
        `φ_min`: minimum value of φ-coordinates in rectangular domain.
        `dα`: spatial resolution in the α-direction, taking values greater than
          0.
        `dβ`: spatial resolution in the β-direction, taking values greater than
          0.
        `dφ`: orientational resolution, taking values greater than 0.
        `αs_np`: α-coordinate at every point in the grid on which `cost_np` is
          sampled.
        `φs_np`: Orientation coordinate at every point in the grid on which
          `cost_np` is sampled.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
      Optional:
        `dt`: Step size, taking values greater than 0. Defaults to 1.
        `n_max`: Maximum number of points in geodesic, taking positive integral
          values. Defaults to 10000.

    Returns:
        np.ndarray of geodesic connecting `target_point` to `source_point`.
    
    References:
        [1]: N.J. van den Berg, F.M. Sherry, T.T.J.M. Berendschot, and R. Duits.
          "Crossing-Preserving Geodesic Tracking on Spherical Images."
          In: Scale Space and Variational Methods in Computer Vision (2025),
          pp. .
          DOI:.
        [2]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    # Set hyperparameters
    shape = grad_W_np.shape[0:-1]

    # Initialise Taichi objects
    grad_W = ti.Vector.field(n=3, dtype=ti.f32, shape=shape)
    grad_W.from_numpy(grad_W_np)
    cost = ti.field(dtype=ti.f32, shape=shape)
    cost.from_numpy(cost_np)
    # We perform backtracking in real coordinates instead of in array indices.
    source_points_np = np.array(tuple(coordinate_array_to_real(*p, α_min, β_min, φ_min, dα, dβ, dφ) for p in source_points))
    N_source_points = len(source_points)
    source_points = ti.Vector.field(n=3, shape=(N_source_points,), dtype=ti.f32)
    source_points.from_numpy(source_points_np)
    target_point = coordinate_array_to_real(*target_point, α_min, β_min, φ_min, dα, dβ, dφ)
    target_point = ti.Vector(target_point, dt=ti.f32)
    αs = ti.field(dtype=ti.f32, shape=αs_np.shape)
    αs.from_numpy(αs_np)
    φs = ti.field(dtype=ti.f32, shape=φs_np.shape)
    φs.from_numpy(φs_np)

    # Perform backtracking
    γ = ti.Vector.field(n=3, dtype=ti.f32, shape=n_max)
    distances = ti.field(dtype=ti.f32, shape=(N_source_points,))

    point = target_point
    γ[0] = point
    tol = 2. # Stop if we are within two pixels of the source.
    n = 1
    min_distance = ti.math.inf
    while (min_distance >= tol) and (n < n_max - 1):
        point = geodesic_back_tracking_step(grad_W, point, αs, φs, ξ, cost, α_min, β_min, φ_min, dα, dβ, dφ, dt)
        min_distance = distance_in_pixels_multi_source(point, source_points, distances, dα, dβ, dφ)
        γ[n] = point
        n += 1
    γ_len = n
    print(f"Geodesic consists of {γ_len} points.")
    γ_np = γ.to_numpy()[:γ_len]
    distances = distances.to_numpy()
    γ_np[-1] = source_points_np[np.argmin(distances)]
    return γ_np

@ti.kernel
def geodesic_back_tracking_step(
    grad_W: ti.template(),
    point: ti.types.vector(3, ti.f32),
    αs: ti.template(),
    φs: ti.template(),
    ξ: ti.f32,
    cost: ti.template(),
    α_min: ti.f32,
    β_min: ti.f32,
    φ_min: ti.f32,
    dα: ti.f32,
    dβ: ti.f32,
    dφ: ti.f32,
    dt: ti.f32
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.kernel

    Find the geodesic connecting `target_point` to `source_point`, using 
    gradient descent back tracking, as first described by Bekkers et al.[2] and
    generalised in [1].

    Args:
      Static:
        `grad_W`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ, 3]) of upwind
          gradient with respect to some cost of the approximate distance map.
        `point`: ti.types.vector(n=3, dtype=[float]) current point.
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point.
        `ξ`: Stiffness of moving in the A1 direction compared to the A3
          direction, taking values greater than 0.
        `cost`: ti.field(dtype=[float], shape=[Nα, Nβ, Nφ]) of cost function,
          taking values between 0 and 1.
        `α_min`: minimum value of α-coordinates in rectangular domain.
        `β_min`: minimum value of β-coordinates in rectangular domain.
        `φ_min`: minimum value of φ-coordinates in rectangular domain.
        `dα`: spatial resolution in the α-direction, taking values greater than
          0.
        `dβ`: spatial resolution in the β-direction, taking values greater than
          0.
        `dφ`: orientational resolution, taking values greater than 0.
        `dt`: Gradient descent step size, taking values greater than 0.

    Returns:
        Next point.
    
    References:
        [1]: N.J. van den Berg, F.M. Sherry, T.T.J.M. Berendschot, and R. Duits.
          "Crossing-Preserving Geodesic Tracking on Spherical Images."
          In: Scale Space and Variational Methods in Computer Vision (2025),
          pp. .
          DOI:.
        [2]: E. J. Bekkers, R. Duits, A. Mashtakov, and G. R. Sanguinetti.
          "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in SE(2)".
          In: SIAM Journal on Imaging Sciences 8.4 (2015), pp. 2740--2770.
          DOI:10.1137/15M1018460.
    """
    # To get the gradient, we need the corresponding array indices.
    point_array = coordinate_real_to_array_ti(point, α_min, β_min, φ_min, dα, dβ, dφ)
    # Get gradient using componentwise trilinear interpolation.
    gradient_at_point_LI = vectorfield_trilinear_interpolate_LI(grad_W, point_array, ξ, cost)
    α = point[0]
    φ = point[2]
    # Get gradient with respect to static frame.
    gradient_at_point = vector_LI_to_static(gradient_at_point_LI, α, φ)
    new_point = get_next_point(point, gradient_at_point, dα, dβ, dφ, dt)
    return new_point