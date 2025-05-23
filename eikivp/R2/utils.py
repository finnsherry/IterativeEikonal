"""
    utils
    =====

    Provides miscellaneous computational utilities that can be used on R^2.
"""

import numpy as np
import taichi as ti

# Safe Indexing

@ti.func
def sanitize_index(
    index: ti.types.vector(2, ti.i32),
    input: ti.template()
) -> ti.types.vector(2, ti.i32):
    """
    @taichi.func
    
    Make sure the `index` is inside the shape of `input`. Adapted from Gijs.

    Args:
        `index`: ti.types.vector(n=2, dtype=ti.i32) index.
        `input`: ti.field in which we want to index.

    Returns:
        ti.types.vector(n=2, dtype=ti.i32) of index that is within `input`.
    """
    shape = ti.Vector(ti.static(input.shape), dt=ti.i32)
    return ti.Vector([
        ti.math.clamp(index[0], 0, shape[0] - 1),
        ti.math.clamp(index[1], 0, shape[1] - 1),
    ], dt=ti.i32)

# Distance Map

def get_boundary_conditions(source_point):
    """
    Determine the boundary conditions from `source_point`, giving the boundary
    points and boundary values as TaiChi objects.
    """
    i_0, j_0 = source_point
    boundarypoints_np = np.array([[i_0 + 1, j_0 + 1]], dtype=int) # Account for padding.
    boundaryvalues_np = np.array([0.], dtype=float)
    boundarypoints = ti.Vector.field(n=2, dtype=ti.i32, shape=1)
    boundarypoints.from_numpy(boundarypoints_np)
    boundaryvalues = ti.field(shape=1, dtype=ti.f32)
    boundaryvalues.from_numpy(boundaryvalues_np)
    return boundarypoints, boundaryvalues

def get_boundary_conditions_multi_source(source_points):
    """
    Determine the boundary conditions from `source_point`, giving the boundary
    points and boundary values as TaiChi objects.
    """
    N_points = len(source_points)
    boundarypoints_np = np.array([[i_0 + 1, j_0 + 1] for (i_0, j_0) in source_points], dtype=int) # Account for padding.
    boundaryvalues_np = np.array([0.] * N_points, dtype=float)
    boundarypoints = ti.Vector.field(n=2, dtype=ti.i32, shape=N_points)
    boundarypoints.from_numpy(boundarypoints_np)
    boundaryvalues = ti.field(shape=N_points, dtype=ti.f32)
    boundaryvalues.from_numpy(boundaryvalues_np)
    return boundarypoints, boundaryvalues

@ti.kernel
def field_abs_max(
    scalar_field: ti.template()
) -> ti.f32:
    """
    @taichi.kernel

    Find the largest absolute value in `scalar_field`.

    Args:
        `scalar_field`: ti.field(dtype=[float], shape=shape) of 2D scalar field.

    Returns:
        Largest absolute value in `scalar_field`.
    """
    value = ti.abs(scalar_field[0, 0])
    for I in ti.grouped(scalar_field):
        ti.atomic_max(value, ti.abs(scalar_field[I]))
    return value

def check_convergence(dW_dt, source_point, tol=1e-3, target_point=None):
    """
    Check whether the IVP method has converged by comparing the Hamiltonian
    `dW_dt` to tolerance `tol`. If `target_point` is provided, only check
    convergence at `target_point`; otherwise check throughout the domain.
    """
    if target_point is None:
        dW_dt[source_point[0]+1, source_point[1]+1] = 0. # Source is fixed.
        error = field_abs_max(dW_dt)
    else:
        error = ti.abs(dW_dt[target_point])
    print(error)
    is_converged = error < tol
    return is_converged

def check_convergence_multi_source(dW_dt, source_points, tol=1e-3, target_point=None):
    """
    Check whether the IVP method has converged by comparing the Hamiltonian
    `dW_dt` to tolerance `tol`. If `target_point` is provided, only check
    convergence at `target_point`; otherwise check throughout the domain.
    """
    if target_point is None:
        for i_0, j_0 in source_points:
            dW_dt[i_0+1, j_0+1] = 0. # Source is fixed.
        error = field_abs_max(dW_dt)
    else:
        error = ti.abs(dW_dt[target_point])
    print(error)
    is_converged = error < tol
    return is_converged

# Coordinate Transforms

def coordinate_real_to_array(x, y, x_min, y_min, dxy):
    """
    Compute the array indices (I, J) of the point defined by real coordinates 
    (`x`, `y`). Can broadcast over entire arrays of real coordinates.

    Args:
        `x`: x-coordinate of the point.
        `y`: y-coordinate of the point.
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
    """
    I = np.rint((x - x_min) / dxy).astype(int)
    J = np.rint((y - y_min) / dxy).astype(int)
    return I, J

@ti.func
def coordinate_real_to_array_ti(
    point: ti.types.vector(2, ti.f32),
    x_min: ti.f32,
    y_min: ti.f32,
    dxy: ti.f32
) -> ti.types.vector(2, ti.f32):
    """
    @taichi.func

    Compute the array indices (I, J) of the point defined by real coordinates 
    `point`. Can broadcast over entire arrays of real coordinates.

    Args:
        `point`: vector of x- and y-coordinates of the point.
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
    """
    I = (point[0] - x_min) / dxy
    J = (point[1] - y_min) / dxy
    return ti.Vector([I, J], dt=ti.f32)

def coordinate_array_to_real(I, J, x_min, y_min, dxy):
    """
    Compute the real coordinates (x, y) of the point defined by array indices 
    (`I`, `J`). Can broadcast over entire arrays of array indices.

    Args:
        `I`: I index of the point.
        `J`: J index of the point.
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
    """
    x = x_min + I * dxy
    y = y_min + J * dxy
    return x, y

def align_to_real_axis_point(point, shape):
    """
    Express `point`, given in indices with respect to standard array convention,
    in terms of indices with respect to arrays aligned with real axes (see Notes
    for more explanation). Here, `shape` gives the shape of the array in which
    we index _after_ aligning with real axes, so [Nx, Ny].

    Args:
        `point`: Tuple[int, int] describing point with respect to standard array
          indexing convention.
        `shape`: shape of array, aligned to real axes, in which we want to
          index. Note that `0 <= point[0] <= shape[1] - 1` and 
          `0 <= point[1] <= shape[0] - 1`.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up, you do so by decreasing I, while if you want to move a single
        pixel to the right, you do so by increasing J. Hence, the shape of the
        array is [Ny, Nx]

        When aligned with real axes, moving up a single pixel is achieved by
        increasing J, and moving right a single pixel is achieved by increasing
        I. Hence, the shape of the array is [Nx, Ny].

        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
               standard                  real axes aligned
            I ^ ------                    I x ------
            | | |    |        =>          | | |    |
            v y ------                    v v ------
                 x ->                          y ->
                 J ->                          J ->  
    """
    return point[1], shape[1] - 1 - point[0]

def align_to_real_axis_scalar_field(field):
    """
    Align `field`, given in indices with respect to standard array convention, 
    with real axes (see Notes for more explanation).

    Args:
        `field`: np.ndarray of scalar field given with respect to standard array
          convention.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up, you do so by decreasing I, while if you want to move a single
        pixel to the right, you do so by increasing J. Hence, the shape of the
        array is [Ny, Nx]

        When aligned with real axes, moving up a single pixel is achieved by
        increasing J, and moving right a single pixel is achieved by increasing
        I. Hence, the shape of the array is [Nx, Ny].

        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
               standard                  real axes aligned
            I ^ ------                    I x ------
            | | |    |        =>          | | |    |
            v y ------                    v v ------
                 x ->                          y ->
                 J ->                          J ->  
    """
    field_flipped = np.flip(field, axis=0)
    field_aligned = np.transpose(field_flipped, axes=(1, 0))
    return field_aligned

def align_to_real_axis_vector_field(vector_field):
    """
    Align `vector_field`, given in indices with respect to standard array 
    convention, with real axes (see Notes for more explanation).
    
    Args:
        `vector_field`: np.ndarray of vector field given with respect to 
          standard array convention.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up, you do so by decreasing I, while if you want to move a single
        pixel to the right, you do so by increasing J. Hence, the shape of the
        array is [Ny, Nx]

        When aligned with real axes, moving up a single pixel is achieved by
        increasing J, and moving right a single pixel is achieved by increasing
        I. Hence, the shape of the array is [Nx, Ny].

        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
               standard                  real axes aligned
            I ^ ------                    I x ------
            | | |    |        =>          | | |    |
            v y ------                    v v ------
                 x ->                          y ->
                 J ->                          J ->  
    """
    vector_field_flipped = np.flip(vector_field, axis=0)
    vector_field_aligned = np.transpose(vector_field_flipped, axes=(1, 0, 2))
    return vector_field_aligned

def align_to_standard_array_axis_point(point, shape):
    """
    Express `point`, given in indices with respect to arrays aligned with real 
    axes, in terms of indices with respect to standard array convention, (see 
    Notes for more explanation). Here, `shape` gives the shape of the array in 
    which we index _after_ aligning with standard array convention, so [Ny, Nx].

    Args:
        `point`: Tuple[int, int] describing point with respect to arrays aligned
          with real axes.
        `shape`: shape of array, with respect to standard array convention, in 
          which we want to index. Note that `0 <= point[0] <= shape[1] - 1` and 
          `0 <= point[1] <= shape[0] - 1`.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up, you do so by decreasing I, while if you want to move a single
        pixel to the right, you do so by increasing J. Hence, the shape of the
        array is [Ny, Nx]

        When aligned with real axes, moving up a single pixel is achieved by
        increasing J, and moving right a single pixel is achieved by increasing
        I. Hence, the shape of the array is [Nx, Ny].

        Alignment is achieved by first transposing and subsequently flipping the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
           real axes aligned                 standard
            I x ------                    I ^ ------
            | | |    |        =>          | | |    |
            v v ------                    v y ------
                 y ->                          x ->
                 J ->                          J ->  
    """
    return point[1], shape[1] - 1 - point[0]

def align_to_standard_array_axis_scalar_field(field):
    """
    Align `field`, given in indices with respect to arrays aligned with real
    axes, with respect to standard array convention (see Notes for more 
    explanation).

    Args:
        `field`: np.ndarray of scalar field given in indices with respect to
          arrays aligned with real axes.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up, you do so by decreasing I, while if you want to move a single
        pixel to the right, you do so by increasing J. Hence, the shape of the
        array is [Ny, Nx]

        When aligned with real axes, moving up a single pixel is achieved by
        increasing J, and moving right a single pixel is achieved by increasing
        I. Hence, the shape of the array is [Nx, Ny].

        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
           real axes aligned                 standard
            I x ------                    I ^ ------
            | | |    |        =>          | | |    |
            v v ------                    v y ------
                 y ->                          x ->
                 J ->                          J ->  
    """
    field_transposed = np.transpose(field, axes=(1, 0))
    field_aligned = np.flip(field_transposed, axis=0)
    return field_aligned

def align_to_standard_array_axis_vector_field(vector_field):
    """
    Align `vector_field`, given in with respect to standard array convention, 
    with real axes (see Notes for more explanation).

    Args:
        `vector_field`: np.ndarray of vector field given in indices with respect
          to arrays aligned with real axes.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up, you do so by decreasing I, while if you want to move a single
        pixel to the right, you do so by increasing J. Hence, the shape of the
        array is [Ny, Nx]

        When aligned with real axes, moving up a single pixel is achieved by
        increasing J, and moving right a single pixel is achieved by increasing
        I. Hence, the shape of the array is [Nx, Ny].
        
        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
           real axes aligned                 standard
            I x ------                    I ^ ------
            | | |    |        =>          | | |    |
            v v ------                    v y ------
                 y ->                          x ->
                 J ->                          J ->  
    """
    vector_field_transposed = np.transpose(vector_field, axes=(1, 0, 2))
    vector_field_aligned = np.flip(vector_field_transposed, axis=0)
    return vector_field_aligned