"""
    interpolate
    ===========

    Provides tools to interpolate vector fields, normalised to 1 with respect to
    a Finsler function, on W2. The primary methods are:
      1. `vectorfield_trilinear_interpolate_LI`: interpolate a vector field,
      with norm 1, given with respect to the left invariant frame, trilinearly
      at some point in the domain. This method seems not to work properly.
      2. `vectorfield_trilinear_interpolate_static`: interpolate a vector field,
      with norm 1, given with respect to the static frame, trilinearly at some
      point in the domain.
"""

import taichi as ti
from eikivp.W2.utils import (
    trilinear_interpolate,
    scalar_trilinear_interpolate,
    sanitize_index
)
from eikivp.W2.plus.metric import (
    normalise_LI,
    normalise_static
)

# I don't think this works properly, we should actually also interpolate the
# frame... But for sufficiently fine discretisation it won't matter.
@ti.func
def vectorfield_trilinear_interpolate_LI(
    vectorfield: ti.template(),
    index: ti.template(),
    ξ: ti.f32,
    cost_field: ti.template()
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    Interpolate vector field, normalised to 1 and given in the left invariant
    frame, `vectorfield` at continuous `index` trilinearly, via repeated linear
    interpolation (α, β, φ).

    Args:
        `vectorfield`: ti.Vector.field(n=3, dtype=[float]) in which we want to 
          interpolate.
        `index`: ti.types.vector(n=3, dtype=[float]) continuous index at which 
          we want to interpolate.
        `ξ`: Stiffness of moving in the B1 direction compared to the B3
          direction, taking values greater than 0.
        `cost_field`: ti.field(dtype=[float]) of cost function, taking values 
          between 0 and 1.

    Returns:
        ti.types.vector(n=3, dtype=[float]) of value `vectorfield` interpolated 
          at `index`.
    """
    r = ti.math.fract(index)
    f = ti.math.floor(index, ti.i32)
    f = sanitize_index(f, vectorfield)
    c = ti.math.ceil(index, ti.i32)
    c = sanitize_index(c, vectorfield)

    u000, v000, w000 = vectorfield[f[0], f[1], f[2]]
    u001, v001, w001 = vectorfield[f[0], f[1], c[2]]
    u010, v010, w010 = vectorfield[f[0], c[1], f[2]]
    u011, v011, w011 = vectorfield[f[0], c[1], c[2]]
    u100, v100, w100 = vectorfield[c[0], f[1], f[2]]
    u101, v101, w101 = vectorfield[c[0], f[1], c[2]]
    u110, v110, w110 = vectorfield[c[0], c[1], f[2]]
    u111, v111, w111 = vectorfield[c[0], c[1], c[2]]

    u = trilinear_interpolate(u000, u001, u010, u011, u100, u101, u110, u111, r)
    v = trilinear_interpolate(v000, v001, v010, v011, v100, v101, v110, v111, r)
    w = trilinear_interpolate(w000, w001, w010, w011, w100, w101, w110, w111, r)

    cost = scalar_trilinear_interpolate(cost_field, index)

    return normalise_LI(ti.Vector([u, v, w]), ξ, cost)

@ti.func
def vectorfield_trilinear_interpolate_static(
    vectorfield: ti.template(),
    index: ti.template(),
    αs: ti.template(),
    φs: ti.template(),
    ξ: ti.f32,
    cost_field: ti.template()
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    Interpolate vector field, normalised to 1 and given in the static frame,
    `vectorfield` at continuous `index` trilinearly, via repeated linear
    interpolation (α, β, φ).

    Args:
        `vectorfield`: ti.Vector.field(n=3, dtype=[float]) in which we want to 
          interpolate.
        `index`: ti.types.vector(n=3, dtype=[float]) continuous index at which 
          we want to interpolate.
        `αs`: α-coordinate at each grid point.
        `φs`: angle coordinate at each grid point.
        `ξ`: Stiffness of moving in the B1 direction compared to the B3
          direction, taking values greater than 0.
        `cost_field`: ti.field(dtype=[float]) of cost function, taking values 
          between 0 and 1.

    Returns:
        ti.types.vector(n=3, dtype=[float]) of value `vectorfield` interpolated 
          at `index`.
    """
    r = ti.math.fract(index)
    f = ti.math.floor(index, ti.i32)
    f = sanitize_index(f, vectorfield)
    c = ti.math.ceil(index, ti.i32)
    c = sanitize_index(c, vectorfield)

    u000, v000, w000 = vectorfield[f[0], f[1], f[2]]
    u001, v001, w001 = vectorfield[f[0], f[1], c[2]]
    u010, v010, w010 = vectorfield[f[0], c[1], f[2]]
    u011, v011, w011 = vectorfield[f[0], c[1], c[2]]
    u100, v100, w100 = vectorfield[c[0], f[1], f[2]]
    u101, v101, w101 = vectorfield[c[0], f[1], c[2]]
    u110, v110, w110 = vectorfield[c[0], c[1], f[2]]
    u111, v111, w111 = vectorfield[c[0], c[1], c[2]]

    u = trilinear_interpolate(u000, u001, u010, u011, u100, u101, u110, u111, r)
    v = trilinear_interpolate(v000, v001, v010, v011, v100, v101, v110, v111, r)
    w = trilinear_interpolate(w000, w001, w010, w011, w100, w101, w110, w111, r)

    α = scalar_trilinear_interpolate(αs, index)
    φ = scalar_trilinear_interpolate(φs, index)

    cost = scalar_trilinear_interpolate(cost_field, index)

    return normalise_static(ti.Vector([u, v, w]), ξ, cost, α, φ)