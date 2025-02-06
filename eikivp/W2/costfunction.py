"""
    costfunction
    ============

    Compute the cost function on W2 by interpolating the cost function on
    M2.
"""

import taichi as ti
from eikivp.W2.utils import Π_forward
from eikivp.M2.utils import(
    scalar_trilinear_interpolate,
    coordinate_real_to_array_ti
)

def cost(C_M2, αs, βs, φs, a, c, x_min, y_min, θ_min, dxy, dθ):
    shape = C_M2.shape
    C_M2_ti = ti.field(dtype=ti.f32, shape=shape)
    C_M2_ti.from_numpy(C_M2)
    αs_ti = ti.field(dtype=ti.f32, shape=shape)
    αs_ti.from_numpy(αs)
    βs_ti = ti.field(dtype=ti.f32, shape=shape)
    βs_ti.from_numpy(βs)
    φs_ti = ti.field(dtype=ti.f32, shape=shape)
    φs_ti.from_numpy(φs)
    CW2_ti = ti.field(dtype=ti.f32, shape=shape)
    interpolate_cost_function(C_M2_ti, αs_ti, βs_ti, φs_ti, a, c, x_min, y_min, θ_min, dxy, dθ, CW2_ti)
    return CW2_ti.to_numpy()


@ti.kernel
def interpolate_cost_function(
    cost_SE2: ti.template(),
    αs: ti.template(),
    βs: ti.template(),
    φs: ti.template(),
    a: ti.f32,
    c: ti.f32,
    x_min: ti.f32,
    y_min: ti.f32,
    θ_min: ti.f32,
    dxy: ti.f32,
    dθ: ti.f32,
    cost_SO3: ti.template()
):
    """
    @ti.kernel

    Sample cost function `cost_SE2`, given as a volume sampled uniformly on
    SE(2), as a volume in SO(3)

    Args:
        `αs`: α-coordinates at which we want to sample.
        `βs`: β-coordinates at which we want to sample.
        `φs`: φ-coordinates at which we want to sample.
        `a`: distance between nodal point of projection and centre of sphere.
        `c`: distance between projection plane and centre of sphere reflected
          around nodal point.
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `θ_min`: minimum value of θ-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
        `dθ`: orientational resolution, taking values greater than 0.
    """
    for I in ti.grouped(cost_SE2):
        point = Π_forward(αs[I], βs[I], φs[I], a, c)
        index = coordinate_real_to_array_ti(point, x_min, y_min, θ_min, dxy, dθ)
        cost_SO3[I] = scalar_trilinear_interpolate(cost_SE2, index)