"""
    vesselness
    ==========

    Provides tools compute vesselness scores on R^2. The available methods are:
      1. `vesselness`: compute the multiscale vesselness with preprocessing to
      remove boundary effects.
      2. `rc_vessel_enhancement`: compute the singlescale vesselness using a
      Frangi filter[1].
      3. `multiscale_frangi_filter`: compute the multiscale vesselness by
      applying the Frangi filter at numerous scales and combining the results
      via maximum projection.
      
    References:
        [1]: A. F. Frangi, W. J. Niessen, K. L. Vincken, and M. A. Viergever.
        "Multiscale vessel enhancement filtering". In: Medical Image Computing
        and Computer-Assisted Intervention (1998), pp. 130--137.
        DOI:10.1007/BFb0056195.
"""

import numpy as np
import scipy as sp
import diplib as dip
import h5py
from eikivp.utils import image_rescale

def import_vesselness(params, folder):
    """
    Import the vesselness matching `params`.
    """
    scales = params["scales"]
    α = params["α"]
    γ = params["γ"]
    ε = params["ε"]
    image_name = params["image_name"]
    vesselness_filename = f"{folder}\\R2_ss={[s for s in scales]}_a={α}_g={γ}_e={ε}.hdf5"
    with h5py.File(vesselness_filename, "r") as vesselness_file:
        assert (
            np.all(scales == vesselness_file.attrs["scales"]) and
            α == vesselness_file.attrs["α"] and
            γ == vesselness_file.attrs["γ"] and
            ε == vesselness_file.attrs["ε"] and
            image_name == vesselness_file.attrs["image_name"]
        ), "There is a parameter mismatch!"
        V = vesselness_file["Vesselness"][()]
    return V
        
def export_vesselness(V, params, folder):
    """
    Export the vesselness to hdf5 with `params` stored as metadata.
    """
    scales = params["scales"]
    α = params["α"]
    γ = params["γ"]
    ε = params["ε"]
    image_name = params["image_name"]
    vesselness_filename = f"{folder}\\R2_ss={[s for s in scales]}_a={α}_g={γ}_e={ε}.hdf5"
    with h5py.File(vesselness_filename, "w") as vesselness_file:
        vesselness_file.create_dataset("Vesselness", data=V)
        vesselness_file.attrs["scales"] = scales
        vesselness_file.attrs["α"] = α
        vesselness_file.attrs["γ"] = γ
        vesselness_file.attrs["ε"] = ε
        vesselness_file.attrs["image_name"] = image_name

def vesselness(retinal_array, scales, α, γ, ε):
    V_unmasked = multiscale_frangi_filter(-retinal_array, scales, α=α, γ=γ, ε=ε)
    mask = (retinal_array > 0) # Remove boundary
    V_unnormalised = V_unmasked * sp.ndimage.binary_erosion(mask, iterations=int(np.ceil(scales.max() * 2)))
    print(f"Before rescaling, vesselness is in [{V_unnormalised.min()}, {V_unnormalised.max()}].")
    return image_rescale(V_unnormalised)

def rc_vessel_enhancement(image, σ, α=0.2, γ=0.75, ε=0.2):
    """
    Compute Frangi filter[1] of vessels in `image` at a single scale `σ`.

    Args:
        `image`: np.ndarray of a grayscale image, taking values between 0 and 1,
          with shape [Nx, Ny].
        `σ`: Standard deviation of Gaussian derivatives, taking values greater 
          than 0.
        `α`: Anisotropy penalty, taking values between 0 and 1.
        `γ`: Variance sensitivity, taking values between 0 and 1.
        `ε`: Structure penalty, taking values between 0 and 1.

    Returns:
        np.ndarray of the vesselness of `image`, taking values between 0 and 1.
    
    References:
        [1]: A. F. Frangi, W. J. Niessen, K. L. Vincken, and M. A. Viergever.
          "Multiscale vessel enhancement filtering". In: Medical Image Computing
          and Computer-Assisted Intervention (1998), pp. 130--137.
          DOI:10.1007/BFb0056195.
    """
    # Calculate Hessian derivatives.
    Lxx = np.array(dip.Gauss(image, (σ, σ), (2, 0)))
    Lxy = np.array(dip.Gauss(image, (σ, σ), (1, 1)))
    Lyy = np.array(dip.Gauss(image, (σ, σ), (0, 2)))

    # Calculate eigenvalues.
    λ = Lxx + Lyy
    λδ = np.sign(λ) * np.sqrt((2 * Lxy)**2 + (Lxx - Lyy)**2)
    λ1, λ2 = (σ**γ / 2) * np.array((λ + λδ, λ - λδ))

    # Calculate vesselness. Not quite sure what these variables represent.
    R2 = (λ2 / (λ1 + np.finfo(np.float64).eps)) ** 2
    nR2 = -1 / (2 * α**2)
    S2 = λ1**2 + λ2**2
    nS2 = -1 / (2 * ε**2 * np.max(S2))
    vesselness = (np.exp(nR2 * R2**2)
                  * (1 - np.exp(nS2 * S2))
                  * np.heaviside(-λ1, 1.))
    return vesselness


def multiscale_frangi_filter(image, σs, α=0.3, γ=0.75, ε=0.3):
    """
    Compute Frangi filter[1] of vessels in `image` at scales in `σs`.
    Implementation adapted from "Code A - Vesselness in SE(2)".

    Args:
        `image`: np.ndarray of a grayscale image, taking values between 0 and 1.
        `σs`: Iterable of standard deviations of Gaussian derivatives, taking
          values greater than 0.
        `α`: Anisotropy penalty, taking values between 0 and 1.
        `γ`: Variance sensitivity, taking values between 0 and 1.
        `ε`: Structure penalty, taking values between 0 and 1.

    Returns:
        np.ndarray of the vesselness of `image`, taking values between 0 and 1.
    
    References:
        [1]: A. F. Frangi, W. J. Niessen, K. L. Vincken, and M. A. Viergever.
          "Multiscale vessel enhancement filtering". In: Medical Image Computing
          and Computer-Assisted Intervention (1998), pp. 130--137.
          DOI:10.1007/BFb0056195.
    """
    # Compute vesselness at each scale σ in σs, and select the maximum at
    # each point.
    vesselnesses = []
    for σ in σs:
        vesselnesses.append(rc_vessel_enhancement(image, σ, α=α, γ=γ, ε=ε))
    vesselness = np.maximum.reduce(vesselnesses)
    return vesselness