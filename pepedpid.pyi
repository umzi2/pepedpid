from __future__ import annotations
import numpy as np

def dpid_resize(
    input:np.ndarray,
    h:int,
    w:int,
    l:float
)->np.ndarray:
    """
    Resizes the image using the DPID (Detail-Preserving Image Downscaling) method.

    Parameters:
        input (np.ndarray): Input image in HWC or CHW format (depending on implementation).
        h (int): Desired height of the output image.
        w (int): Desired width of the output image.
        l (float): Lambda (λ) parameter controlling the balance between detail preservation and smoothing.
                   Higher values of λ emphasize edges and details during downscaling.

    Returns:
        np.ndarray: Resized image of shape (h, w) with visually significant details preserved.

    Based on the method:
        "Rapid, Detail-Preserving Image Downscaling" (Cho et al., 2015)
    """
def cubic_resize(
    input:np.ndarray,
    h:int,
    w:int,
)->np.ndarray:
    """
    Resizes the image using the bicubic interpolation method, following the MATLAB-style bicubic kernel.

    Parameters:
        input (np.ndarray): Input image in HWC or CHW format (depending on implementation).
        h (int): Desired height of the output image.
        w (int): Desired width of the output image.

    Returns:
        np.ndarray: Resized image of shape (h, w), interpolated using bicubic filtering for smooth transitions
                    and natural-looking results, especially effective for photographic and continuous-tone content.

    Based on the MATLAB-style bicubic interpolation:
        Utilizes a 4x4 neighborhood with a cubic convolution kernel for computing pixel values.
    """