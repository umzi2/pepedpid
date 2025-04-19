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