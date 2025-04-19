from __future__ import annotations
import numpy as np

#     input: PyReadonlyArrayDyn<f32>,
#     h: usize,
#     w: usize,
#     lambda: f32,
def dpid_resize(
    input:np.ndarray,
    h:int,
    w:int,
    l:float
)->np.ndarray:...