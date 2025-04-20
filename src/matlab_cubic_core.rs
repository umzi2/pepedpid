use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
fn derive_scale(in_shape: (usize, usize), out_shape: (usize, usize)) -> (f32, f32) {
    (
        out_shape.0 as f32 / in_shape.0 as f32,
        out_shape.1 as f32 / in_shape.1 as f32,
    )
}

#[inline(always)]
fn cubic_scalar(x: f32) -> f32 {
    let absx = x.abs();
    let absx2 = absx * absx;
    let absx3 = absx2 * absx;
    if absx <= 1.0 {
        absx3 * 1.5 - absx2 * 2.5 + 1.0
    } else if absx <= 2.0 {
        absx3 * -0.5 + absx2 * 2.5 - absx * 4.0 + 2.0
    } else {
        0.0
    }
}

fn contributions(
    in_len: usize,
    out_len: usize,
    scale: f32,
    k_width: f32,
) -> (Array2<f32>, Array2<usize>) {
    let inv_scale = 1.0 / scale;
    let kernel_width = if scale < 1.0 {
        k_width / scale
    } else {
        k_width
    };
    let scale_func = if scale < 1.0 { Some(scale) } else { None };

    let p = kernel_width.ceil() as usize + 2;
    let mut weights = Array2::<f32>::zeros((out_len, p));
    let mut indices = Array2::<usize>::zeros((out_len, p));

    let aux_len = in_len * 2;
    let mut aux = Vec::with_capacity(aux_len);
    aux.extend(0..in_len);
    aux.extend((0..in_len).rev());

    for i in 0..out_len {
        let u = (i as f32 + 1.0) * inv_scale + 0.5 * (1.0 - inv_scale);
        let left = (u - kernel_width / 2.0).floor() as isize;

        let mut sum = 0.0;
        for j in 0..p {
            let idx = left + j as isize - 1;
            let dist = u - (idx as f32) - 1.0;

            let weight = if let Some(s) = scale_func {
                s * cubic_scalar(s * dist)
            } else {
                cubic_scalar(dist)
            };

            weights[[i, j]] = weight;
            let wrapped = ((idx % aux_len as isize + aux_len as isize) % aux_len as isize) as usize;
            indices[[i, j]] = aux[wrapped];
            sum += weight;
        }

        if sum != 0.0 {
            for j in 0..p {
                weights[[i, j]] /= sum;
            }
        }
    }

    // Обрезка нулевых весов (сжимаем)
    let mut active_cols = vec![];
    for j in 0..p {
        if weights.column(j).iter().any(|&v| v != 0.0) {
            active_cols.push(j);
        }
    }

    let new_p = active_cols.len();
    let mut w2 = Array2::<f32>::zeros((out_len, new_p));
    let mut i2 = Array2::<usize>::zeros((out_len, new_p));

    for (new_j, &j) in active_cols.iter().enumerate() {
        for i in 0..out_len {
            w2[[i, new_j]] = weights[[i, j]];
            i2[[i, new_j]] = indices[[i, j]];
        }
    }

    (w2, i2)
}
pub fn imresize3_mut(
    input: &ArrayView3<f32>,
    output: &mut Array3<f32>,
    out_h: usize,
    out_w: usize,
) {
    let (in_h, in_w, _) = input.dim();
    let (scale_h, scale_w) = derive_scale((in_h, in_w), (out_h, out_w));

    let order = if scale_h < scale_w { [0, 1] } else { [1, 0] };

    let (w0, i0) = contributions(in_h, out_h, scale_h, 4.0);
    let (w1, i1) = contributions(in_w, out_w, scale_w, 4.0);

    let mut result = input.to_owned();
    for &dim in &order {
        result = if dim == 0 {
            unsafe { resize_along_dim3_mut(&result, &w0, &i0, 0) }
        } else {
            unsafe { resize_along_dim3_mut(&result, &w1, &i1, 1) }
        };
    }

    output.assign(&result);
}

pub fn imresize2_mut(
    input: &ArrayView2<f32>,
    output: &mut Array2<f32>,
    out_h: usize,
    out_w: usize,
) {
    let (in_h, in_w) = input.dim();
    let (scale_h, scale_w) = derive_scale((in_h, in_w), (out_h, out_w));

    let order = if scale_h < scale_w { [0, 1] } else { [1, 0] };

    let (w0, i0) = contributions(in_h, out_h, scale_h, 4.0);
    let (w1, i1) = contributions(in_w, out_w, scale_w, 4.0);

    let mut result = input.to_owned();
    for &dim in &order {
        result = if dim == 0 {
            unsafe { resize_along_dim2_mut(&result, &w0, &i0, 0) }
        } else {
            unsafe { resize_along_dim2_mut(&result, &w1, &i1, 1) }
        };
    }

    output.assign(&result);
}

unsafe fn resize_along_dim3_mut(
    input: &Array3<f32>,
    weights: &Array2<f32>,
    indices: &Array2<usize>,
    dim: usize,
) -> Array3<f32> {
    let (in_h, in_w, in_c) = input.dim();
    let (out_len, kernel_size) = weights.dim();
    let mut output = match dim {
        0 => Array3::<f32>::uninit((out_len, in_w, in_c)).assume_init(),
        1 => Array3::<f32>::uninit((in_h, out_len, in_c)).assume_init(),
        _ => unreachable!(),
    };

    if dim == 0 {
        for y in 0..out_len {
            for x in 0..in_w {
                for c in 0..in_c {
                    let mut acc = 0.0;
                    for k in 0..kernel_size {
                        let idx = indices[[y, k]];
                        acc += weights[[y, k]] * input[[idx, x, c]];
                    }
                    output[[y, x, c]] = acc;
                }
            }
        }
    } else {
        for y in 0..in_h {
            for x in 0..out_len {
                for c in 0..in_c {
                    let mut acc = 0.0;
                    for k in 0..kernel_size {
                        let idx = indices[[x, k]];
                        acc += weights[[x, k]] * input[[y, idx, c]];
                    }
                    output[[y, x, c]] = acc;
                }
            }
        }
    }

    output
}

unsafe fn resize_along_dim2_mut(
    input: &Array2<f32>,
    weights: &Array2<f32>,
    indices: &Array2<usize>,
    dim: usize,
) -> Array2<f32> {
    let (in_h, in_w) = input.dim();
    let (out_len, kernel_size) = weights.dim();
    let mut output = match dim {
        0 => Array2::<f32>::uninit((out_len, in_w)).assume_init(),
        1 => Array2::<f32>::uninit((in_h, out_len)).assume_init(),
        _ => unreachable!(),
    };

    if dim == 0 {
        for y in 0..out_len {
            for x in 0..in_w {
                let mut acc = 0.0;
                for k in 0..kernel_size {
                    let idx = indices[[y, k]];
                    acc += weights[[y, k]] * input[[idx, x]];
                }
                output[[y, x]] = acc;
            }
        }
    } else {
        for y in 0..in_h {
            for x in 0..out_len {
                let mut acc = 0.0;
                for k in 0..kernel_size {
                    let idx = indices[[x, k]];
                    acc += weights[[x, k]] * input[[y, idx]];
                }
                output[[y, x]] = acc;
            }
        }
    }

    output
}
