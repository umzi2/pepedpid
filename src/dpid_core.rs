use ndarray::{Array2, Array3};

pub fn dpid_resample_rgb(x: &Array3<f32>, o_h: usize, o_w: usize, lambda: f32) -> Array3<f32> {
    let (i_h, i_w, _) = x.dim();
    let p_w = i_w as f32 / o_w as f32;
    let p_h = i_h as f32 / o_h as f32;

    const K: [[f32; 3]; 3] = [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]];

    let mut out = Array3::<f32>::zeros((o_h, o_w, 3));

    for py in 0..o_h {
        for px in 0..o_w {
            // === 1. Сглаженные avg-патчи по 3×3 ===
            let mut s0 = 0.0;
            let mut s1 = 0.0;
            let mut s2 = 0.0;
            let mut sw = 0.0;
            for ky in 0..3 {
                let ny = (py + ky).saturating_sub(1).min(o_h - 1);
                for kx in 0..3 {
                    let nx = (px + kx).saturating_sub(1).min(o_w - 1);
                    let w_k = K[ky][kx];

                    let sx = nx as f32 * p_w;
                    let ex = sx + p_w;
                    let sy = ny as f32 * p_h;
                    let ey = sy + p_h;
                    let sxr = sx.floor() as usize;
                    let syr = sy.floor() as usize;
                    let exr = ex.ceil().min(i_w as f32) as usize;
                    let eyr = ey.ceil().min(i_h as f32) as usize;

                    let mut a0 = 0.0;
                    let mut a1 = 0.0;
                    let mut a2 = 0.0;
                    let mut aw = 0.0;
                    for iy in syr..eyr {
                        for ix in sxr..exr {
                            let fx1 = (1.0 - (sx - ix as f32).max(0.0)).min(1.0);
                            let fx2 = (1.0 - ((ix + 1) as f32 - ex).max(0.0)).min(1.0);
                            let fy1 = (1.0 - (sy - iy as f32).max(0.0)).min(1.0);
                            let fy2 = (1.0 - ((iy + 1) as f32 - ey).max(0.0)).min(1.0);
                            let cov = fx1 * fx2 * fy1 * fy2;

                            let px0 = x[(iy, ix, 0)];
                            let px1 = x[(iy, ix, 1)];
                            let px2 = x[(iy, ix, 2)];

                            a0 += px0 * cov;
                            a1 += px1 * cov;
                            a2 += px2 * cov;
                            aw += cov;
                        }
                    }
                    let inv_aw = 1.0 / aw.max(f32::EPSILON);
                    let avg0 = a0 * inv_aw;
                    let avg1 = a1 * inv_aw;
                    let avg2 = a2 * inv_aw;

                    s0 += avg0 * w_k;
                    s1 += avg1 * w_k;
                    s2 += avg2 * w_k;
                    sw += w_k;
                }
            }

            let inv_sw = 1.0 / sw.max(f32::EPSILON);
            let m0 = s0 * inv_sw;
            let m1 = s1 * inv_sw;
            let m2 = s2 * inv_sw;

            let sx = px as f32 * p_w;
            let ex = sx + p_w;
            let sy = py as f32 * p_h;
            let ey = sy + p_h;
            let sxr = sx.floor() as usize;
            let syr = sy.floor() as usize;
            let exr = ex.ceil().min(i_w as f32) as usize;
            let eyr = ey.ceil().min(i_h as f32) as usize;

            let mut o0 = 0.0;
            let mut o1 = 0.0;
            let mut o2 = 0.0;
            let mut ow = 0.0;
            for iy in syr..eyr {
                for ix in sxr..exr {
                    let fx1 = (1.0 - (sx - ix as f32).max(0.0)).min(1.0);
                    let fx2 = (1.0 - ((ix + 1) as f32 - ex).max(0.0)).min(1.0);
                    let fy1 = (1.0 - (sy - iy as f32).max(0.0)).min(1.0);
                    let fy2 = (1.0 - ((iy + 1) as f32 - ey).max(0.0)).min(1.0);
                    let cov = fx1 * fx2 * fy1 * fy2;

                    let d0 = m0 - x[(iy, ix, 0)];
                    let d1 = m1 - x[(iy, ix, 1)];
                    let d2 = m2 - x[(iy, ix, 2)];

                    let f = if lambda == 0.0 {
                        cov
                    } else {
                        ((d0 * d0 + d1 * d1 + d2 * d2).sqrt()).powf(lambda) * cov
                    };

                    o0 += x[(iy, ix, 0)] * f;
                    o1 += x[(iy, ix, 1)] * f;
                    o2 += x[(iy, ix, 2)] * f;
                    ow += f;
                }
            }

            let (r, g, b) = if ow > 0.0 {
                (o0 / ow, o1 / ow, o2 / ow)
            } else {
                (m0, m1, m2)
            };

            out[(py, px, 0)] = r;
            out[(py, px, 1)] = g;
            out[(py, px, 2)] = b;
        }
    }

    out
}
pub fn dpid_resample_gray(x: &Array2<f32>, o_h: usize, o_w: usize, lambda: f32) -> Array2<f32> {
    let (i_h, i_w) = x.dim();
    let p_w = i_w as f32 / o_w as f32;
    let p_h = i_h as f32 / o_h as f32;

    const K: [[f32; 3]; 3] = [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]];

    let mut out = Array2::<f32>::zeros((o_h, o_w));

    for py in 0..o_h {
        for px in 0..o_w {
            // === 1. Сглаженные avg-патчи по 3×3 ===
            let mut s0 = 0.0;
            let mut sw = 0.0;
            for ky in 0..3 {
                let ny = (py + ky).saturating_sub(1).min(o_h - 1);
                for kx in 0..3 {
                    let nx = (px + kx).saturating_sub(1).min(o_w - 1);
                    let w_k = K[ky][kx];

                    let sx = nx as f32 * p_w;
                    let ex = sx + p_w;
                    let sy = ny as f32 * p_h;
                    let ey = sy + p_h;
                    let sxr = sx.floor() as usize;
                    let syr = sy.floor() as usize;
                    let exr = ex.ceil().min(i_w as f32) as usize;
                    let eyr = ey.ceil().min(i_h as f32) as usize;

                    let mut a0 = 0.0;
                    let mut aw = 0.0;
                    for iy in syr..eyr {
                        for ix in sxr..exr {
                            let fx1 = (1.0 - (sx - ix as f32).max(0.0)).min(1.0);
                            let fx2 = (1.0 - ((ix + 1) as f32 - ex).max(0.0)).min(1.0);
                            let fy1 = (1.0 - (sy - iy as f32).max(0.0)).min(1.0);
                            let fy2 = (1.0 - ((iy + 1) as f32 - ey).max(0.0)).min(1.0);
                            let cov = fx1 * fx2 * fy1 * fy2;

                            let px0 = x[(iy, ix)];

                            a0 += px0 * cov;
                            aw += cov;
                        }
                    }
                    let inv_aw = 1.0 / aw.max(f32::EPSILON);
                    let avg0 = a0 * inv_aw;

                    s0 += avg0 * w_k;
                    sw += w_k;
                }
            }

            let inv_sw = 1.0 / sw.max(f32::EPSILON);
            let m0 = s0 * inv_sw;

            // === 2. Ресемплинг по патчу с дистанционными весами ===
            let sx = px as f32 * p_w;
            let ex = sx + p_w;
            let sy = py as f32 * p_h;
            let ey = sy + p_h;
            let sxr = sx.floor() as usize;
            let syr = sy.floor() as usize;
            let exr = ex.ceil().min(i_w as f32) as usize;
            let eyr = ey.ceil().min(i_h as f32) as usize;

            let mut o0 = 0.0;
            let mut ow = 0.0;
            for iy in syr..eyr {
                for ix in sxr..exr {
                    let fx1 = (1.0 - (sx - ix as f32).max(0.0)).min(1.0);
                    let fx2 = (1.0 - ((ix + 1) as f32 - ex).max(0.0)).min(1.0);
                    let fy1 = (1.0 - (sy - iy as f32).max(0.0)).min(1.0);
                    let fy2 = (1.0 - ((iy + 1) as f32 - ey).max(0.0)).min(1.0);
                    let cov = fx1 * fx2 * fy1 * fy2;

                    let d0 = m0 - x[(iy, ix)];

                    let f = if lambda == 0.0 {
                        cov
                    } else {
                        d0.abs().powf(lambda) * cov
                    };

                    o0 += x[(iy, ix)] * f;
                    ow += f;
                }
            }

            let r = if ow > 0.0 { o0 / ow } else { m0 };

            out[(py, px)] = r;
        }
    }

    out
}
