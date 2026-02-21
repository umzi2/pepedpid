mod dpid_core;
mod matlab_cubic_core;

use ndarray::{Array2, Array3, Ix2, Ix3};
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::prelude::*;

#[pyfunction]
fn dpid_resize<'py>(
    input: PyReadonlyArrayDyn<'py, f32>,
    h: usize,
    w: usize,
    l: f32,
    py: Python<'py>,
) -> Bound<'py, PyArrayDyn<f32>> {
    let input = input.to_owned_array();
    let result = py.detach(|| {
        if input.ndim() != 2 {
            dpid_core::dpid_resample_rgb(
                &input.into_dimensionality::<Ix3>().unwrap(),
                h,
                w,
                l,
            )
                .into_dyn()
        } else {
            dpid_core::dpid_resample_gray(
                &input.into_dimensionality::<Ix2>().unwrap(),
                h,
                w,
                l,
            )
                .into_dyn()
        }
    });
    result.into_pyarray(py)
}

#[pyfunction]
fn cubic_resize<'py>(
    input: PyReadonlyArrayDyn<'py, f32>,
    h: usize,
    w: usize,
    py: Python<'py>,
) -> Bound<'py, PyArrayDyn<f32>> {
    let input = input.to_owned_array();
    let result = py.detach(|| {
        if input.ndim() == 3 {
            let mut output = Array3::<f32>::zeros((h, w, 3));
            matlab_cubic_core::imresize3_mut(
                &input.into_dimensionality::<Ix3>().unwrap().view(),
                &mut output,
                h,
                w,
            );
            output.into_dyn()
        } else {
            let mut output = Array2::<f32>::zeros((h, w));
            matlab_cubic_core::imresize2_mut(
                &input.into_dimensionality::<Ix2>().unwrap().view(),
                &mut output,
                h,
                w,
            );
            output.into_dyn()
        }
    });

    result.into_pyarray(py)
}

#[pymodule]
fn pepedpid(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dpid_resize, m)?)?;
    m.add_function(wrap_pyfunction!(cubic_resize, m)?)?;
    Ok(())
}