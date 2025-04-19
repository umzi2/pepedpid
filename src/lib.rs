mod dpid_core;

use ndarray::{Ix2, Ix3};
use numpy::{ PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, ToPyArray};
use pyo3::prelude::*;

#[pyfunction]
fn dpid_resize<'py>(
    input: PyReadonlyArrayDyn<f32>,
    h: usize,
    w: usize,
    l: f32,
    py: Python<'py>,
) -> Bound<'py, PyArrayDyn<f32>> {
    let input = input.to_owned_array();
    let input = py.allow_threads(|| {
        if input.shape().len() != 2 {
            dpid_core::dpid_resample_rgb(&input.into_dimensionality::<Ix3>().unwrap(), h, w, l)
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
    input.to_pyarray(py)
}

#[pymodule]
fn pepedpid(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dpid_resize, m)?)?;
    Ok(())
}
