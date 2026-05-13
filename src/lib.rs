use pyo3::prelude::*;
use std::path::PathBuf;
pub mod classifier;
pub mod jats;
pub mod scanner;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn build_scanner(input_ids: String) -> PyResult<scanner::PatternIndex> {
    let id_path = PathBuf::from(input_ids);
    let pattern_scanner = scanner::build_search_automaton(id_path)?;

    Ok(pattern_scanner)
}

#[pyfunction]
fn scan_one(searcher: &scanner::PatternIndex, xml_string: String) -> PyResult<Vec<scanner::Hit>> {

    let article = jats::parse(&xml_string)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let hits = scanner::search_article(searcher, &article)?;

    Ok(hits)

}




/// A Python module implemented in Rust.
#[pymodule]
fn litscan_2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_scanner, m)?)?;
    m.add_function(wrap_pyfunction!(scan_one, m)?)?;
    Ok(())
}



