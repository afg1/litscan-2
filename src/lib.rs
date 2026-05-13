use pyo3::prelude::*;
use std::{io::Read, path::PathBuf, path::Path};
use std::collections::HashMap;
use rayon::prelude::*;

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

#[pyfunction]
fn scan_shard(searcher: &scanner::PatternIndex, shard_path: String) -> PyResult<HashMap<String, Vec<scanner::Hit>>> {
    let path = Path::new(&shard_path);
    let mut file = std::fs::File::open(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("failed to open {}: {}", path.display(), e)))?;
    let mut xml_string = String::new();
    file.read_to_string(&mut xml_string)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("failed to read {}: {}", path.display(), e)))?;
    let shard_doc = jats::parse_document(&xml_string)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("failed to parse {}: {}", path.display(), e)))?;
    let article_nodes: Vec<_> = jats::article_nodes(&shard_doc).collect();

    let results = article_nodes
        .par_iter()
        .map(|(pmcid, node)| {
            let article = jats::parse_article_node(*node);
            let hits = scanner::search_article(searcher, &article)?;
            Ok((pmcid.clone(), hits))
        })
        .collect::<PyResult<HashMap<String, Vec<scanner::Hit>>>>()?;

    Ok(results)
}



/// A Python module implemented in Rust.
#[pymodule]
fn litscan_2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_scanner, m)?)?;
    m.add_function(wrap_pyfunction!(scan_one, m)?)?;
    m.add_function(wrap_pyfunction!(scan_shard, m)?)?;
    Ok(())
}



