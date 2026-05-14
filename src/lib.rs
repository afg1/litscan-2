use pyo3::prelude::*;
use std::{io::Read, path::PathBuf, path::Path};
use std::sync::Arc;
use std::sync::mpsc::sync_channel;
use std::thread;
use std::fs::File;
use rayon::prelude::*;

use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

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

fn hits_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new_dictionary("pmcid", DataType::UInt32, DataType::Utf8, false),
        Field::new_dictionary("urs_taxid", DataType::UInt32, DataType::Utf8, false),
        Field::new_dictionary("section", DataType::UInt32, DataType::Utf8, false),
        Field::new_dictionary("symbol", DataType::UInt32, DataType::Utf8, false),
        Field::new_dictionary("sentence", DataType::UInt32, DataType::Utf8, false),
        Field::new("offset", DataType::UInt32, false),
    ]))
}

#[pyfunction]
fn scan_shard(
    searcher: &scanner::PatternIndex,
    shard_path: String,
    output_path: String,
    filter_model: Option<&classifier::TfidfClassifier>
) -> PyResult<u64> {
    let path = Path::new(&shard_path);
    let mut file = std::fs::File::open(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("failed to open {}: {}", path.display(), e)))?;
    let mut xml_string = String::new();
    file.read_to_string(&mut xml_string)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("failed to read {}: {}", path.display(), e)))?;
    let shard_doc = jats::parse_document(&xml_string)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("failed to parse {}: {}", path.display(), e)))?;
    let article_nodes: Vec<_> = jats::article_nodes(&shard_doc).collect();

    let schema = hits_schema();

    let out_path = PathBuf::from(&output_path);
    let out_file = File::create(&out_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("failed to create {}: {}", out_path.display(), e)))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        .build();
    let mut writer = ArrowWriter::try_new(out_file, schema.clone(), Some(props))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("parquet writer init failed: {}", e)))?;

    let n_threads = rayon::current_num_threads().max(1);
    let (tx, rx) = sync_channel::<RecordBatch>(n_threads * 2);

    let writer_handle = thread::spawn(move || -> Result<u64, parquet::errors::ParquetError> {
        let mut total: u64 = 0;
        for batch in rx {
            total += batch.num_rows() as u64;
            writer.write(&batch)?;
        }
        writer.close()?;
        Ok(total)
    });

    let producer_result: PyResult<()> = article_nodes
        .par_iter()
        .try_for_each_with(tx.clone(), |tx, (pmcid, node)| {
            let article = jats::parse_article_node(*node);

            

            let maybe_batch = scanner::search_article_to_batch(searcher, &article, pmcid, &schema, &filter_model)?;
            // Drop the article (and its cloned-out section text) before blocking on send.
            drop(article);
            if let Some(batch) = maybe_batch {
                tx.send(batch).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("writer thread died: {}", e))
                })?;
            }
            Ok(())
        });

    drop(tx);

    let writer_result = writer_handle
        .join()
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("writer thread panicked"))?;

    producer_result?;
    let total = writer_result
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("parquet write failed: {}", e)))?;

    Ok(total)
}



/// A Python module implemented in Rust.
#[pymodule]
fn litscan_2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_scanner, m)?)?;
    m.add_function(wrap_pyfunction!(scan_one, m)?)?;
    m.add_function(wrap_pyfunction!(scan_shard, m)?)?;

    m.add_class::<classifier::TfidfClassifier>()?;
    m.add_class::<classifier::ClassificationResult>()?;
    Ok(())
}
