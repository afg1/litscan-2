use pyo3::prelude::*;
use std::{io::{self, BufRead, BufReader, Read}, path::PathBuf, path::Path};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::sync_channel;
use std::thread;
use std::fs::File;
use flate2::read::GzDecoder;
use rayon::prelude::*;

use std::collections::HashMap;

use arrow::array::{ArrayRef, RecordBatch, StringArray, StringDictionaryBuilder, UInt32Builder};
use arrow::datatypes::{DataType, Field, Schema, UInt32Type};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

pub mod classifier;
pub mod jats;
pub mod paragraphs;
pub mod scanner;

/// Read a shard into a String, transparently gunzipping if the file
/// starts with the gzip magic bytes (0x1f 0x8b).
fn read_shard_to_string(path: &Path) -> io::Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let is_gzip = {
        let buf = reader.fill_buf()?;
        buf.len() >= 2 && buf[0] == 0x1f && buf[1] == 0x8b
    };
    let mut s = String::new();
    if is_gzip {
        GzDecoder::new(reader).read_to_string(&mut s)?;
    } else {
        reader.read_to_string(&mut s)?;
    }
    Ok(s)
}

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

fn scan_shard_inner(
    searcher: &scanner::PatternIndex,
    shard_path: &Path,
    output_path: &Path,
    filter_model: Option<&classifier::TfidfClassifier>,
    scan_pool: Option<&rayon::ThreadPool>,
) -> PyResult<(u64, u64, u64, u64)> {
    let xml_string = read_shard_to_string(shard_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("failed to read {}: {}", shard_path.display(), e)))?;
    let shard_doc = jats::parse_document(&xml_string)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("failed to parse {}: {}", shard_path.display(), e)))?;
    let article_nodes: Vec<_> = jats::article_nodes(&shard_doc).collect();
    let articles_total = article_nodes.len() as u64;
    let articles_scanned = Arc::new(AtomicU64::new(0));
    let articles_filtered = Arc::new(AtomicU64::new(0));

    let schema = hits_schema();

    let out_file = File::create(output_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("failed to create {}: {}", output_path.display(), e)))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        .build();
    let mut writer = ArrowWriter::try_new(out_file, schema.clone(), Some(props))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("parquet writer init failed: {}", e)))?;

    let n_threads = scan_pool
        .map(|p| p.current_num_threads())
        .unwrap_or_else(|| rayon::current_num_threads())
        .max(1);
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

    let run_scan = || -> PyResult<()> {
        article_nodes
            .par_iter()
            .try_for_each_with(tx.clone(), |tx, (pmcid, node)| {
                let article = jats::parse_article_node(*node);

                let outcome = scanner::search_article_to_batch(searcher, &article, pmcid, &schema, &filter_model)?;
                drop(article);
                match outcome {
                    scanner::ScanOutcome::Filtered => {
                        articles_filtered.fetch_add(1, Ordering::Relaxed);
                    }
                    scanner::ScanOutcome::Scanned(maybe_batch) => {
                        articles_scanned.fetch_add(1, Ordering::Relaxed);
                        if let Some(batch) = maybe_batch {
                            tx.send(batch).map_err(|e| {
                                pyo3::exceptions::PyRuntimeError::new_err(format!("writer thread died: {}", e))
                            })?;
                        }
                    }
                }
                Ok(())
            })
    };

    let producer_result: PyResult<()> = match scan_pool {
        Some(pool) => pool.install(run_scan),
        None => run_scan(),
    };

    drop(tx);

    let writer_result = writer_handle
        .join()
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("writer thread panicked"))?;

    producer_result?;
    let total = writer_result
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("parquet write failed: {}", e)))?;

    Ok((
        articles_total,
        articles_scanned.load(Ordering::Relaxed),
        articles_filtered.load(Ordering::Relaxed),
        total,
    ))
}

#[pyfunction]
fn scan_shard(
    searcher: &scanner::PatternIndex,
    shard_path: String,
    output_path: String,
    filter_model: Option<&classifier::TfidfClassifier>
) -> PyResult<(u64, u64, u64, u64)> {
    scan_shard_inner(
        searcher,
        Path::new(&shard_path),
        Path::new(&output_path),
        filter_model,
        None,
    )
}

#[pyfunction]
#[pyo3(signature = (searcher, shard_paths, output_paths, filter_model=None, n_decompress=None))]
fn scan_shards(
    py: Python<'_>,
    searcher: &scanner::PatternIndex,
    shard_paths: Vec<String>,
    output_paths: Vec<String>,
    filter_model: Option<&classifier::TfidfClassifier>,
    n_decompress: Option<usize>,
) -> PyResult<(u64, u64, u64, u64)> {
    if shard_paths.len() != output_paths.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "shard_paths ({}) and output_paths ({}) length mismatch",
            shard_paths.len(),
            output_paths.len()
        )));
    }

    let ncpu = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let n_decomp = n_decompress
        .unwrap_or_else(|| ((ncpu + 3) / 4).max(1))
        .max(1);
    let n_scan = ncpu.saturating_sub(n_decomp).max(1);

    let scan_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_scan)
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("scan pool init failed: {}", e)))?;

    let work: Vec<(PathBuf, PathBuf)> = shard_paths
        .into_iter()
        .zip(output_paths.into_iter())
        .map(|(s, o)| (PathBuf::from(s), PathBuf::from(o)))
        .collect();
    let work_iter = Arc::new(Mutex::new(work.into_iter()));

    let total_articles = AtomicU64::new(0);
    let total_scanned = AtomicU64::new(0);
    let total_filtered = AtomicU64::new(0);
    let total_hits = AtomicU64::new(0);

    let result: PyResult<()> = py.allow_threads(|| {
        thread::scope(|s| -> PyResult<()> {
            let mut handles = Vec::with_capacity(n_decomp);
            for _ in 0..n_decomp {
                let work_iter = work_iter.clone();
                let scan_pool = &scan_pool;
                let total_articles = &total_articles;
                let total_scanned = &total_scanned;
                let total_filtered = &total_filtered;
                let total_hits = &total_hits;
                let h = s.spawn(move || -> PyResult<()> {
                    loop {
                        let next = {
                            let mut guard = work_iter.lock().unwrap();
                            guard.next()
                        };
                        let Some((shard, output)) = next else { break Ok(()) };
                        let (a, b, c, d) = scan_shard_inner(
                            searcher,
                            &shard,
                            &output,
                            filter_model,
                            Some(scan_pool),
                        )?;
                        total_articles.fetch_add(a, Ordering::Relaxed);
                        total_scanned.fetch_add(b, Ordering::Relaxed);
                        total_filtered.fetch_add(c, Ordering::Relaxed);
                        total_hits.fetch_add(d, Ordering::Relaxed);
                    }
                });
                handles.push(h);
            }
            for h in handles {
                match h.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => return Err(e),
                    Err(_) => {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            "decompression worker panicked",
                        ))
                    }
                }
            }
            Ok(())
        })
    });
    result?;

    Ok((
        total_articles.load(Ordering::Relaxed),
        total_scanned.load(Ordering::Relaxed),
        total_filtered.load(Ordering::Relaxed),
        total_hits.load(Ordering::Relaxed),
    ))
}



fn paragraphs_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new_dictionary("pmcid", DataType::UInt32, DataType::Utf8, false),
        Field::new_dictionary("section", DataType::UInt32, DataType::Utf8, false),
        Field::new("offset", DataType::UInt32, false),
        Field::new_dictionary("sentence", DataType::UInt32, DataType::Utf8, false),
        Field::new("paragraph", DataType::Utf8, false),
    ]))
}

/// Extract paragraphs for every `(section, offset)` request in
/// `requests_by_pmcid` whose PMCID is present in `shard_path`. Writes a
/// parquet file with columns `(pmcid, section, offset, paragraph)` that can
/// be inner-joined onto the hits parquet by `(pmcid, section, offset)`.
///
/// Returns the number of paragraph rows written.
#[pyfunction]
fn extract_paragraphs_for_shard(
    shard_path: String,
    requests_by_pmcid: HashMap<String, Vec<(String, u32, String)>>,
    output_path: String,
) -> PyResult<u64> {
    let path = Path::new(&shard_path);
    let xml_string = read_shard_to_string(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("failed to read {}: {}", path.display(), e)))?;
    let shard_doc = jats::parse_document(&xml_string)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("failed to parse {}: {}", path.display(), e)))?;

    let schema = paragraphs_schema();
    let out_path = PathBuf::from(&output_path);
    let out_file = File::create(&out_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("failed to create {}: {}", out_path.display(), e)))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        .build();
    let mut writer = ArrowWriter::try_new(out_file, schema.clone(), Some(props))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("parquet writer init failed: {}", e)))?;

    let mut pmcid_b: StringDictionaryBuilder<UInt32Type> = StringDictionaryBuilder::new();
    let mut section_b: StringDictionaryBuilder<UInt32Type> = StringDictionaryBuilder::new();
    let mut offset_b = UInt32Builder::new();
    let mut sentence_b: StringDictionaryBuilder<UInt32Type> = StringDictionaryBuilder::new();
    let mut paragraph_b: Vec<String> = Vec::new();

    let mut n_rows: u64 = 0;

    for (pmcid, node) in jats::article_nodes(&shard_doc) {
        let Some(reqs) = requests_by_pmcid.get(&pmcid) else { continue };
        let typed: Vec<paragraphs::Request> = reqs
            .iter()
            .map(|(section, offset, sentence)| paragraphs::Request {
                section: section.clone(),
                offset: *offset,
                sentence: sentence.clone(),
            })
            .collect();
        for resolved in paragraphs::extract_paragraphs_for_article(node, &typed) {
            pmcid_b.append_value(&pmcid);
            section_b.append_value(&resolved.section);
            offset_b.append_value(resolved.offset);
            sentence_b.append_value(&resolved.sentence);
            paragraph_b.push(resolved.paragraph);
            n_rows += 1;
        }
    }

    if n_rows > 0 {
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(pmcid_b.finish()),
            Arc::new(section_b.finish()),
            Arc::new(offset_b.finish()),
            Arc::new(sentence_b.finish()),
            Arc::new(StringArray::from(paragraph_b)),
        ];
        let batch = RecordBatch::try_new(schema.clone(), arrays)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("batch build failed: {}", e)))?;
        writer.write(&batch)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("parquet write failed: {}", e)))?;
    }
    writer.close()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("parquet close failed: {}", e)))?;

    Ok(n_rows)
}

/// A Python module implemented in Rust.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_scanner, m)?)?;
    m.add_function(wrap_pyfunction!(scan_one, m)?)?;
    m.add_function(wrap_pyfunction!(scan_shard, m)?)?;
    m.add_function(wrap_pyfunction!(scan_shards, m)?)?;
    m.add_function(wrap_pyfunction!(extract_paragraphs_for_shard, m)?)?;

    m.add_class::<classifier::TfidfClassifier>()?;
    m.add_class::<classifier::ClassificationResult>()?;
    Ok(())
}
