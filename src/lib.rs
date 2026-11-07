use pyo3::prelude::*;
use std::{io::{self, Read}, path::PathBuf, path::Path};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::sync_channel;
use std::thread;
use std::fs::File;
use flate2::read::GzDecoder;
use rayon::prelude::*;

use std::collections::HashMap;

use arrow::array::{ArrayRef, RecordBatch, StringArray, StringDictionaryBuilder};
use arrow::datatypes::{DataType, Field, Schema, UInt32Type};

use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

pub mod classifier;
pub mod jats;
pub mod paragraphs;
pub mod scanner;

/// Decode shard bytes, transparently gunzipping if they start with the gzip
/// magic bytes (0x1f 0x8b). The double-UTF-8 repair is applied per article
/// by `for_each_article`, not here.
fn decode_shard(bytes: &[u8]) -> io::Result<String> {
    let mut s = String::new();
    if bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b {
        GzDecoder::new(bytes).read_to_string(&mut s)?;
    } else {
        s = String::from_utf8(bytes.to_vec()).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    }
    Ok(s)
}

/// Call `f(pmcid, article_node)` for every article with a pmcid, in parallel
/// on the current rayon pool, and return how many there were.
///
/// When the shard splits cleanly (`jats::split_articles`), each article is
/// repaired and parsed as its own small document inside the pool, so almost
/// no per-shard work is serial and no shard-sized tree is ever built.
/// Otherwise the whole document is repaired and parsed once.
fn for_each_article<F>(text: &str, f: F) -> Result<u64, scanner::Error>
where
    F: Fn(&str, roxmltree::Node) -> Result<(), scanner::Error> + Sync + Send,
{
    let count = AtomicU64::new(0);
    match jats::split_articles(text) {
        Some(pieces) => (0..pieces.len()).into_par_iter().try_for_each(|i| {
            let src = repair_double_utf8(pieces.document_source(i));
            let doc = jats::parse_document(&src)
                .map_err(|e| scanner::Error::Other(format!("article #{} in shard: {}", i + 1, e)))?;
            for (pmcid, node) in jats::article_nodes(&doc) {
                count.fetch_add(1, Ordering::Relaxed);
                f(&pmcid, node)?;
            }
            Ok::<(), scanner::Error>(())
        })?,
        None => {
            let src = repair_double_utf8(text.to_owned());
            let doc = jats::parse_document(&src).map_err(|e| scanner::Error::Other(e.to_string()))?;
            let nodes: Vec<_> = jats::article_nodes(&doc).collect();
            count.store(nodes.len() as u64, Ordering::Relaxed);
            nodes.par_iter().try_for_each(|(pmcid, node)| f(pmcid, *node))?;
        }
    }
    Ok(count.into_inner())
}

/// Europe PMC's bulk OA shards are UTF-8 that was decoded as Latin-1 and
/// re-encoded as UTF-8 (`Å` arrives as `Ã\u{85}`). If every char fits in a
/// byte and those bytes are valid UTF-8, undo it. Correctly encoded text with
/// non-ASCII chars essentially never passes both tests (e.g. a lone `é` is
/// invalid UTF-8 as a byte), so it is returned untouched.
pub fn repair_double_utf8(s: String) -> String {
    if s.is_ascii() || !s.chars().all(|c| (c as u32) < 256) {
        return s;
    }
    let bytes: Vec<u8> = s.chars().map(|c| c as u8).collect();
    String::from_utf8(bytes).unwrap_or(s)
}

#[cfg(test)]
mod tests {
    use super::repair_double_utf8 as fix;

    #[test]
    fn repairs_double_encoded_shard_text() {
        assert_eq!(fix("Naluai, \u{c3}\u{85}sa; CD4\u{e2}\u{80}\u{af}+".into()), "Naluai, Åsa; CD4\u{202f}+");
    }

    #[test]
    fn leaves_correct_text_alone() {
        for s in ["café", "αMIR21", "plain ascii", "Ångström µg"] {
            assert_eq!(fix(s.into()), s);
        }
    }
}

fn read_shard_to_string(path: &Path) -> io::Result<String> {
    decode_shard(&std::fs::read(path)?)
}

/// Output name for a shard: its basename minus `.gz` and `.xml`.
pub fn shard_name(path: &Path) -> String {
    let base = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let base = base.strip_suffix(".gz").unwrap_or(&base);
    base.strip_suffix(".xml").unwrap_or(base).to_string()
}

const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "+", env!("LITSCAN_GIT_SHA"));

fn now_rfc3339() -> String {
    humantime::format_rfc3339_seconds(std::time::SystemTime::now()).to_string()
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShardCounts {
    pub articles_total: u64,
    pub articles_scanned: u64,
    pub articles_below_threshold: u64,
    pub mentions: u64,
    pub notices: u64,
}

fn parquet_writer(path: &Path, schema: Arc<Schema>) -> Result<ArrowWriter<File>, scanner::Error> {
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        .build();
    Ok(ArrowWriter::try_new(File::create(path)?, schema, Some(props))?)
}

enum Msg {
    Mentions(RecordBatch),
    Article(scanner::ArticleRow),
    Notices(Vec<scanner::NoticeRow>),
}

/// Scan one shard into `output_dir/{mentions,articles,manifest}/<shard>.*`.
///
/// The manifest is the "done" marker: it is removed first and written last
/// (via rename), so it exists only when both parquet files are complete.
pub fn scan_shard_inner(
    searcher: &scanner::PatternIndex,
    shard_path: &Path,
    output_dir: &Path,
    filter_model: Option<&classifier::TfidfClassifier>,
    min_probability: Option<f64>,
    scan_pool: Option<&rayon::ThreadPool>,
) -> Result<ShardCounts, scanner::Error> {
    if min_probability.is_some() && filter_model.is_none() {
        return Err(scanner::Error::Other("min_probability requires a filter_model".into()));
    }
    let started_at = now_rfc3339();
    let name = shard_name(shard_path);
    let mentions_path = output_dir.join("mentions").join(format!("{name}.parquet"));
    let articles_path = output_dir.join("articles").join(format!("{name}.parquet"));
    let notices_path = output_dir.join("notices").join(format!("{name}.parquet"));
    let manifest_dir = output_dir.join("manifest");
    let manifest_path = manifest_dir.join(format!("{name}.json"));
    for d in ["mentions", "articles", "notices", "manifest"] {
        std::fs::create_dir_all(output_dir.join(d))?;
    }
    match std::fs::remove_file(&manifest_path) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e.into()),
        _ => {}
    }

    let ctx = |e: &dyn std::fmt::Display| scanner::Error::Other(format!("{}: {}", shard_path.display(), e));
    let shard_bytes = std::fs::read(shard_path).map_err(|e| ctx(&e))?;
    // Hash on a second thread while this one decompresses.
    let (shard_sha256, xml_string) = thread::scope(|s| {
        let sha = s.spawn(|| scanner::sha256_hex(&shard_bytes));
        let xml = decode_shard(&shard_bytes);
        (sha.join().expect("sha256 thread panicked"), xml)
    });
    let xml_string = xml_string.map_err(|e| ctx(&e))?;
    drop(shard_bytes);
    let articles_scanned = AtomicU64::new(0);
    let articles_below = AtomicU64::new(0);

    let schema = scanner::mentions_schema();
    let mut mentions_writer = parquet_writer(&mentions_path, schema.clone())?;

    let n_threads = scan_pool
        .map(|p| p.current_num_threads())
        .unwrap_or_else(rayon::current_num_threads)
        .max(1);
    let (tx, rx) = sync_channel::<Msg>(n_threads * 2);

    type Collected = (u64, Vec<scanner::ArticleRow>, Vec<scanner::NoticeRow>);
    let writer_handle = thread::spawn(move || -> Result<Collected, scanner::Error> {
        let mut total: u64 = 0;
        let mut rows = Vec::new();
        let mut notices = Vec::new();
        for msg in rx {
            match msg {
                Msg::Mentions(batch) => {
                    total += batch.num_rows() as u64;
                    mentions_writer.write(&batch)?;
                }
                Msg::Article(row) => rows.push(row),
                Msg::Notices(n) => notices.extend(n),
            }
        }
        mentions_writer.close()?;
        Ok((total, rows, notices))
    });

    let run_scan = || -> Result<u64, scanner::Error> {
        for_each_article(&xml_string, |pmcid, node| {
            let dead = |_| scanner::Error::Other("writer thread died".into());
            // Notices come from every article, hits or not.
            let found_notices = jats::notices(node);
            if !found_notices.is_empty() {
                let article_type = node.attribute("article-type").map(String::from);
                let rows = found_notices
                    .into_iter()
                    .map(|notice| scanner::NoticeRow { pmcid: pmcid.to_string(), article_type: article_type.clone(), notice })
                    .collect();
                tx.send(Msg::Notices(rows)).map_err(dead)?;
            }
            let article = jats::parse_article_node(node);
            let outcome = scanner::search_article_to_batch(
                searcher, &article, pmcid, &schema, filter_model, min_probability,
            )?;
            drop(article);
            match outcome {
                scanner::ScanOutcome::BelowThreshold => {
                    articles_below.fetch_add(1, Ordering::Relaxed);
                }
                scanner::ScanOutcome::Scanned(found) => {
                    articles_scanned.fetch_add(1, Ordering::Relaxed);
                    if let Some((batch, row)) = found {
                        tx.send(Msg::Mentions(batch)).map_err(dead)?;
                        tx.send(Msg::Article(row)).map_err(dead)?;
                    }
                }
            }
            Ok(())
        })
        .map_err(|e| ctx(&e))
    };

    let producer_result = match scan_pool {
        Some(pool) => pool.install(run_scan),
        None => run_scan(),
    };
    drop(tx);

    let writer_result = writer_handle
        .join()
        .map_err(|_| scanner::Error::Other("writer thread panicked".into()))?;
    let articles_total = producer_result?;
    let (mentions, mut rows, mut notice_rows) = writer_result?;

    rows.sort_unstable_by(|a, b| a.pmcid.cmp(&b.pmcid));
    let mut articles_writer = parquet_writer(&articles_path, scanner::articles_schema())?;
    articles_writer.write(&scanner::articles_to_batch(&rows)?)?;
    articles_writer.close()?;

    notice_rows.sort_unstable_by(|a, b| (&a.pmcid, &a.notice).cmp(&(&b.pmcid, &b.notice)));
    let mut notices_writer = parquet_writer(&notices_path, scanner::notices_schema())?;
    notices_writer.write(&scanner::notices_to_batch(&notice_rows)?)?;
    notices_writer.close()?;

    let counts = ShardCounts {
        articles_total,
        articles_scanned: articles_scanned.load(Ordering::Relaxed),
        articles_below_threshold: articles_below.load(Ordering::Relaxed),
        mentions,
        notices: notice_rows.len() as u64,
    };
    let manifest = serde_json::json!({
        "shard": name,
        "shard_sha256": shard_sha256,
        "terms_sha256": searcher.terms_sha256,
        "n_terms": searcher.n_terms,
        "litscan_version": VERSION,
        "articles_total": counts.articles_total,
        "articles_scanned": counts.articles_scanned,
        "articles_below_threshold": counts.articles_below_threshold,
        "mentions": counts.mentions,
        "notices": counts.notices,
        "started_at": started_at,
        "finished_at": now_rfc3339(),
    });
    let tmp = manifest_dir.join(format!(".{name}.json.tmp"));
    std::fs::write(&tmp, serde_json::to_vec_pretty(&manifest)?)?;
    std::fs::rename(&tmp, &manifest_path)?;

    Ok(counts)
}

type Counts = (u64, u64, u64, u64, u64);

fn counts_tuple(c: ShardCounts) -> Counts {
    (c.articles_total, c.articles_scanned, c.articles_below_threshold, c.mentions, c.notices)
}

/// Build the automaton from a `term_id<TAB>pattern` file.
#[pyfunction]
fn build_scanner(terms_path: String) -> PyResult<scanner::PatternIndex> {
    Ok(scanner::build_search_automaton(PathBuf::from(terms_path))?)
}

#[pyfunction]
fn scan_one(searcher: &scanner::PatternIndex, xml_string: String) -> PyResult<Vec<scanner::Hit>> {
    let article = jats::parse(&xml_string)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(scanner::search_article(searcher, &article)?)
}

/// Scan one shard. Returns (articles_total, articles_scanned, articles_below_threshold, mentions, notices).
#[pyfunction]
#[pyo3(signature = (searcher, shard_path, output_dir, filter_model=None, min_probability=None))]
fn scan_shard(
    py: Python<'_>,
    searcher: &scanner::PatternIndex,
    shard_path: String,
    output_dir: String,
    filter_model: Option<&classifier::TfidfClassifier>,
    min_probability: Option<f64>,
) -> PyResult<Counts> {
    let c = py.allow_threads(|| {
        scan_shard_inner(searcher, Path::new(&shard_path), Path::new(&output_dir), filter_model, min_probability, None)
    })?;
    Ok(counts_tuple(c))
}

#[pyfunction]
#[pyo3(signature = (searcher, shard_paths, output_dir, filter_model=None, min_probability=None, n_decompress=None, progress_callback=None))]
fn scan_shards(
    py: Python<'_>,
    searcher: &scanner::PatternIndex,
    shard_paths: Vec<String>,
    output_dir: String,
    filter_model: Option<&classifier::TfidfClassifier>,
    min_probability: Option<f64>,
    n_decompress: Option<usize>,
    progress_callback: Option<PyObject>,
) -> PyResult<Counts> {
    let ncpu = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let n_decomp = n_decompress
        .unwrap_or_else(|| ((ncpu + 3) / 4).max(1))
        .max(1);
    // Shard workers only read + gunzip + split; everything else runs on the
    // pool, and a worker sleeps while the pool scans its shard. So the pool
    // gets every core, and n_decompress mostly bounds shards held in memory.
    let n_scan = ncpu;

    let scan_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_scan)
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("scan pool init failed: {}", e)))?;

    let output_dir = PathBuf::from(output_dir);
    let work: Vec<PathBuf> = shard_paths.into_iter().map(PathBuf::from).collect();
    let work_iter = Arc::new(Mutex::new(work.into_iter()));
    let totals = Mutex::new(ShardCounts::default());
    let callback = progress_callback.as_ref();

    py.allow_threads(|| {
        thread::scope(|s| -> PyResult<()> {
            let mut handles = Vec::with_capacity(n_decomp);
            for _ in 0..n_decomp {
                let work_iter = work_iter.clone();
                let (scan_pool, totals, output_dir) = (&scan_pool, &totals, &output_dir);
                let h = s.spawn(move || -> PyResult<()> {
                    loop {
                        let next = work_iter.lock().unwrap().next();
                        let Some(shard) = next else { break Ok(()) };
                        let c = scan_shard_inner(searcher, &shard, output_dir, filter_model, min_probability, Some(scan_pool))?;
                        {
                            let mut t = totals.lock().unwrap();
                            t.articles_total += c.articles_total;
                            t.articles_scanned += c.articles_scanned;
                            t.articles_below_threshold += c.articles_below_threshold;
                            t.mentions += c.mentions;
                            t.notices += c.notices;
                        }
                        if let Some(cb) = callback {
                            Python::with_gil(|py| -> PyResult<()> {
                                cb.call1(py, (shard.to_string_lossy().into_owned(),))?;
                                Ok(())
                            })?;
                        }
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
    })?;

    Ok(counts_tuple(totals.into_inner().unwrap()))
}

fn paragraphs_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new_dictionary("pmcid", DataType::UInt32, DataType::Utf8, false),
        Field::new_dictionary("section", DataType::UInt32, DataType::Utf8, false),
        Field::new("sentence", DataType::Utf8, false),
        Field::new("paragraph", DataType::Utf8, false),
    ]))
}

/// Extract paragraphs for every `(section, sentence)` request in
/// `requests_by_pmcid` whose PMCID is present in `shard_path`. Writes a
/// parquet file with columns `(pmcid, section, sentence, paragraph)` that can
/// be joined onto mentions by `(pmcid, section, sentence)`.
///
/// Returns the number of paragraph rows written.
#[pyfunction]
fn extract_paragraphs_for_shard(
    shard_path: String,
    requests_by_pmcid: HashMap<String, Vec<(String, String)>>,
    output_path: String,
) -> PyResult<u64> {
    let path = Path::new(&shard_path);
    let xml_string = read_shard_to_string(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("failed to read {}: {}", path.display(), e)))?;

    // Same per-article repair and parsing as the scanner, so sentences match.
    let found: Mutex<Vec<(String, paragraphs::ResolvedParagraph)>> = Mutex::new(Vec::new());
    for_each_article(&xml_string, |pmcid, node| {
        let Some(reqs) = requests_by_pmcid.get(pmcid) else { return Ok(()) };
        let typed: Vec<paragraphs::Request> = reqs
            .iter()
            .map(|(section, sentence)| paragraphs::Request { section: section.clone(), sentence: sentence.clone() })
            .collect();
        let resolved = paragraphs::extract_paragraphs_for_article(node, &typed);
        found.lock().unwrap().extend(resolved.into_iter().map(|r| (pmcid.to_string(), r)));
        Ok(())
    })
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("failed to parse {}: {}", path.display(), e)))?;
    let mut found = found.into_inner().unwrap();
    found.sort_by(|a, b| (&a.0, &a.1.section, &a.1.sentence).cmp(&(&b.0, &b.1.section, &b.1.sentence)));

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
    let mut sentence_b: Vec<String> = Vec::new();
    let mut paragraph_b: Vec<String> = Vec::new();

    let mut n_rows: u64 = 0;

    for (pmcid, resolved) in found {
        pmcid_b.append_value(&pmcid);
        section_b.append_value(&resolved.section);
        sentence_b.push(resolved.sentence);
        paragraph_b.push(resolved.paragraph);
        n_rows += 1;
    }

    if n_rows > 0 {
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(pmcid_b.finish()),
            Arc::new(section_b.finish()),
            Arc::new(StringArray::from(sentence_b)),
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
