mod common;

use common::{metadata_path, mini_shard, safetensors_path};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use arrow::datatypes::DataType;
use litscan_2::classifier::TfidfClassifier;
use litscan_2::scanner::build_search_automaton;
use litscan_2::{scan_shard_inner, ShardCounts};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("litscan2_{}_{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn terms(dir: &PathBuf) -> PathBuf {
    let p = dir.join("terms.tsv");
    File::create(&p).unwrap().write_all(b"11\tHOTAIR\n22\tMIR21\n33\tunfound\n").unwrap();
    p
}

fn read(path: PathBuf) -> (Vec<(String, DataType)>, Vec<arrow::array::RecordBatch>) {
    let b = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap()).unwrap();
    let schema = b
        .schema()
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.data_type().clone()))
        .collect();
    (schema, b.build().unwrap().map(|r| r.unwrap()).collect())
}

#[test]
fn writes_mentions_articles_manifest() {
    let dir = tmpdir("manifest");
    let idx = build_search_automaton(terms(&dir)).unwrap();
    let clf = TfidfClassifier::new(safetensors_path(), metadata_path()).unwrap();
    let out = dir.join("out");

    let c = scan_shard_inner(&idx, &mini_shard(), &out, Some(&clf), None, None).unwrap();
    // PMC100: HOTAIR in title + abstract, MIR21 in results; PMC300: MIR21 in discussion.
    assert_eq!(
        c,
        ShardCounts { articles_total: 4, articles_scanned: 4, articles_below_threshold: 0, mentions: 4, notices: 1 }
    );

    let (schema, batches) = read(out.join("mentions/mini_shard.parquet"));
    let names: Vec<&str> = schema.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["term_id", "pmcid", "section", "symbol", "sentence", "sentence_offset"]);
    assert_eq!(schema[0].1, DataType::Int64);
    assert_eq!(schema[5].1, DataType::Int32);
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 4);

    let (schema, batches) = read(out.join("articles/mini_shard.parquet"));
    let names: Vec<&str> = schema.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        ["pmcid", "pmid", "doi", "title", "abstract", "authors", "journal", "year", "article_type", "rna_probability"]
    );
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 2, "only articles with mentions");
    let b = &batches[0];
    let col = |n: &str| b.column_by_name(n).unwrap().clone();
    let authors = arrow::array::as_string_array(&col("authors")).value(0).to_string();
    assert_eq!(authors, "Smith, Jo; Núñez, Ana");
    let year = col("year");
    assert_eq!(arrow::array::as_primitive_array::<arrow::datatypes::Int32Type>(&year).value(0), 2020);
    let proba = col("rna_probability");
    assert!(proba.is_valid(0) && proba.is_valid(1));

    let m: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("manifest/mini_shard.json")).unwrap()).unwrap();
    assert_eq!(m["shard"], "mini_shard");
    assert_eq!(m["n_terms"], 3);
    assert_eq!(m["mentions"], 4);
    assert_eq!(m["articles_total"], 4);
    assert_eq!(m["notices"], 1);

    // Notices come from every article, including PMC400, which has no mentions.
    let (schema, batches) = read(out.join("notices/mini_shard.parquet"));
    let names: Vec<&str> = schema.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["pmcid", "article_type", "relation", "target_id_type", "target_id"]);
    let b = &batches[0];
    let s = |n: &str| arrow::array::as_string_array(b.column_by_name(n).unwrap()).value(0).to_string();
    assert_eq!(
        [s("pmcid"), s("article_type"), s("relation"), s("target_id_type"), s("target_id")],
        ["PMC400", "retraction", "retracted-article", "pmcid", "PMC100"]
    );
    assert_eq!(m["terms_sha256"], idx.terms_sha256);
    assert_eq!(m["shard_sha256"].as_str().unwrap().len(), 64);
    for k in ["litscan_version", "started_at", "finished_at", "articles_scanned", "articles_below_threshold"] {
        assert!(!m[k].is_null(), "manifest missing {k}");
    }

    // Idempotent: a re-run overwrites and gives identical counts.
    let c2 = scan_shard_inner(&idx, &mini_shard(), &out, Some(&clf), None, None).unwrap();
    assert_eq!(c, c2);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn min_probability_counts_skipped_articles() {
    let dir = tmpdir("threshold");
    let idx = build_search_automaton(terms(&dir)).unwrap();
    let clf = TfidfClassifier::new(safetensors_path(), metadata_path()).unwrap();
    let out = dir.join("out");

    // Threshold above any probability: every article with an abstract is
    // skipped. PMC400 has no abstract, so it is scanned (and has no hits).
    let c = scan_shard_inner(&idx, &mini_shard(), &out, Some(&clf), Some(1.01), None).unwrap();
    assert_eq!(c.articles_below_threshold, 3);
    assert_eq!(c.articles_scanned, 1);
    assert_eq!(c.mentions, 0);
    assert_eq!(c.notices, 1, "notices don't depend on the classifier");
    let (_, batches) = read(out.join("articles/mini_shard.parquet"));
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 0);
    assert!(out.join("manifest/mini_shard.json").exists());

    assert!(scan_shard_inner(&idx, &mini_shard(), &out, None, Some(0.5), None).is_err());
    std::fs::remove_dir_all(dir).ok();
}
