use serde::Serialize;
#[allow(unused_code)]

use serde::{Deserialize};
use std::fs::File;
use std::io::{BufWriter, Read};
use std::path::Path;


#[derive(Deserialize, Clone)]
struct PerChunkScore {
    pub score_dns: f64,
    pub score_dpi: f64,
    pub bytes: u64,
    pub flows: u64
}

#[derive(Deserialize, Clone)]
struct PerSampleScore {
    pub sample_id: String,
    pub is_mal: bool,
    pub scores: Vec<PerChunkScore>
}

#[derive(Deserialize, Clone)]
struct PerFoldScore {
    pub dns_score_scale: f64,
    pub samples: Vec<PerSampleScore>
}

#[derive(Deserialize, Clone)]
struct Scores {
    pub per_fold_scores: Vec<PerFoldScore>
}

#[derive(Deserialize, Clone)]
struct NestedScores {
    pub outter: Scores,
    pub folds: Vec<Scores>
}

fn read_inner_scores(data: serde_json::Value, hours: u32) -> Result<Scores, Box<dyn std::error::Error>> {
    let mut n_mal = 0;
    let mut n_ben = 0;
    let mut per_fold_scores = vec![];
    let mut num_dpi_ben_gt_09 = 0usize;
    let mut num_dns_ben_gt_800 = 0usize;
    for fold in 0..5 {
        let fold_scores = data.get(format!("fold-{}", fold)).unwrap();
        let mut dns_score_scale = fold_scores.get("dns_score_scale").unwrap().as_f64().unwrap();
        let mut samples = vec![];
        for (k, sample_value) in fold_scores.as_object().unwrap() {
            if *k != "dns_score_scale" {
                let label = sample_value.get("label").unwrap().as_str().unwrap();
                let is_mal = label.to_owned() == "mal";
                if is_mal {
                    n_mal += 1;
                } else {
                    n_ben += 1;
                }
                let mut chunks = vec![];
                for chunk_id in 0..(hours * 2) {
                    let chunk_value = sample_value.get(format!("{}", chunk_id)).unwrap();
                    let chunk_data: PerChunkScore = serde_json::from_value(chunk_value.clone()).unwrap();
                    chunks.push(chunk_data);
                }
                let sample = PerSampleScore {
                    sample_id: k.clone(),
                    is_mal: is_mal,
                    scores: chunks,
                };
                if !is_mal {
                    let dpi_score = sample.scores.iter().map(|x| x.score_dpi).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                    if dpi_score > 0.999 {
                        num_dpi_ben_gt_09 += 1;
                    }
                    let dns_score = sample.scores.iter().map(|x| x.score_dns).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                    if dns_score > 800.0 {
                        num_dns_ben_gt_800 += 1;
                    }
                }
                samples.push(sample);
            }
        }
        per_fold_scores.push(PerFoldScore { dns_score_scale, samples })
    }
    println!("n_mal: {}, n_ben: {}", n_mal, n_ben);
    Ok(Scores { per_fold_scores })
}

fn read_and_deserialize_json<P: AsRef<Path>>(path: P, hours: u32) -> Result<NestedScores, Box<dyn std::error::Error>> {
    // Open the file
    let mut file = File::open(path)?;

    // Read the contents of the file into a string
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    // Deserialize the JSON string into `YourStruct`
    let data: serde_json::Value = serde_json::from_str(&contents)?;
    let outter = data.get("outter").unwrap();
    let outter = read_inner_scores(outter.clone(), hours).unwrap();
    let mut inner = vec![];
    for fold in 0..5 {
        let fold_scores = data.get(format!("fold-{}", fold)).unwrap();
        let fold_scores = read_inner_scores(fold_scores.clone(), hours).unwrap();
        inner.push(fold_scores);
    }


    Ok(NestedScores { outter: outter, folds: inner })
}

fn detection_process(
    sample: &PerSampleScore,
    mut thres_dns_on: f64,
    thres_dns_off: f64,
    thres_dpi: f64,
    dns_quant_scale: f64,
    chunks: u32,
    gt_label: u32,
    perf_dns_on: &mut Prf1,
    perf_dns_off: &mut Prf1,
    perf_dpi: &mut Prf1,
    per_chunk_off_path_ues: &mut Vec<u64>,
    // per_chunk_total_flows: &mut Vec<u64>,
    // per_chunk_off_path_flows: &mut Vec<u64>,
    // per_chunk_total_bytes: &mut Vec<u64>,
    // per_chunk_off_path_bytes: &mut Vec<u64>,
) -> (u32, u64, u64, u64, u64) {
    let mut total_bytes = 0;
    let mut total_flows = 0;
    let mut dpi_bytes = 0;
    let mut dpi_flows = 0;
    //thres_dns_on = 0.9999;
    let thres_dns_inv_on = ((thres_dns_on / (1f64 - thres_dns_on)).ln() * dns_quant_scale).round();
    let thres_dns_inv_off = ((thres_dns_off / (1f64 - thres_dns_off)).ln() * dns_quant_scale).round();
    let mut is_dns_phase = true; // DNS phase
    for cid in 0..(chunks as usize) {
        let score_dns = sample.scores[cid].score_dns;
        let score_dpi = sample.scores[cid].score_dpi;
        total_bytes += sample.scores[cid].bytes;
        total_flows += sample.scores[cid].flows;
        // per_chunk_total_flows[cid] += sample.scores[cid].flows;
        // per_chunk_total_bytes[cid] += sample.scores[cid].bytes;
        if is_dns_phase {
            let pred_label_dns_on = (score_dns >= thres_dns_inv_on) as u32;
            let pred_label_dns_off = (score_dns >= thres_dns_inv_off) as u32;
            perf_dns_off.add_test_sample(gt_label, pred_label_dns_off);
            perf_dns_on.add_test_sample(gt_label, pred_label_dns_on);
            if score_dns >= thres_dns_inv_on {
                //println!("dns_on: {} {} {}", score_dns, thres_dns_inv_on, thres_dns_inv_off);
                //perf_dns_on.add_test_sample(gt_label, 1);
                return (1, total_bytes, total_flows, dpi_bytes, dpi_flows);
            } else if score_dns >= thres_dns_inv_off {
                is_dns_phase = false;
            }
        } else {
            per_chunk_off_path_ues[cid] += 1;
            // per_chunk_off_path_flows[cid] += sample.scores[cid].flows;
            // per_chunk_off_path_bytes[cid] += sample.scores[cid].bytes;
            dpi_bytes += sample.scores[cid].bytes;
            dpi_flows += sample.scores[cid].flows;
            let pred_label_dpi = (score_dpi >= thres_dpi) as u32;
            perf_dpi.add_test_sample(gt_label, pred_label_dpi);
            if score_dpi >= thres_dpi {
                //perf_dns_on.add_test_sample(gt_label, 0);
                return (1, total_bytes, total_flows, dpi_bytes, dpi_flows);
            } else {
                is_dns_phase = false;
            }
            is_dns_phase = false;
        }
    }
    //perf_dns_on.add_test_sample(gt_label, 0);
    return (0, total_bytes, total_flows, dpi_bytes, dpi_flows);
}


struct Prf1 {
    true_positives: u32,
    true_negatives: u32,
    false_positives: u32,
    false_negatives: u32,
}

impl Prf1 {
    /// Creates a new `Prf1` instance.
    pub fn new() -> Self {
        Prf1 {
            true_positives: 0,
            true_negatives: 0,
            false_positives: 0,
            false_negatives: 0,
        }
    }

    /// Add the result of a test sample, both gt_label and pred_label take value of either 0 or 1
    pub fn add_test_sample(&mut self, gt_label: u32, pred_label: u32) {
        match (gt_label, pred_label) {
            (1, 1) => self.true_positives += 1,
            (0, 0) => self.true_negatives += 1,
            (0, 1) => self.false_positives += 1,
            (1, 0) => self.false_negatives += 1,
            _ => {} // No action needed for invalid labels
        }
    }

    /// Returns precision, recall, f1, False positive rate and false negative rate
    pub fn calculate_scores(&self) -> (f64, f64, f64, f64, f64) {
        let tp = self.true_positives as f64;
        let tn = self.true_negatives as f64;
        let fp = self.false_positives as f64;
        let fneg = self.false_negatives as f64;

        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fneg > 0.0 { tp / (tp + fneg) } else { 0.0 };
        let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
        let false_positive_rate = if fp + tn > 0.0 { fp / (fp + tn) } else { f64::INFINITY };
        let false_negative_rate = if fneg + tp > 0.0 { fneg / (fneg + tp) } else { f64::INFINITY };

        (precision, recall, f1, false_positive_rate, false_negative_rate)
    }
}

trait Statistics {
    fn mean_std(&self) -> (f64, f64);
}

impl Statistics for Vec<f64> {
    fn mean_std(&self) -> (f64, f64) {
        let sum: f64 = self.iter().sum();
        let count = self.len() as f64;
        let mean = sum / count;

        let variance: f64 = self.iter().map(|value| {
            let diff = mean - *value;
            diff * diff
        }).sum::<f64>() / count;

        let std_deviation = variance.sqrt();

        (mean, std_deviation)
    }
}

#[derive(Debug, Serialize)]
struct ExpResult {
    pub p_m: f64,
    pub p_s: f64,
    pub r_m: f64,
    pub r_s: f64,
    pub f1_m: f64,
    pub f1_s: f64,
    pub fpr_m: f64,
    pub fpr_s: f64,
    pub fnr_m: f64,
    pub fnr_s: f64,
}

#[derive(Debug, Serialize)]
struct MultiExpResult {
    pub combined: ExpResult,
    pub dns_on: ExpResult,
    pub dns_off: ExpResult,
    pub dpi: ExpResult,
}

struct ExpResultTraitCache {
    pub l_p: Vec<f64>,
    pub l_r: Vec<f64>,
    pub l_f1: Vec<f64>,
    pub l_fpr: Vec<f64>,
    pub l_fnr: Vec<f64>,
}

impl ExpResultTraitCache {
    pub fn new(n: usize) -> Self {
        Self {
            l_p: Vec::with_capacity(n),
            l_r: Vec::with_capacity(n),
            l_f1: Vec::with_capacity(n),
            l_fpr: Vec::with_capacity(n),
            l_fnr: Vec::with_capacity(n),
        }
    }
    pub fn clear(&mut self) {
        self.l_p.clear();
        self.l_r.clear();
        self.l_f1.clear();
        self.l_fpr.clear();
        self.l_fnr.clear();
    }
}

trait ExpResultTrait {
    fn to_exp_result(&self, cache: &mut ExpResultTraitCache) -> ExpResult;
}

impl ExpResultTrait for Vec<Prf1> {
    fn to_exp_result(&self, cache: &mut ExpResultTraitCache) -> ExpResult {
        cache.clear();
        for it in self.iter() {
            let (p, r, f1, fpr, fnr) = it.calculate_scores();
            cache.l_p.push(p);
            cache.l_r.push(r);
            cache.l_f1.push(f1);
            cache.l_fpr.push(fpr);
            cache.l_fnr.push(fnr);
        }
        let (p_m, p_s) = cache.l_p.mean_std();
        let (r_m, r_s) = cache.l_r.mean_std();
        let (f1_m, f1_s) = cache.l_f1.mean_std();
        let (fpr_m, fpr_s) = cache.l_fpr.mean_std();
        let (fnr_m, fnr_s) = cache.l_fnr.mean_std();
        ExpResult {
            p_m,
            p_s,
            r_m,
            r_s,
            f1_m,
            f1_s,
            fpr_m,
            fpr_s,
            fnr_m,
            fnr_s,
        }
    }
}

fn print_per_chunk_stats(stats: &str, data: &Vec<Vec<u64>>) {
    let mut per_chunk_folds_data = vec![vec![0f64; data.len()]; data[0].len()];
    for (ifold, fold_data) in data.iter().enumerate() {
        for (ichunk, data) in fold_data.iter().enumerate() {
            per_chunk_folds_data[ichunk][ifold] = *data as f64;
        }
    }
    println!("{}:", stats);
    for (ichunk, data) in per_chunk_folds_data.iter().enumerate() {
        let (m, s) = data.mean_std();
        println!("  {}: {:.3}±{:.3}", ichunk, m, s);
    }
}

fn threshold_search(scores: &Scores, chunks: u32, fpr_target: f64, off_path_ue_target: f64, fold_id: usize) -> Option<(MultiExpResult, f64, f64, f64, String)> {
    let mut best_thres = None;
    let mut best_fnr = f64::MAX;
    let mut best_results = None;
    use zzz::ProgressBarIterExt as _;
    // let mut per_fold_num_total_flows = Vec::with_capacity(5);
    // let mut per_fold_num_off_path_flows = Vec::with_capacity(5);
    // let mut per_fold_num_total_bytes = Vec::with_capacity(5);
    // let mut per_fold_num_off_path_bytes = Vec::with_capacity(5);
    let mut per_fold_combined = Vec::with_capacity(5);
    let mut per_fold_dns_on = Vec::with_capacity(5);
    let mut per_fold_dns_off = Vec::with_capacity(5);
    let mut per_fold_dpi = Vec::with_capacity(5);
    let mut per_fold_num_ue_off_path = Vec::with_capacity(5);
    let mut per_fold_combined_cache = ExpResultTraitCache::new(5);
    let mut per_fold_dns_on_cache = ExpResultTraitCache::new(5);
    let mut per_fold_dns_off_cache = ExpResultTraitCache::new(5);
    let mut per_fold_dpi_cache = ExpResultTraitCache::new(5);
    let mut per_chunk_off_path_ues = vec![0u64; chunks as usize];
    //let mut candidates = vec![];
    // let step = if fold_id == 4 {
    //     66
    // } else {
    //     62
    // };
    let step1 = 54;
    let step2 = 66;

    let mut candidates = vec![];
    let mut logs = vec![];

    // you may want to use special step sizes to skip local minima during search to get better results
    // for dns_on_t in (40..1000).step_by(step1).into_iter().progress() {
    //     let dns_on = dns_on_t as f64 / 1000.0f64;
    //     for dns_off_t in (1..100) {
    //         let dns_off = dns_off_t as f64 / 100.0f64;
    //         for dpi_t in (1..1000).step_by(step2) {
    //            let dpi = dpi_t as f64 / 1000.0f64;

    for dns_on_t in (40..100).step_by(1).into_iter().progress() {
        let dns_on = dns_on_t as f64 / 100.0f64;
        for dns_off_t in (1..100) {
            let dns_off = dns_off_t as f64 / 100.0f64;
            for dpi_t in (1..100).step_by(1) {
                let dpi = dpi_t as f64 / 100.0f64;

                per_chunk_off_path_ues.fill(0);
                let mut total_forwarded_ues = 0u64;
                let mut total_chunks = 0u64;
                for per_fold_scores in scores.per_fold_scores.iter() {
                    // for each fold
                    let mut perf_combined = Prf1::new();
                    let mut perf_dns_on = Prf1::new();
                    let mut perf_dns_off = Prf1::new();
                    let mut perf_dpi = Prf1::new();
                    // let mut per_chunk_total_flows = vec![0u64; chunks as usize];
                    // let mut per_chunk_off_path_flows = vec![0u64; chunks as usize];
                    // let mut per_chunk_total_bytes = vec![0u64; chunks as usize];
                    // let mut per_chunk_off_path_bytes = vec![0u64; chunks as usize];
                    for sample in per_fold_scores.samples.iter() {
                        let gt_label = sample.is_mal as u32;
                        let mut forwarded = false;
                        let (pred_label, total_bytes, total_flows, dpi_bytes, dpi_flows) = detection_process(
                            sample,
                            dns_on,
                            dns_off,
                            dpi,
                            per_fold_scores.dns_score_scale,
                            chunks,
                            gt_label,
                            &mut perf_dns_on,
                            &mut perf_dns_off,
                            &mut perf_dpi,
                            &mut per_chunk_off_path_ues,
                            // &mut per_chunk_total_flows,
                            // &mut per_chunk_off_path_flows,
                            // &mut per_chunk_total_bytes,
                            // &mut per_chunk_off_path_bytes
                        );
                        perf_combined.add_test_sample(gt_label, pred_label);
                        total_chunks += chunks as u64;
                    }
                    per_fold_combined.push(perf_combined);
                    per_fold_dns_on.push(perf_dns_on);
                    per_fold_dns_off.push(perf_dns_off);
                    per_fold_dpi.push(perf_dpi);
                    per_fold_num_ue_off_path.push(per_chunk_off_path_ues.clone());
                }
                let total_off_path_ue_chunks = per_chunk_off_path_ues.iter().sum::<u64>();
                let result_combined = per_fold_combined.to_exp_result(&mut per_fold_combined_cache);
                let off_path_condition = total_off_path_ue_chunks as f64 / total_chunks as f64 <= off_path_ue_target;
                if result_combined.fpr_m <= fpr_target && off_path_condition {
                    if result_combined.fnr_m < best_fnr {
                        //println!("found better parameters for fpr_target={} off_path_ue_target={}! at dns_on={}, dns_off={}, dpi={}, fnr={}", fpr_target, off_path_ue_target, dns_on, dns_off, dpi, result_combined.fnr_m);
                        logs.push(format!("found better parameters for fpr_target={} off_path_ue_target={}! at dns_on={}, dns_off={}, dpi={}, fpr={}±{}, fnr={}±{}",
                            fpr_target, off_path_ue_target, dns_on, dns_off, dpi, result_combined.fpr_m, result_combined.fpr_s, result_combined.fnr_m, result_combined.fnr_s));
                        best_fnr = result_combined.fnr_m;
                        best_thres = Some((dns_on, dns_off, dpi));
                        let result_dns_ond = per_fold_dns_on.to_exp_result(&mut per_fold_dns_on_cache);
                        let result_dns_off = per_fold_dns_off.to_exp_result(&mut per_fold_dns_off_cache);
                        let result_dpi = per_fold_dpi.to_exp_result(&mut per_fold_dpi_cache);
                        best_results = Some(MultiExpResult {
                            combined: result_combined,
                            dns_on: result_dns_ond,
                            dns_off: result_dns_off,
                            dpi: result_dpi,
                        });
                        candidates.clear();
                        candidates.push((dns_on, dns_off, dpi));
                    } else if (best_fnr - result_combined.fnr_m).abs() < 1e-8 {
                        candidates.push((dns_on, dns_off, dpi));
                    }
                }
                per_fold_combined.clear();
                per_fold_dns_on.clear();
                per_fold_dns_off.clear();
                per_fold_dpi.clear();
                per_fold_num_ue_off_path.clear();
            }
        }
    }
    if let (Some((dns_on, dns_off, dpi)), Some(best_results)) = (best_thres, best_results) {
        logs.push(format!("found best parameters for fpr_target={} off_path_ue_target={}! at dns_on={}, dns_off={}, dpi={}", fpr_target, off_path_ue_target, dns_on, dns_off, dpi));
        Some((best_results, dns_on, dns_off, dpi, logs.join("\n")))
    } else {
        //println!("failed to find best parameters! for fpr_target={} off_path_ue_target={}", fpr_target, off_path_ue_target);
        logs.push(format!("failed to find best parameters! for fpr_target={} off_path_ue_target={}", fpr_target, off_path_ue_target));
        None
    }
}

fn eval_single_fold(scores: &PerFoldScore, chunks: u32, dns_on: f64, dns_off: f64, dpi: f64) -> ((f64, f64, f64, f64, f64), (f64, f64, f64, f64, f64), (f64, f64, f64, f64, f64), (f64, f64, f64, f64, f64)) {    
    let mut perf_combined = Prf1::new();
    let mut perf_dns_on = Prf1::new();
    let mut perf_dns_off = Prf1::new();
    let mut perf_dpi = Prf1::new();
    let mut per_chunk_off_path_ues = vec![0u64; chunks as usize];
    for sample in scores.samples.iter() {
        let gt_label = sample.is_mal as u32;
        let mut forwarded = false;
        let (pred_label, total_bytes, total_flows, dpi_bytes, dpi_flows) = detection_process(
            sample,
            dns_on,
            dns_off,
            dpi,
            scores.dns_score_scale,
            chunks,
            gt_label,
            &mut perf_dns_on,
            &mut perf_dns_off,
            &mut perf_dpi,
            &mut per_chunk_off_path_ues,
            // &mut per_chunk_total_flows,
            // &mut per_chunk_off_path_flows,
            // &mut per_chunk_total_bytes,
            // &mut per_chunk_off_path_bytes
        );
        perf_combined.add_test_sample(gt_label, pred_label);
    }

    (perf_combined.calculate_scores(), perf_dns_on.calculate_scores(), perf_dns_off.calculate_scores(), perf_dpi.calculate_scores())
}

fn run(path: &str) -> String {
    let data_hours = 4;
    let chunks = data_hours * 2;
    let off_path_target = 10.4f64;
    let fpr_target = 0.01;

    // replace scores.json with args.txt
    let args_file = path.replace("scores.json", "args.txt");
    // read everything from args.txt
    let mut args_file = File::open(args_file).unwrap();
    let mut args = String::new();
    args_file.read_to_string(&mut args).unwrap();

    let mut results = path.to_owned() + "\n" + &args.clone();

    match read_and_deserialize_json(path, data_hours) {
        Ok(scores) => {
            println!("done parsing, procceed with threshold searching");
            let mut all_p = vec![];
            let mut all_r = vec![];
            let mut all_f1 = vec![];
            let mut all_fpr = vec![];
            let mut all_fnr = vec![];
            let mut all_dns_on_p = vec![];
            let mut all_dns_on_r = vec![];
            let mut all_dns_on_f1 = vec![];
            let mut all_dns_on_fpr = vec![];
            let mut all_dns_on_fnr = vec![];
            let mut all_dns_off_p = vec![];
            let mut all_dns_off_r = vec![];
            let mut all_dns_off_f1 = vec![];
            let mut all_dns_off_fpr = vec![];
            let mut all_dns_off_fnr = vec![];
            let mut all_dpi_p = vec![];
            let mut all_dpi_r = vec![];
            let mut all_dpi_f1 = vec![];
            let mut all_dpi_fpr = vec![];
            let mut all_dpi_fnr = vec![];
            for fold in 0usize..5 {
                let x = threshold_search(&scores.folds[fold], chunks, fpr_target, off_path_target, fold);
                if x.is_none() {
                    results += &format!("failed to find best parameters for fold {}! fpr_target={} off_path_ue_target={}", fold, fpr_target, off_path_target);
                    continue;
                }
                let (_, dns_on, dns_off, dpi, logs_inner) = x.unwrap();
                // let dns_on = 0.67;
                // let dns_off = 0.48;
                // let dpi = 0.71;
                results += &format!("{}\n", logs_inner);
                let (
                    (p, r, f1, fpr, fnr),
                    (dns_on_p, dns_on_r, dns_on_f1, dns_on_fpr, dns_on_fnr),
                    (dns_off_p, dns_off_r, dns_off_f1, dns_off_fpr, dns_off_fnr),
                    (dpi_p, dpi_r, dpi_f1, dpi_fpr, dpi_fnr),
                ) = eval_single_fold(&scores.outter.per_fold_scores[fold], chunks, dns_on, dns_off, dpi);
                all_p.push(p);
                all_r.push(r);
                all_f1.push(f1);
                all_fpr.push(fpr);
                all_fnr.push(fnr);
                all_dns_on_p.push(dns_on_p);
                all_dns_on_r.push(dns_on_r);
                all_dns_on_f1.push(dns_on_f1);
                all_dns_on_fpr.push(dns_on_fpr);
                all_dns_on_fnr.push(dns_on_fnr);
                all_dns_off_p.push(dns_off_p);
                all_dns_off_r.push(dns_off_r);
                all_dns_off_f1.push(dns_off_f1);
                all_dns_off_fpr.push(dns_off_fpr);
                all_dns_off_fnr.push(dns_off_fnr);
                all_dpi_p.push(dpi_p);
                all_dpi_r.push(dpi_r);
                all_dpi_f1.push(dpi_f1);
                all_dpi_fpr.push(dpi_fpr);
                all_dpi_fnr.push(dpi_fnr);
                //println!("fold: {} dns_on: {} dns_off: {} dpi: {} p: {} r: {} f1: {} fpr: {} fnr: {}", fold, dns_on, dns_off, dpi, p, r, f1, fpr, fnr);
                results += &format!("fold: {} dns_on: {} dns_off: {} dpi: {} p: {} r: {} f1: {} fpr: {} fnr: {}\n", fold, dns_on, dns_off, dpi, p, r, f1, fpr, fnr);
            }
            let (p_m, p_s) = all_p.mean_std();
            let (r_m, r_s) = all_r.mean_std();
            let (f1_m, f1_s) = all_f1.mean_std();
            let (fpr_m, fpr_s) = all_fpr.mean_std();
            let (fnr_m, fnr_s) = all_fnr.mean_std();
            //println!("combined: p: {:.3}±{:.3} r: {:.3}±{:.3} f1: {:.3}±{:.3} fpr: {:.3}±{:.3} fnr: {:.3}±{:.3}", p_m, p_s, r_m, r_s, f1_m, f1_s, fpr_m, fpr_s, fnr_m, fnr_s);
            results += &format!("combined: p: {:.3}±{:.3} r: {:.3}±{:.3} f1: {:.3}±{:.3} fpr: {:.3}±{:.3} fnr: {:.3}±{:.3}\n", p_m, p_s, r_m, r_s, f1_m, f1_s, fpr_m, fpr_s, fnr_m, fnr_s);
            let (dns_on_p_m, dns_on_p_s) = all_dns_on_p.mean_std();
            let (dns_on_r_m, dns_on_r_s) = all_dns_on_r.mean_std();
            let (dns_on_f1_m, dns_on_f1_s) = all_dns_on_f1.mean_std();
            let (dns_on_fpr_m, dns_on_fpr_s) = all_dns_on_fpr.mean_std();
            let (dns_on_fnr_m, dns_on_fnr_s) = all_dns_on_fnr.mean_std();
            //println!("dns_on: p: {:.3}±{:.3} r: {:.3}±{:.3} f1: {:.3}±{:.3} fpr: {:.3}±{:.3} fnr: {:.3}±{:.3}", dns_on_p_m, dns_on_p_s, dns_on_r_m, dns_on_r_s, dns_on_f1_m, dns_on_f1_s, dns_on_fpr_m, dns_on_fpr_s, dns_on_fnr_m, dns_on_fnr_s);
            results += &format!("dns_on: p: {:.3}±{:.3} r: {:.3}±{:.3} f1: {:.3}±{:.3} fpr: {:.3}±{:.3} fnr: {:.3}±{:.3}\n", dns_on_p_m, dns_on_p_s, dns_on_r_m, dns_on_r_s, dns_on_f1_m, dns_on_f1_s, dns_on_fpr_m, dns_on_fpr_s, dns_on_fnr_m, dns_on_fnr_s);
        },
        Err(e) => println!("Error reading or deserializing JSON: {}", e)
    }

    results += "\n\n\n";
    results += "====================\n";

    results.to_owned()
}

fn main() {
    use std::fs;
    use std::path::PathBuf;
    use std::io::prelude::*;

    let path = "../runs-cic-nested-final-2";
    let mut paths = vec![];
    for entry in fs::read_dir(path).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            paths.push(path);
        }
    }
    let score_files = paths.iter().filter_map(|path| {
        let path = path.join("scores.json");
        if path.exists() {
            Some(path)
        } else {
            None
        }
    }).collect::<Vec<_>>();
    println!("found {} score files", score_files.len());
    let mut all_results = String::new();
    // spawn threads to run each score file and collect results
    let mut handles = vec![];
    for score_file in score_files {
        let path = score_file.clone();
        let handle = std::thread::spawn(move || {
            run(path.to_str().unwrap())
        });
        handles.push(handle);
    }
    for handle in handles {
        let result = handle.join().unwrap();
        all_results += &result;
    }
    // write all results to all_results.txt
    let mut all_results_file = File::create("all_results.txt").unwrap();
    all_results_file.write_all(all_results.as_bytes()).unwrap();
    println!("all results written to all_results.txt");

}
