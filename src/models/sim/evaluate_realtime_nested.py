
from collections import Counter, defaultdict
import glob
import json
import os
from pathlib import Path
import pickle
import time
from typing import List

import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm

from mixed_trainer_realtime_nested import Args, DomainWeightedClassifier, FlowDatasetTorch, DomainWeightedDataset, Quantization, collate_fn
from models import DPIModel, all_android_domains

from crccheck.crc import CrcArc
import math
import zlib

# Closest power of 2
def cl_p2(n):
    lg2 = int(math.log2(n))
    return 2 ** lg2



def crc32(domain: str, variant='32', reverse = False):
    max_bytes_per_label = 31
    max_bytes = max_bytes_per_label
    closest_power2 = cl_p2(max_bytes)
    sum_reg = b''
    sum_16 = b''
    splitdomains = reversed(domain.split('.')) if reverse else domain.split('.')
    for label in splitdomains:
        i = closest_power2
        temp_label = label
        while i >= 1:
            if len(temp_label) >= i:
                if i >= 16:
                    sum_16 += temp_label[:i].encode('ascii')   
                else:
                    sum_reg += temp_label[:i].encode('ascii')   
                temp_label = temp_label[i:]
            else:
                if i >= 16:
                    sum_16 += (0).to_bytes(i, byteorder='big')
                else:
                    sum_reg += (0).to_bytes(i, byteorder='big')
            i = int(i/2)
    total = (CrcArc.calc(sum_reg) if variant == '16' else zlib.crc32(sum_reg))
    if max_bytes >= 16:
        return (total + (CrcArc.calc(sum_16) if variant == '16' else zlib.crc32(sum_16))) % (2 ** int(variant))
    return total

def detect_hash_collision(domains: List[str], variant='32', reverse = False):
    hash2domains = defaultdict(set)
    for domain in domains:
        hash_val = crc32(domain, reverse=False)
        hash2domains[hash_val].add(domain)
    for hash_val, domain_set in hash2domains.items():
        if len(domain_set) > 1:
            print(f'hash {hash_val} has collision {domain_set}')
            return True
    return False


def load_dns_model(filename: str, args: Args) :
    sd = torch.load(filename, map_location = 'cpu')
    model = DomainWeightedClassifier(sd['domains'], n_score = 1, score_scale = args.dns_score_scale)
    model.load_state_dict(sd['sd'])
    all_values = list(model.embd.weight.data.flatten().numpy())
    #model = model.cuda()
    model.quant = Quantization(all_values, max_bits = args.dns_qbits)
    model.eval()
    return model

def load_dpi_model(filename: str, args: Args) :
    sd = torch.load(filename, map_location = 'cpu')
    model = DPIModel(65, num_category_inputs = 2, num_cat_per_category_input = 16, dpi_bytes = args.dpi_n_dpi_bytes)
    model.load_state_dict(sd)
    model = model.cuda()
    model.eval()
    return model

def load_args(fp) -> Args :
    args_dict: dict = pickle.load(fp)
    args = Args()
    for k, v in args_dict.items() :
        setattr(args, k, v)
    return args

TLDs = set()
with open('tlds_simple.txt', 'r') as fp :
    for line in fp.readlines() :
        line = line.strip()
        if line.startswith('//') :
            continue
        if line :
            n_parts = line.count('.')
            if n_parts <= 1 :
                TLDs.add(line)

def keep_tld_and_one_label(domain: str) :
    #return domain
    parts = domain.split('.')
    if len(parts) <= 2 :
        return domain
    # last two parts
    tld = '.'.join(parts[-2:])
    if tld in TLDs :
        return domain
    return tld


def run_per_chunk_scores(source_folder: str, models_folder: str) :
    with open(os.path.join(models_folder, 'args.pkl'), 'rb') as fp :
        args = load_args(fp)
    all_result = {}
    for fold in range(args.n_folds) :
        all_sample_ids = set()
        sample_id2is_malicious = {}
        max_chunk_id = 0
        source_folder_fold = os.path.join(source_folder, f'fold-{fold}', 'test')
        all_csvs = glob.glob(os.path.join(source_folder_fold, '*.csv'))
        for f in all_csvs :
            [label, sample_id, chunk_id] = Path(f).stem.split('-')
            all_sample_ids.add(sample_id)
            max_chunk_id = max(max_chunk_id, int(chunk_id))
            sample_id2is_malicious[sample_id] = int(label == 'mal')
        if args.dns_enabled :
            dns_model = load_dns_model(os.path.join(models_folder, f'fold-{fold}', 'dns.pth'), args)
        dpi_model = load_dpi_model(os.path.join(models_folder, f'fold-{fold}', 'dpi.pth'), args)
        with open(os.path.join(models_folder, f'fold-{fold}', 'normalize_solution.pkl'), 'rb') as fp :
            normalize_solution = pickle.load(fp)
        if args.dns_enabled :
            per_fold_result = {'dns_score_scale': dns_model.quant.scale}
            dns_ds = DomainWeightedDataset(source_folder_fold, 768, args = args, train = False, all_domains = dns_model.all_domains)
        else :
            per_fold_result = {}
        #continue
        dpi_ds = FlowDatasetTorch(source_folder_fold, args, flow_dropout_ratio = 0, normalize_solution = normalize_solution)
        n_chunks = 0
        used_time = []
        for sample_id in tqdm(all_sample_ids) :
            is_mal = sample_id2is_malicious[sample_id]
            label = 'mal' if is_mal else 'ben'
            per_fold_result[sample_id] = {'label': label}
            for chunk_id in range(max_chunk_id + 1) :
                per_fold_result[sample_id][chunk_id] = {'score_dns': 0, 'score_dpi': 0, 'bytes': 0, 'flows': 0}

                dns_file = os.path.join(source_folder_fold, f'{label}-{sample_id}-{chunk_id}.csv.dns.txt')
                with open(dns_file, 'r', encoding = 'utf-8') as fp :
                    domains = [x.strip() for x in fp.readlines()]
                    domains = [x for x in domains if x]
                    if not args.dns_use_raw_domains :
                        domains = [keep_tld_and_one_label(s) for s in domains]
                    if args.dns_filter_out_android_domain :
                        domains = [s for s in domains if s not in all_android_domains]
                    domains_2 = set(domains + list(dns_model.all_domains))
                    if detect_hash_collision(domains_2) :
                        print('** hash collision')
                if args.dns_enabled :
                    (indices, counts, _) = dns_ds.sample_from_domains_and_label(domains, is_mal)
                    score_dns = dns_model.get_feats_quant_fast(indices.unsqueeze_(0), counts.unsqueeze_(0), True)[0]
                else :
                    score_dns = [0]
                # if score_dns[0] > 800 and label == 'ben' :
                #     print(Counter(domains))
                per_fold_result[sample_id][chunk_id]['score_dns'] = float(score_dns[0])

                dpi_file = os.path.join(source_folder_fold, f'{label}-{sample_id}-{chunk_id}.csv')
                dpi_x, dpi_xc, dpi_x_start_ts, dpi_x_end_ts, dpi_x_dpi_bytes, dpi_x_host_ids, dpi_mask, _, n_flows, n_bytes = collate_fn([dpi_ds.sample_from_csv_label(dpi_file, is_mal)])
                per_fold_result[sample_id][chunk_id]['flows'] = int(n_flows[0])
                per_fold_result[sample_id][chunk_id]['bytes'] = int(n_bytes[0])
                t0 = time.perf_counter()
                with torch.no_grad() :
                    score_dpi = dpi_model(
                        dpi_x.cuda(),
                        dpi_xc.cuda(),
                        dpi_x_start_ts,
                        dpi_x_end_ts,
                        dpi_x_dpi_bytes.cuda(),
                        dpi_x_host_ids.cuda(),
                        dpi_mask.cuda()
                    ).softmax(dim = 1)[:, 1]
                    score_dpi = score_dpi.cpu().item()
                t1 = time.perf_counter()
                used_time.append(t1 - t0)
                per_fold_result[sample_id][chunk_id]['score_dpi'] = float(score_dpi)
                n_chunks += 1
        all_result[f'fold-{fold}'] = per_fold_result
        print('n_chunks', n_chunks)
        print('average time', np.mean(used_time))
        print('std time', np.std(used_time))
    return all_result

def detection_process(sample_dict, threshold_dns_on, threshold_dns_off, threshold_dpi, dns_quant_scale, hours) :
    total_bytes = sum([sample_dict[str(cid)]['bytes'] for cid in range(hours * 2)])
    total_flows = sum([sample_dict[str(cid)]['flows'] for cid in range(hours * 2)])
    num_flow_sent = 0
    num_bytes_sent = 0
    thres_dns_inv_on = int(np.round(np.log(threshold_dns_on / (1 - threshold_dns_on)) * dns_quant_scale))
    thres_dns_inv_off = int(np.round(np.log(threshold_dns_off / (1 - threshold_dns_off)) * dns_quant_scale))
    state = 0 # DNS phase
    for cid in range(hours * 2) :
        score_dns = sample_dict[str(cid)]['score_dns']
        score_dpi = sample_dict[str(cid)]['score_dpi']
        if state == 0 :
            if score_dns >= thres_dns_inv_on :
                return 1, total_bytes, num_bytes_sent, total_flows, num_flow_sent
            elif score_dns >= thres_dns_inv_off :
                state = 1
        elif state == 1 : # DPI phase
            num_flow_sent += sample_dict[str(cid)]['flows']
            num_bytes_sent += sample_dict[str(cid)]['bytes']
            if score_dpi >= threshold_dpi :
                return 1, total_bytes, num_bytes_sent, total_flows, num_flow_sent
            else :
                state = 0
    return 0, total_bytes, num_bytes_sent, total_flows, num_flow_sent

def precompute_scores(folder: str) :
    malware_csv = '/mnt/nvme0/malware_csv'
    results = {}
    run_dir = folder
    scores_file = os.path.join(run_dir, 'scores.json')
    if os.path.exists(scores_file) :
        print('skip', scores_file)
        return
    if not os.path.exists(os.path.join(run_dir, 'fold-4', 'fold-4')) :
        print('wrong', run_dir)
        return
    if os.path.exists(os.path.join(run_dir, 'fold-4')) :
        with open(os.path.join(run_dir, 'args.pkl'), 'rb') as fp :
            args: Args = load_args(fp)
        print('evaluating', run_dir, 'run name:', args.dataset)
        scores = run_per_chunk_scores(os.path.join(malware_csv, args.dataset), run_dir)
        results['outter'] = scores
    for fold in range(5) :
        cur_run_dir = os.path.join(run_dir, f'fold-{fold}')
        with open(os.path.join(cur_run_dir, 'args.pkl'), 'rb') as fp :
            args: Args = load_args(fp)
        print('evaluating', cur_run_dir, 'run name:', args.dataset)
        scores = run_per_chunk_scores(os.path.join(malware_csv, args.dataset, f'fold-{fold}'), cur_run_dir)
        results[f'fold-{fold}'] = scores
    with open(scores_file, 'w') as fp :
        json.dump(results, fp, indent = 2)

def prf1(gt, pred) :
    try :
        cm = confusion_matrix(gt, pred)
        # print(cm)
        # Compute TP, TN, FP, FN for each class
        tn = cm[0, 0]
        fn = cm[1, 0]
        tp = cm[1, 1]
        fp = cm[0, 1]

        # Compute TPR and FPR for each class
        tpr = tp / float(tp + fn)
        fpr = fp / float(fp + tn)
        tnr = tn / float(tn + fp)
        fnr = fn / float(tp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        return precision, recall, f1, fpr, fnr
    except Exception :
        return 0, 0, 0, 0, 0
    
def find_threshold(scores: dict, hours, fpr_target) :
    best_thres = None
    best_fnr = 100000
    best_result = None
    for dns_thres_on_t in tqdm(range(900, 1000)) :
        dns_thres_on = dns_thres_on_t / 1000.0
        for dns_thres_off_t in range(1, 100) :
            dns_thres_off = dns_thres_off_t / 1000.0
            for dpi_thres_t in range(900, 1000) :
                dpi_thres = dpi_thres_t / 1000.0
                per_fold_result = {}
                for fold_name, fold_samples in scores.items() :
                    per_fold_result[fold_name] = {}
                    per_fold_gt = []
                    per_fold_pred = []
                    for sample_id in fold_samples.keys() :
                        if sample_id == 'dns_score_scale' :
                            continue
                        is_mal = int(fold_samples[sample_id]['label'] == 'mal')
                        pred_is_mal, _, _, _, _ = detection_process(fold_samples[sample_id], dns_thres_on, dns_thres_off, dpi_thres, fold_samples['dns_score_scale'], hours)
                        per_fold_gt.append(is_mal)
                        per_fold_pred.append(int(pred_is_mal))
                    precision, recall, f1, fpr, fnr = prf1(per_fold_gt, per_fold_pred)
                    per_fold_result = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'fpr': fpr,
                        'fnr': fnr,
                    }
                precision = []
                recall = []
                f1 = []
                fpr = []
                fnr = []
                for fold_name in scores.keys() :
                    precision.append(per_fold_result['precision'])
                    recall.append(per_fold_result['recall'])
                    f1.append(per_fold_result['f1'])
                    fpr.append(per_fold_result['fpr'])
                    fnr.append(per_fold_result['fnr'])
                if np.mean(fpr) <= fpr_target :
                    if np.mean(fnr) < best_fnr :
                        best_fnr = np.mean(fnr)
                        best_thres = (dns_thres_on, dns_thres_off, dpi_thres)
                        best_result = {
                            'precision': {'m': np.mean(precision), 's': np.std(precision)},
                            'recall': {'m': np.mean(recall), 's': np.std(recall)},
                            'f1': {'m': np.mean(f1), 's': np.std(f1)},
                            'fpr': {'m': np.mean(fpr), 's': np.std(fpr)},
                            'fnr': {'m': np.mean(fnr), 's': np.std(fnr)},
                        }
    if best_thres :
        print(f'for {hours} hours run and target FPR of {fpr_target}, best threshold is found at {best_thres}')
        print('best result')
        print('precision', best_result['precision']['m'], best_result['precision']['s'])
        print('recall', best_result['recall']['m'], best_result['recall']['s'])
        print('f1', best_result['f1']['m'], best_result['f1']['s'])
        print('fpr', best_result['fpr']['m'], best_result['fpr']['s'])
        print('fnr', best_result['fnr']['m'], best_result['fnr']['s'])

def eval_worker(rank, dirs) :
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    for d in dirs :
        print(f'[{rank}] processing', d)
        precompute_scores(d)

import multiprocessing as mp

def main() :
    n_gpu = 3
    folder = 'runs-cic-nested-final-2'
    runs = os.listdir(folder)
    run_dir = [os.path.join(folder, p) for p in runs]
    mp.set_start_method('spawn')
    procs = []
    for rank in range(n_gpu) :
        os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
        to_proc_dirs = run_dir[rank::n_gpu]
        p = mp.Process(target = eval_worker, args = (rank, to_proc_dirs, ))
        p.start()
        procs.append(p)
        #eval_worker(rank, to_proc_dirs)
    for p in procs :
        p.join()

def threshold_find() :
    run = 'runs/1f92d42101d213150460bb90059c3a89'
    with open(os.path.join(run, 'scores.json'), 'r') as fp :
        score_dict = json.load(fp)
    find_threshold(score_dict, 1, 0.05)

if __name__ == '__main__' :
    main()
