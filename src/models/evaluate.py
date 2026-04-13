
from collections import Counter
import glob
import json
import os
from pathlib import Path
import pickle
import time
from typing import List, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm

from trainer import Args, DomainWeightedClassifier, FlowDatasetTorch, DomainWeightedDataset, Quantization, collate_fn, keep_tld_and_one_label
from models import DPIModel, all_android_domains

def load_dns_model(filename: str, args: Args) :
    sd = torch.load(filename, map_location = 'cpu')
    model = DomainWeightedClassifier(sd['domains'], n_score = 1, score_scale = args.dns_score_scale)
    model.load_state_dict(sd['sd'])
    all_values = list(model.embd.weight.data.flatten().numpy())
    #model = model.cuda()
    model.quant = Quantization(all_values, max_bits = args.dns_qbits)
    model.eval()
    return model

def run_per_chunk_scores(list_ben_csv: List[Tuple[str, int, str]], list_mal_csv: List[Tuple[str, int, str]], model_dpi: DPIModel, model_dns: DomainWeightedClassifier, args, normalize_solution) :
    model_dpi.eval()
    
    max_chunk_id = 0
    
    sample_id2is_malicious = {}
    sample_id_chunk_idx2csv_filename = {}
    all_csv_files = list_ben_csv + list_mal_csv
    all_sample_ids = set([sample_id for _, _, sample_id in all_csv_files])
    for csv_filename, chunk_idx, sample_id in list_ben_csv + list_mal_csv :
        sample_id_chunk_idx2csv_filename[(sample_id, chunk_idx)] = csv_filename
        sample_id2is_malicious[sample_id] = '-mal.csv' in csv_filename
        max_chunk_id = max(max_chunk_id, chunk_idx)
        
    per_fold_result = {'dns_score_scale': model_dns.quant.scale}
    dns_ds = DomainWeightedDataset([], 768, args = args, train = False, all_domains = model_dns.all_domains)
    dpi_ds = FlowDatasetTorch([], args, flow_dropout_ratio = 0, normalize_solution = normalize_solution)
    n_chunks = 0
    used_time = []
    for sample_id in tqdm(all_sample_ids) :
        is_mal = sample_id[csv_filename]
        label = 'mal' if is_mal else 'ben'
        per_fold_result[csv_filename] = {'label': label}
        for chunk_id in range(max_chunk_id + 1) :
            per_fold_result[csv_filename][chunk_id] = {'score_dns': 0, 'score_dpi': 0, 'bytes': 0, 'flows': 0}

            csv_filename = sample_id_chunk_idx2csv_filename[(sample_id, chunk_id)]

            dns_file = csv_filename + '.dns.txt'
            with open(dns_file, 'r', encoding = 'utf-8') as fp :
                domains = [x.strip() for x in fp.readlines()]
                domains = [x for x in domains if x]
                if not args.dns_use_raw_domains :
                    domains = [keep_tld_and_one_label(s) for s in domains]
                if args.dns_filter_out_android_domain :
                    domains = [s for s in domains if s not in all_android_domains]
            (indices, counts, _) = dns_ds.sample_from_domains_and_label(domains, is_mal)
            score_dns = model_dns.get_feats_quant_fast(indices.unsqueeze_(0), counts.unsqueeze_(0), True)[0]
            
            per_fold_result[sample_id][chunk_id]['score_dns'] = float(score_dns[0])

            dpi_file = csv_filename
            dpi_x, dpi_xc, dpi_x_start_ts, dpi_x_end_ts, dpi_x_dpi_bytes, dpi_x_host_ids, dpi_mask, _, n_flows, n_bytes = collate_fn([dpi_ds.sample_from_csv_label(dpi_file, is_mal)])
            per_fold_result[sample_id][chunk_id]['flows'] = int(n_flows[0])
            per_fold_result[sample_id][chunk_id]['bytes'] = int(n_bytes[0])
            t0 = time.perf_counter()
            with torch.no_grad() :
                score_dpi = model_dpi(
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
    return per_fold_result
