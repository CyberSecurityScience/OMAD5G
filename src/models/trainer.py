
from collections import Counter
import copy
import glob
import math
import os
import pickle
import secrets
import sys
import time
from typing import List, Tuple
import einops
import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import preprocess_flow_stats
from utils import AvgMeter
from models import DPIModel, all_android_domains

TLDs = set()
with open('tlds.txt', 'r') as fp :
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

class Args :
    dns_enabled = True
    dns_ben_w = 3
    dns_mal_w = 1
    dns_score_scale = 1
    dns_weight_decay = 0.1
    dns_label_smoothing = 0.05
    dns_temperture = 0.2
    dns_lr = 1e-3
    dns_epochs = '500x400x450'
    dns_max_domains = 512
    dns_batch_size = 128
    dns_offset_prob = 1
    dns_offset_ratio = 0.5
    dns_mal_drop = 0.05
    dns_ben_drop = 0.05
    dns_use_raw_domains = False
    dns_increase_mal_domain_occurence = False
    dns_filter_out_android_domain = True
    dns_qbits = 16

    dpi_enabled = True
    dpi_enabled_fp16 = False
    dpi_norm_version = 'all' # one of 'none', 'nolog', 'all'
    dpi_posweight = 1
    dpi_temperture = 0.1
    dpi_lr = 3e-4
    dpi_label_smoothing = 0.01
    dpi_flow_dropout_ratio = 0.01
    dpi_max_flows = 4000
    dpi_n_dpi_bytes = 160
    dpi_epochs = '5x3x4'
    dpi_batch_size = 24


class Quantization :
    def __init__(self, values: List[float], q_bits = 9, max_bits = 16) -> None:
        if 0.0 not in values :
            values.append(0.0)
        max_val = np.max(np.abs(values))
        self.val2int_map = {}
        max_val_int = 1 << (q_bits - 1)
        self.scale = 1.0 / max_val * max_val_int
        print('quant scale', self.scale)
        self.max_range = 1 << (max_bits - 1)
        self.max_bits = max_bits
        for v in values :
            v_closet = ((v / max_val) * max_val_int)
            v_closet = int(round(v_closet))
            self.val2int_map[v] = v_closet
        #print(self.val2int_map)
        self.quant_range_lb = -(2 ** (self.max_bits - 1))
        self.quant_range_ub = (2 ** (self.max_bits - 1)) - 1

    def get_val(self, v: float) :
        assert v in self.val2int_map
        return self.val2int_map[v]

    def quantize_np(self, a: np.ndarray) :
        shape = a.shape
        dt = a.dtype
        a = a.flatten()
        values = [self.val2int_map[v] for v in a]
        values = np.asarray(values).astype(np.int32).reshape(shape)
        return values

#@njit
def reduce_2(values: List[int], lb: int, ub: int) -> int :
    if len(values) == 1 :
        return values[0]
    new_values = []
    for i in range(0, len(values) - 1, 2) :
        new_values.append(np.clip(values[i] + values[i + 1], float(lb), float(ub)))
    if len(values) % 2 == 1 :
        new_values.append(values[-1])
    return reduce_2(new_values, lb, ub)


class DomainWeightedDataset(Dataset) :
    def __init__(self, source_dns_files: List[str], max_domains: int, train, args: Args, all_domains = None, random_dropout_prob = 0, random_offset_prob = 0.03) -> None :
        super().__init__()
        self.samples = []
        all_requests = []
        for txt in source_dns_files :
            lbl = 'mal' if '-mal.csv.dns.txt' in txt else 'ben'
            with open(txt, 'r', encoding = 'utf-8') as fp :
                domains = [x.strip() for x in fp.readlines()]
                domains = [s for s in domains if s]
                if not args.dns_use_raw_domains :
                    domains = [keep_tld_and_one_label(s) for s in domains]
                if args.dns_filter_out_android_domain :
                    domains = [s for s in domains if s not in all_android_domains]
                all_requests.extend(domains)
            self.samples.append((domains, lbl == 'mal'))
        self.malicious_domains = set()
        self.benign_domains = set()
        for (domains, is_mal) in self.samples :
            if is_mal :
                self.malicious_domains.update(domains)
            else :
                self.benign_domains.update(domains)
        if all_domains is None :
            self.all_domains = set()
        else :
            self.all_domains = all_domains
        # self.all_domains = list(sorted(set(self.all_domains)))
        # print('Total', len(self.all_domains), 'domains')
        self.domain2idx = {}
        self.domain2idx['<PAD>'] = 0
        self.domain2idx['<UNKNOWN>'] = 1
        for i, (domain, count) in enumerate(Counter(all_requests).most_common()) :
            #self.domain2idx[domain] = i + 2
            if all_domains is None :
                self.all_domains.add(domain)
            if i + 1 >= max_domains :
                break
        self.all_domains = list(sorted(set(self.all_domains)))
        print('Total', len(self.all_domains), 'domains')
        for i, d in enumerate(self.all_domains) :
            self.domain2idx[d] = i + 2
        self.idx2domain = {v: k for k, v in self.domain2idx.items()}
        self.max_domains = max_domains
        if train :
            self.random_offset_prob = args.dns_offset_prob
            self.mal_dropout_prob = args.dns_mal_drop
            self.ben_dropout_prob = args.dns_ben_drop
            self.dns_increase_mal_domain_occurence = args.dns_increase_mal_domain_occurence
        else :
            self.random_offset_prob = -1
            self.mal_dropout_prob = -1
            self.ben_dropout_prob = -1
            self.dns_increase_mal_domain_occurence = False
        self.offset_ratio = args.dns_offset_ratio

    def __len__(self) :
        return len(self.samples)
    
    def sample_from_domains_and_label(self, domains, label) :
        ret_indices = torch.zeros(self.max_domains + 3, dtype = torch.long)
        ret_counts = torch.zeros(self.max_domains + 3, dtype = torch.float32)
        for i, (domain, count) in enumerate(Counter(domains).items()) :
            domain_index = self.domain2idx.get(domain, 1)
            # if np.random.rand() < self.mal_dropout_prob and domain in self.malicious_domains :
            #     domain_index = 1 # set a domain to UNKNOWN
            # if np.random.rand() < self.ben_dropout_prob and domain not in self.malicious_domains :
            #     domain_index = 1 # set a domain to UNKNOWN
            if np.random.rand() < self.mal_dropout_prob :
                domain_index = 1
            if np.random.rand() < self.random_offset_prob :
                count_tail = count * self.offset_ratio
                count = int(max(float(count) + np.random.normal(0, count_tail), 1))
            # if i >= len(ret_indices) :
            #     break
            ret_indices[domain_index] = domain_index
            ret_counts[domain_index] += float(count)
        return ret_indices, ret_counts, int(label)

    def __getitem__(self, idx) :
        if torch.is_tensor(idx) :
            idx = idx.tolist()
        domains, label = self.samples[idx]
        # if len(set(domains)) >= 768 :
        #     breakpoint()
        return self.sample_from_domains_and_label(domains, label)
        

class DomainWeightedClassifier(nn.Module) :
    def __init__(self, domains: set, n_score: int = 4, score_scale = 1) -> None:
        super().__init__()
        self.all_domains = domains
        self.idx2domain = {}
        # 0: padding, 1: unknown
        self.embd = nn.Embedding(len(domains) + 2, n_score)
        self.d_embd = n_score
        weights = []
        for i in range(n_score) :
            weights.append(1 << i)
        self.weights = nn.Parameter(torch.tensor(weights, dtype = torch.float32), requires_grad = False)
        self.score_scale = score_scale
        self.quant: Quantization = None

    def forward(self, domain_indices: torch.Tensor, counts: torch.Tensor) :
        embds = self.embd(domain_indices)
        n, d, e = embds.shape
        scores = torch.sum(embds * counts.view(n, d, 1), dim = 1)
        # N, E
        out = torch.matmul(scores, self.weights.data.view(e, 1)).view(n, 1) * self.score_scale
        return out
    
    def get_domain_reputation_scores(self) :
        if not self.idx2domain :
            self.idx2domain[0] = '<PAD>'
            self.idx2domain[1] = '<UNKNOWN>'
            self.all_domains = list(sorted(self.all_domains))
            for i, domain in enumerate(self.all_domains) :
                self.idx2domain[i + 2] = domain
        result = {}
        for idx, domain in self.idx2domain.items() :
            score_np = self.embd(torch.tensor([idx])).detach().cpu().numpy().flatten()
            score_quant = self.quant.quantize_np(score_np)
            result[domain] = int(score_quant)
        return result
    
    def get_feats_quant_fast(self, domain_indices: torch.Tensor, counts: torch.Tensor, use_correct) -> List[int] :
        all_scores = []
        for (indices, count) in zip(domain_indices, counts) :
            scores = np.zeros((self.d_embd, ), dtype = np.int32)
            for idx, cnt in zip(indices, count) :
                idx = idx.item()
                cnt = int(cnt.item())
                score_for_current_domain = self.embd.weight.data[idx].numpy()
                # if idx == 1 :
                #     score_for_current_domain[:] = 0 # UNKNOWN
                score_for_current_domain = self.quant.quantize_np(score_for_current_domain)
                scores += score_for_current_domain * cnt
                # for _ in range(cnt) :
                #     scores += score_for_current_domain
                    # if any(scores > self.quant_range_ub) or any(scores < self.quant_range_lb) :
                    #     print('warn', scores)
                    #scores = np.clip(scores, self.quant.quant_range_lb, self.quant.quant_range_ub)
            if use_correct :
                scores = np.clip(scores * self.weights.data.numpy(), self.quant.quant_range_lb, self.quant.quant_range_ub)
            score_single = reduce_2(scores.tolist(), self.quant.quant_range_lb, self.quant.quant_range_ub)
            all_scores.append(score_single)
        return np.asarray(all_scores).reshape(-1, 1)
    
class BCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None,label_smoothing=0.0, reduction='mean'):
        super(BCEWithLogitsLoss, self).__init__()
        assert 0 <= label_smoothing < 1, "label_smoothing value must be between 0 and 1."
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(pos_weight=pos_weight,reduction=reduction)

    def forward(self, input, target):
        if self.label_smoothing > 0:
            positive_smoothed_labels = 1.0 - self.label_smoothing
            negative_smoothed_labels = self.label_smoothing
            target = target * positive_smoothed_labels + \
                (1 - target) * negative_smoothed_labels

        loss = self.bce_with_logits(input, target)
        return loss
        
def train_dns(source_dns_files: List[str], args: Args) :
    dns_ds = DomainWeightedDataset(source_dns_files, max_domains = args.dns_max_domains, train = True, args = args, random_dropout_prob = 0.001)
    model = DomainWeightedClassifier(dns_ds.all_domains, n_score = 1, score_scale = args.dns_score_scale)
    model = model.cuda()
    opt = optim.AdamW(model.parameters(), args.dns_lr, (0.95, 0.999), weight_decay = args.dns_weight_decay)
    [n_epochs, s1, s2] = args.dns_epochs.split('x')
    [n_epochs, s1, s2] = [int(x) for x in [n_epochs, s1, s2]]
    sch = optim.lr_scheduler.MultiStepLR(opt, [s1, s2], gamma = 0.1)
    dl = DataLoader(dns_ds, batch_size = args.dns_batch_size, shuffle = True, num_workers = 8, pin_memory = True)
    print('total', len(dns_ds), 'samples, using', len(dl), 'batches')
    model.train()
    thres = 0.5
    w = torch.ones((1, ), dtype = torch.float32).cuda()
    w[0] = float(args.dns_mal_w) / float(args.dns_ben_w)
    loss_fn = BCEWithLogitsLoss(pos_weight = w, label_smoothing = args.dns_label_smoothing).cuda()
    temp = args.dns_temperture
    for epoch in range(n_epochs) :
        avg_loss = AvgMeter()
        all_gts = []
        all_preds = []
        for (indices, counts, labels) in dl :
            indices = indices.cuda()
            counts = counts.cuda()
            labels = labels.cuda().unsqueeze(-1).float()
            opt.zero_grad()
            out = model(indices, counts)
            out = out / temp
            pred_prob = out.detach().sigmoid()
            pred_labels = (pred_prob > thres).long().view(-1).cpu().numpy()
            all_gts.extend(labels.cpu().view(-1).numpy())
            all_preds.extend(pred_labels)
            loss = loss_fn(out, labels)
            loss.backward()
            opt.step()
            avg_loss(loss.item())
        sch.step()
        cm = confusion_matrix(all_gts, all_preds)
        tn = cm[0, 0]
        fn = cm[1, 0]
        tp = cm[1, 1]
        fp = cm[0, 1]
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        print(f'[{epoch + 1}/{n_epochs}] train_loss={avg_loss()} P={precision} R={recall} F1={f1}')
    model = model.cpu()
    all_values = list(model.embd.weight.data.flatten().numpy())
    model.quant = Quantization(all_values, max_bits = args.dns_qbits)
    model.all_domains = dns_ds.all_domains
    model.idx2domain = dns_ds.idx2domain
    return model


class FlowDatasetTorch(Dataset) :
    def __init__(self, list_csv_files: List[str], args: Args, category_columns = ['Feat 0', 'Feat 1'], flow_dropout_ratio = 0.5, normalize_solution = None, trainset = True) -> None:
        super().__init__()
        self.all_pcap_files = set()
        self.samples = list_csv_files
        self.category_columns = category_columns
        self.flow_dropout_ratio = flow_dropout_ratio
        self.normalize_solution = normalize_solution
        self.trainset = trainset
        self.args = args

    def __len__(self) :
        return len(self.samples)
    
    def sample_from_csv_label(self, csv_filename, is_malicious) :
        df: pd.DataFrame = pd.read_csv(csv_filename)
        n_bytes = 0
        n = df.shape[0]
        drop_indices = np.random.choice(df.index, int(n * self.flow_dropout_ratio), replace = False)
        df = df.drop(drop_indices)
        if df.shape[0] > self.args.dpi_max_flows and self.trainset :
            drop_indices = np.random.choice(df.index, df.shape[0] - self.args.dpi_max_flows, replace = False)
            df = df.drop(drop_indices)
        n_bytes += int(df['Feat 3'].sum())
        dpi_bytes = [bytearray.fromhex(h)[: self.args.dpi_n_dpi_bytes] for h in df['Content']]
        dpi_bytes = [h + b'\00' * (self.args.dpi_n_dpi_bytes - len(h)) for h in dpi_bytes]
        dpi_bytes = [np.array(h, dtype = np.uint8) for h in dpi_bytes]
        dpi_bytes = np.stack(dpi_bytes, axis = 0)
        col_start_ts = np.array(df['Start TS']).astype(np.int64)
        col_end_ts = np.array(df['End TS']).astype(np.int64)

        ip_counts = df['Host'].value_counts().reset_index()
        ip_counts.columns = ['Host', 'count']
        sorted_ips = ip_counts.sort_values(by='count', ascending=False)

        # Assign IDs
        if len(sorted_ips) > 500:
            sorted_ips['id'] = np.arange(1, 501).tolist() + [0] * (len(sorted_ips) - 500)
        else:
            sorted_ips['id'] = np.arange(1, len(sorted_ips) + 1)

        # Ensure each IP gets a unique ID
        unique_ids = np.random.permutation(sorted_ips['id'].unique())
        id_map = dict(zip(sorted_ips['id'].unique(), unique_ids))
        sorted_ips['id'] = sorted_ips['id'].apply(lambda x: id_map[x])
        df = df.merge(sorted_ips[['Host', 'id']], on='Host', how='left')
        host_ids = df['id']

        df = df.drop(['Content', 'Start TS', 'End TS', 'Host', 'id', 'DNS Query'], axis = 1)
        if self.normalize_solution is not None :
            for col in df.columns :
                df[col] = self.normalize_solution[col].apply(df[col])
        cat_cols = np.stack([df[col].to_numpy() for col in self.category_columns], axis = -1)
        label = int(is_malicious)
        df_clean = df.drop(self.category_columns, axis = 1)
        df_clean = df_clean.to_numpy()
        host_ids = host_ids.to_numpy()
        return df_clean, cat_cols, col_start_ts, col_end_ts, dpi_bytes, host_ids, label, n, n_bytes

    def __getitem__(self, i) :
        #return 1, 2, 3, 4, 5, 6, 7, 8, 9
        sample_filename = self.samples[i]
        is_malicious = '-mal.csv' in sample_filename
        return self.sample_from_csv_label(sample_filename, is_malicious)
        
def collate_fn(data) :
    x, xc, col_start_ts, col_end_ts, dpi_bytes, host_ids, y, n_flows, n_bytes = zip(*data)
    N = len(x)
    max_flows = max([f.shape[0] for f in x])
    n_num_feat = x[0].shape[1]
    n_cat_feat = xc[0].shape[1]
    num_dpi_bytes = dpi_bytes[0].shape[1]
    x_num = torch.zeros(N, max_flows, n_num_feat, dtype = torch.float64)
    x_cat = torch.zeros(N, max_flows, n_cat_feat, dtype = torch.int64)
    x_start_ts = torch.zeros(N, max_flows, dtype = torch.int64)
    x_end_ts = torch.zeros(N, max_flows, dtype = torch.int64)
    x_dpi_bytes = torch.zeros(N, max_flows, num_dpi_bytes, dtype = torch.uint8)
    x_host_ids = torch.zeros(N, max_flows, dtype = torch.int64)
    mask = torch.ones(N, max_flows, dtype = torch.bool)
    for i in range(N) :
        n_flow = x[i].shape[0]
        x_num[i, : n_flow, :] = torch.tensor(x[i])
        x_cat[i, : n_flow, :] = torch.tensor(xc[i])
        x_start_ts[i, : n_flow] = torch.tensor(col_start_ts[i])
        x_end_ts[i, : n_flow] = torch.tensor(col_end_ts[i])
        x_dpi_bytes[i, : n_flow, :] = torch.tensor(dpi_bytes[i])
        x_host_ids[i, : n_flow] = torch.tensor(host_ids[i])
        mask[i, : n_flow] = False
    return x_num.float(), x_cat, x_start_ts, x_end_ts, x_dpi_bytes, x_host_ids, mask, torch.tensor(y, dtype = torch.int64), list(n_flows), list(n_bytes)


def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count()))

def train_dpi(source_folder: str, normalize_solution, args: Args) :
    ds = FlowDatasetTorch(source_folder, args, flow_dropout_ratio = args.dpi_flow_dropout_ratio, normalize_solution = normalize_solution)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size = 24,
        num_workers = 20,
        drop_last = False,
        shuffle = True,
        collate_fn = collate_fn,
        timeout = 50,
        pin_memory = True,
        worker_init_fn = worker_init_fn
    )
    model = DPIModel(65, num_category_inputs = 2, num_cat_per_category_input = 16, dpi_bytes = args.dpi_n_dpi_bytes)
    model = model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr = args.dpi_lr, betas = (0.99, 0.999), weight_decay = 0.1)
    [n_epochs, s1, s2] = args.dpi_epochs.split('x')
    [n_epochs, s1, s2] = [int(x) for x in [n_epochs, s1, s2]]
    lrs = torch.optim.lr_scheduler.MultiStepLR(opt, [s1, s2], gamma = 0.1)
    loss_avg = AvgMeter()
    scaler = torch.amp.GradScaler(enabled = args.dpi_enabled_fp16)
    w = torch.ones((2, ), dtype = torch.float32).cuda()
    w[0] = args.dpi_posweight
    loss_fn = nn.CrossEntropyLoss(weight = w, label_smoothing = args.dpi_label_smoothing).cuda()
    temp = args.dpi_temperture
    for ep in range(n_epochs) :
        acc_avg = AvgMeter()
        print('train ep', ep)
        model.train()
        for x, xc, x_start_ts, x_end_ts, x_dpi_bytes, x_host_ids, mask, y, _, _ in dl :
            opt.zero_grad()
            x = x.cuda()
            y_gt = y.long().numpy()
            y = y.cuda().long()
            x_dpi_bytes = x_dpi_bytes.cuda()
            x_host_ids = x_host_ids.cuda()
            mask = mask.cuda()
            with torch.autocast('cuda', enabled = args.dpi_enabled_fp16) :
                y_pred = model(x, xc, x_start_ts, x_end_ts, x_dpi_bytes, x_host_ids, mask)
                #y_pred = torch.ones_like(y)
                #loss = F.binary_cross_entropy_with_logits(y_pred.view(-1), y.view(-1), pos_weight = w)
                #loss = F.cross_entropy(y_pred, y.view(-1))
                loss = loss_fn(y_pred / temp, y.view(-1))
            #y_pred_cat = (y_pred.sigmoid() > 0.5).long().cpu().numpy()
            y_pred_cat = y_pred.argmax(dim = 1).cpu().numpy()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            acc = np.mean((y_gt == y_pred_cat).astype(np.float32))# / y_gt.shape[0]
            acc_avg(acc)
            loss_avg(loss.item())
            print(f' - acc: {acc}, loss: {loss_avg()}, - acc_avg: {acc_avg()}')
            pass
        lrs.step()
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    return model


def detection_process(sample_dict, threshold_dns_on, threshold_dns_off, threshold_dpi, dns_quant_scale, hours) :
    total_bytes = sum([sample_dict[str(cid)]['bytes'] for cid in range(hours * 2)])
    total_flows = sum([sample_dict[str(cid)]['flows'] for cid in range(hours * 2)])
    num_flow_sent = 0
    num_bytes_sent = 0
    thres_dns_inv_on = int(np.round(np.log(threshold_dns_on / (1 - threshold_dns_on)) * dns_quant_scale))
    thres_dns_inv_off = int(np.round(np.log(threshold_dns_off / (1 - threshold_dns_off)) * dns_quant_scale))
    state = 0 # DNS phase
    total_chunks = 0
    total_off_path_chunks = 0
    for cid in range(hours * 2) :
        score_dns = sample_dict[str(cid)]['score_dns']
        score_dpi = sample_dict[str(cid)]['score_dpi']
        total_chunks += 1
        if state == 0 :
            if score_dns >= thres_dns_inv_on :
                return 1, total_bytes, num_bytes_sent, total_flows, num_flow_sent
            elif score_dns >= thres_dns_inv_off :
                state = 1
        elif state == 1 : # DPI phase
            num_flow_sent += sample_dict[str(cid)]['flows']
            num_bytes_sent += sample_dict[str(cid)]['bytes']
            total_off_path_chunks
            if score_dpi >= threshold_dpi :
                return 1, total_bytes, num_bytes_sent, total_flows, num_flow_sent
            else :
                state = 0
    return 0, total_bytes, num_bytes_sent, total_flows, num_flow_sent, total_chunks, total_off_path_chunks

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
        return 0, 0, 0, 1, 1
    
def find_threshold(scores: dict, hours = 1, fpr_target = 0.05, off_path_limit = 0.4) :
    best_fnr = 100000
    threshold_candidates = []
    for dns_thres_on_t in range(1, 100) :
        dns_thres_on = dns_thres_on_t / 100.0
        for dns_thres_off_t in range(1, 100) :
            dns_thres_off = dns_thres_off_t / 100.0
            for dpi_thres_t in range(1, 100) :
                dpi_thres = dpi_thres_t / 100.0
                per_fold_result = {}
                per_fold_total_chunks = 0
                per_fold_total_off_path_chunks = 0
                for fold_name, fold_samples in scores.items() :
                    per_fold_result[fold_name] = {}
                    per_fold_gt = []
                    per_fold_pred = []
                    for sample_id in fold_samples.keys() :
                        if sample_id == 'dns_score_scale' :
                            continue
                        is_mal = int(fold_samples[sample_id]['label'] == 'mal')
                        pred_is_mal, _, _, _, _, chunks, off_path_chunks = detection_process(fold_samples[sample_id], dns_thres_on, dns_thres_off, dpi_thres, fold_samples['dns_score_scale'], hours)
                        per_fold_gt.append(is_mal)
                        per_fold_pred.append(int(pred_is_mal))
                        per_fold_total_chunks += chunks
                        per_fold_total_off_path_chunks += off_path_chunks
                    precision, recall, f1, fpr, fnr = prf1(per_fold_gt, per_fold_pred)
                    per_fold_result[fold_name] = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'fpr': fpr,
                        'fnr': fnr,
                        'off_path_ratio': per_fold_total_off_path_chunks / per_fold_total_chunks
                    }
                precision = []
                recall = []
                f1 = []
                fpr = []
                fnr = []
                offpath_ratio = []
                for fold_name in scores.keys() :
                    precision.append(per_fold_result[fold_name]['precision'])
                    recall.append(per_fold_result[fold_name]['recall'])
                    f1.append(per_fold_result[fold_name]['f1'])
                    fpr.append(per_fold_result[fold_name]['fpr'])
                    fnr.append(per_fold_result[fold_name]['fnr'])
                    offpath_ratio.append(per_fold_result[fold_name]['off_path_ratio'])
                if np.mean(fpr) <= fpr_target and np.mean(offpath_ratio) <= off_path_limit :
                    result = {
                        'precision': {'m': np.mean(precision), 's': np.std(precision)},
                        'recall': {'m': np.mean(recall), 's': np.std(recall)},
                        'f1': {'m': np.mean(f1), 's': np.std(f1)},
                        'fpr': {'m': np.mean(fpr), 's': np.std(fpr)},
                        'fnr': {'m': np.mean(fnr), 's': np.std(fnr)},
                    }
                    #all_result[(dns_thres_on_t, dns_thres_off_t, dpi_thres_t)] = result
                    fnr = np.mean(fnr)
                    if np.abs(fnr - best_fnr) < 1e-8 :
                        threshold_candidates.append((dns_thres_on_t, dns_thres_off_t, dpi_thres_t))
                    elif fnr < best_fnr :
                        best_fnr = fnr
                        best_result = result
                        threshold_candidates = [(dns_thres_on, dns_thres_off, dpi_thres)]
    if threshold_candidates :
        # sort threshold_candidates first by dns_thres_on_t, then dpi_thres, then dns_thres_off_t in descending order
        threshold_candidates = sorted(threshold_candidates, key = lambda x : (x[0], x[2], x[1]), reverse = True)
        best_thres = threshold_candidates[0]
        print(f'for {hours} hours run and target FPR of {fpr_target}, best threshold is found at {best_thres}')
        print('best result')
        print('precision', best_result['precision']['m'], best_result['precision']['s'])
        print('recall', best_result['recall']['m'], best_result['recall']['s'])
        print('f1', best_result['f1']['m'], best_result['f1']['s'])
        print('fpr', best_result['fpr']['m'], best_result['fpr']['s'])
        print('fnr', best_result['fnr']['m'], best_result['fnr']['s'])
        return best_thres, best_result
    else :
        print('no threshold found')
        return None, None

