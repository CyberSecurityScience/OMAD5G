
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

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import preprocess_flow_stats
from utils import AvgMeter
from models import DPIModel, all_android_domains

class Args :
    dataset = 'cic-2'
    save_folder = 'runs-cic-nested-final-2'
    n_folds = 5

    dns_enabled = True
    dns_ben_w = 3
    dns_mal_w = 1
    dns_score_scale = 1
    dns_weight_decay = 0.1
    dns_label_smoothing = 0.05 # or 0 or 0.01
    dns_temperture = 0.5 # or 0.1 or 0.2
    dns_epochs = '500x400x450'
    dns_max_domains = 512
    dns_batch_size = 128
    dns_offset_prob = 1
    dns_offset_ratio = 0.5
    dns_mal_drop = 0.05
    dns_ben_drop = 0.05
    dns_increase_mal_domain_occurence = False
    dns_filter_out_android_domain = True
    dns_qbits = 16
    dns_use_raw_domains = False

    dpi_enabled = True
    dpi_enabled_fp16 = False
    dpi_norm_version = 'all' # one of 'none', 'nolog', 'all'
    dpi_posweight = 1
    dpi_temperture = 0.1 # or 0.5 or 0.2
    dpi_label_smoothing = 0.01 # or 0 or 0.05
    dpi_flow_dropout_ratio = 0.01
    dpi_max_flows = 3000
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


class DomainWeightedDataset(Dataset) :
    def __init__(self, source_folder: str, max_domains: int, train, args: Args, all_domains = None, random_dropout_prob = 0, random_offset_prob = 0.03) -> None :
        super().__init__()
        donmain_txts = glob.glob(os.path.join(source_folder, '*.txt'))
        self.samples = []
        all_requests = []
        for txt in donmain_txts :
            lbl, sample_id, chunk_idx = Path(txt.replace('.csv.dns.txt', '.txt')).stem.split('-')
            with open(txt, 'r', encoding = 'utf-8') as fp :
                domains = [x.strip() for x in fp.readlines()]
                domains = [s for s in domains if s]
                if not args.dns_use_raw_domains :
                    domains = [keep_tld_and_one_label(s) for s in domains]
                if args.dns_filter_out_android_domain :
                    domains = [s for s in domains if s not in all_android_domains]
                all_requests.extend(domains)
            self.samples.append((domains, lbl == 'mal', sample_id, int(chunk_idx)))
        self.malicious_domains = set()
        self.benign_domains = set()
        for (domains, is_mal, _, _) in self.samples :
            if is_mal :
                self.malicious_domains.update(domains)
            else :
                self.benign_domains.update(domains)
        if all_domains is None :
            self.all_domains = set()
            # for (domains, is_mal, _, _) in self.samples :
            #     self.all_domains.update(domains)
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

    # def sample_from_domains_and_label(self, domains, label) :
    #     ret_indices = torch.zeros(self.max_domains, dtype = torch.long)
    #     ret_counts = torch.zeros(self.max_domains, dtype = torch.float32)
    #     for i, (domain, count) in enumerate(Counter(domains).most_common()) :
    #         domain_index = self.domain2idx.get(domain, 1)
    #         if np.random.rand() < self.mal_dropout_prob and domain in self.malicious_domains :
    #             domain_index = 1 # set a domain to UNKNOWN
    #         if np.random.rand() < self.ben_dropout_prob and domain not in self.malicious_domains :
    #             domain_index = 1 # set a domain to UNKNOWN
    #         if np.random.rand() < self.random_offset_prob :
    #             count_tail = count * self.offset_ratio
    #             count = int(max(float(count) + np.random.normal(0, count_tail), 1))
    #         if i >= len(ret_indices) :
    #             break
    #         ret_indices[i] = domain_index
    #         ret_counts[i] = float(count)
    #     return ret_indices, ret_counts, int(label)
    
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
        domains, label, sample_id, chunk_idx = self.samples[idx]
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
                if True :
                    # no overflow observed during test, so commented out to make it faster
                    # if you observe overflow, you can always change the domain reputation score storage on p4 switch to int<32>, the switch has enough space to accommodate it
                    scores += score_for_current_domain * cnt
                else :
                    # a more accurate emulation of p4 switch behavior
                    for _ in range(cnt) :
                        scores += score_for_current_domain
                        if any(scores > self.quant.quant_range_ub) or any(scores < self.quant.quant_range_lb) :
                            print('warn', scores)
                        scores = np.clip(scores, self.quant.quant_range_lb, self.quant.quant_range_ub)
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
def train_dns(source_folder: str, args: Args) :
    dns_ds = DomainWeightedDataset(source_folder, max_domains = args.dns_max_domains, train = True, args = args, random_dropout_prob = 0.001)
    model = DomainWeightedClassifier(dns_ds.all_domains, n_score = 1, score_scale = args.dns_score_scale)
    model = model.cuda()
    opt = optim.AdamW(model.parameters(), 1e-3, (0.95, 0.999), weight_decay = args.dns_weight_decay)
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
    import gc
    gc.collect()
    torch.cuda.empty_cache()
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
    def __init__(self, source_folder: str, args: Args, category_columns = ['Feat 0', 'Feat 1'], flow_dropout_ratio = 0.5, normalize_solution = None, trainset = True) -> None:
        super().__init__()
        self.all_pcap_files = set()
        self.samples = glob.glob(os.path.join(source_folder, '*.csv'))
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
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', format='ISO8601')
        df = df.dropna(subset = ['Timestamp'])  # Drop rows with NaT values
        df['Start TS'] = (df['Timestamp'] - df['Timestamp'].min()).dt.total_seconds()
        df['End TS'] = df['Start TS'] + df['Feat 2']
        n = df.shape[0]
        drop_indices = np.random.choice(df.index, int(n * self.flow_dropout_ratio), replace = False)
        if df.shape[0] - len(drop_indices) > 100 :
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

        # Flow ID,Host,Src IP,Src Port,Dst IP,Dst Port,Timestamp,Content,DNS Query,DNS Resp
        df = df.drop(['Content', 'Start TS', 'End TS', 'Host', 'id', 'Timestamp', 'DNS Query', 'Flow ID', 'DNS Resp', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port'], axis = 1)
        if self.normalize_solution is not None :
            for col in df.columns :
                df[col] = self.normalize_solution[col].apply(df[col])
        cat_cols = np.stack([df[col].to_numpy() for col in self.category_columns], axis = -1)
        label = int(is_malicious)
        df_clean = df.drop(self.category_columns, axis = 1)
        df_clean = df_clean.replace([np.inf, -np.inf], 0)
        df_clean = df_clean.to_numpy()
        host_ids = host_ids.to_numpy()
        return df_clean, cat_cols, col_start_ts, col_end_ts, dpi_bytes, host_ids, label, n, n_bytes

    def __getitem__(self, i) :
        #return 1, 2, 3, 4, 5, 6, 7, 8, 9
        sample_filename = self.samples[i]
        lbl, sample_id, chunk_idx = Path(sample_filename).stem.split('-')
        is_malicious = lbl == 'mal'
        return self.sample_from_csv_label(sample_filename, is_malicious)
        
def collate_fn(data) :
    #return 1, 2, 3, 4,5 ,6,7,1,[9],10
    # mask_out_ratio = 0.5
    t0 = time.perf_counter()
    k = secrets.token_hex(4)
    #print(f'[{k}] [{time.perf_counter()}] load data')
    x, xc, col_start_ts, col_end_ts, dpi_bytes, host_ids, y, n_flows, n_bytes = zip(*data)
    N = len(x)
    max_flows = max([f.shape[0] for f in x])
    n_num_feat = x[0].shape[1]
    n_cat_feat = xc[0].shape[1]
    num_dpi_bytes = dpi_bytes[0].shape[1]
    #print(f'[{k}] create buffer', N, max_flows)
    x_num = torch.zeros(N, max_flows, n_num_feat, dtype = torch.float64)
    x_cat = torch.zeros(N, max_flows, n_cat_feat, dtype = torch.int64)
    x_start_ts = torch.zeros(N, max_flows, dtype = torch.int64)
    x_end_ts = torch.zeros(N, max_flows, dtype = torch.int64)
    x_dpi_bytes = torch.zeros(N, max_flows, num_dpi_bytes, dtype = torch.uint8)
    x_host_ids = torch.zeros(N, max_flows, dtype = torch.int64)
    mask = torch.ones(N, max_flows, dtype = torch.bool)
    #print(f'[{k}] fill buffer')
    for i in range(N) :
        n_flow = x[i].shape[0]
        x_num[i, : n_flow, :] = torch.tensor(x[i])
        x_cat[i, : n_flow, :] = torch.tensor(xc[i])
        x_start_ts[i, : n_flow] = torch.tensor(col_start_ts[i])
        x_end_ts[i, : n_flow] = torch.tensor(col_end_ts[i])
        x_dpi_bytes[i, : n_flow, :] = torch.tensor(dpi_bytes[i])
        x_host_ids[i, : n_flow] = torch.tensor(host_ids[i])
        mask[i, : n_flow] = False
        # perm = list(range(n_flow))
        # np.random.shuffle(perm)
        # mask[i, perm[: int(n_flow * mask_out_ratio)]] = True
    #print(f'[{k}] [{time.perf_counter()}] return', time.perf_counter() - t0)
    return x_num.float(), x_cat, x_start_ts, x_end_ts, x_dpi_bytes, x_host_ids, mask, torch.tensor(y, dtype = torch.int64), list(n_flows), list(n_bytes)


def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count()))

def train_dpi(source_folder: str, normalize_solution, args: Args) :
    ds = FlowDatasetTorch(source_folder, args, flow_dropout_ratio = args.dpi_flow_dropout_ratio, normalize_solution = normalize_solution)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size = 24,
        num_workers = 20,
        drop_last = True,
        shuffle = True,
        collate_fn = collate_fn,
        timeout = 50,
        pin_memory = True,
        worker_init_fn = worker_init_fn,
    )
    model = DPIModel(65, num_category_inputs = 2, num_cat_per_category_input = 16, dpi_bytes = args.dpi_n_dpi_bytes)
    model = model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr = 3e-4, betas = (0.99, 0.999), weight_decay = 0.1)
    [n_epochs, s1, s2] = args.dpi_epochs.split('x')
    [n_epochs, s1, s2] = [int(x) for x in [n_epochs, s1, s2]]
    lrs = torch.optim.lr_scheduler.MultiStepLR(opt, [s1, s2], gamma = 0.1)
    loss_avg = AvgMeter()
    w = torch.ones((2, ), dtype = torch.float32).cuda()
    w[0] = args.dpi_posweight
    loss_fn = nn.CrossEntropyLoss(weight = w, label_smoothing = args.dpi_label_smoothing).cuda()
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    scaler = torch.amp.GradScaler(enabled = False)
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
            with torch.autocast('cuda', enabled = False) :
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

def get_class_values(cls):
    r = ''
    for attr in dir(cls):
        if not attr.startswith("__") and not callable(getattr(cls, attr)):
            r += f'{attr} = {getattr(cls, attr)}' + '\n'
    return r

def get_class_values_dict(cls):
    ret = {}
    for attr in dir(cls):
        if not attr.startswith("__") and not callable(getattr(cls, attr)):
            ret[attr] = getattr(cls, attr)
    return ret

def main() :
    malware_csv_folder = '/mnt/nvme0/malware_csv'
    args = Args()
    run_id = secrets.token_hex(16)
    run_dir = os.path.join(args.save_folder, run_id)
    os.makedirs(run_dir)
    ds = args.dataset
    with open(os.path.join(run_dir, 'args.txt'), 'w') as fp :
        fp.write(get_class_values(Args))
    with open(os.path.join(run_dir, 'args.pkl'), 'wb') as fp :
        pickle.dump(get_class_values_dict(args), fp)
    out_filename = os.path.join(run_dir, 'logs.txt')
    print(out_filename)
    sys.stdout = open(out_filename, 'w')
    sys.stderr = sys.stdout
    ds_folder = os.path.join(malware_csv_folder, ds)
    for outter_fold in range(args.n_folds) :
        print(f'Outter fold {outter_fold}')
        outter_fold_path = os.path.join(ds_folder, f'fold-{outter_fold}')
        source = os.path.join(outter_fold_path, 'train')
        save_dir = os.path.join(run_dir, f'fold-{outter_fold}')
        os.makedirs(save_dir)
        normalize_solution_file = os.path.join(outter_fold_path, f'normalize_solution_{args.dpi_norm_version}.pkl')
        if not os.path.exists(normalize_solution_file) :
            normalize_solution_file = os.path.join(outter_fold_path, 'normalize_solution.pkl')
        with open(normalize_solution_file, 'rb') as fp :
            normalize_solution = pickle.load(fp)
        with open(os.path.join(save_dir, 'normalize_solution.pkl'), 'wb') as fp :
            pickle.dump(normalize_solution, fp)
        with open(os.path.join(save_dir, 'args.txt'), 'w') as fp :
            fp.write(get_class_values(Args))
        with open(os.path.join(save_dir, 'args.pkl'), 'wb') as fp :
            pickle.dump(get_class_values_dict(args), fp)
        if args.dns_enabled :
            model_dns = train_dns(source, args)
            dns_savefile = os.path.join(save_dir, 'dns.pth')
            dns_vis_file = os.path.join(save_dir, 'dns_vis.txt')
            torch.save({'sd': model_dns.state_dict(), 'domains': model_dns.all_domains}, dns_savefile)
            with open(dns_vis_file, 'w') as fp :
                model_dns.embd = model_dns.embd.cpu()
                for idx, domain in model_dns.idx2domain.items() :
                    score_np = model_dns.embd(torch.tensor([idx])).detach().cpu().numpy().flatten()
                    score_quant = model_dns.quant.quantize_np(score_np)
                    fp.write(f'{idx} {domain} {score_np[0]} {score_quant[0]}\n')
        if args.dpi_enabled :
            model_dpi = train_dpi(source, normalize_solution, args)
            dpi_savefile = os.path.join(save_dir, 'dpi.pth')
            torch.save(model_dpi.state_dict(), dpi_savefile)
        for inner_fold in range(args.n_folds) :
            print(f'Outter fold {outter_fold}, inner fold {inner_fold}')
            inner_fold_path = os.path.join(outter_fold_path, f'fold-{inner_fold}')
            source = os.path.join(inner_fold_path, 'train')
            save_dir = os.path.join(run_dir, f'fold-{outter_fold}', f'fold-{inner_fold}')
            os.makedirs(save_dir)
            normalize_solution_file = os.path.join(inner_fold_path, f'normalize_solution_{args.dpi_norm_version}.pkl')
            if not os.path.exists(normalize_solution_file) :
                normalize_solution_file = os.path.join(inner_fold_path, f'normalize_solution.pkl')
            with open(normalize_solution_file, 'rb') as fp :
                normalize_solution = pickle.load(fp)
            with open(os.path.join(save_dir, 'normalize_solution.pkl'), 'wb') as fp :
                pickle.dump(normalize_solution, fp)
            if args.dns_enabled :
                model_dns = train_dns(source, args)
                dns_savefile = os.path.join(save_dir, 'dns.pth')
                dns_vis_file = os.path.join(save_dir, 'dns_vis.txt')
                torch.save({'sd': model_dns.state_dict(), 'domains': model_dns.all_domains}, dns_savefile)
                with open(dns_vis_file, 'w') as fp :
                    model_dns.embd = model_dns.embd.cpu()
                    for idx, domain in model_dns.idx2domain.items() :
                        score_np = model_dns.embd(torch.tensor([idx])).detach().cpu().numpy().flatten()
                        score_quant = model_dns.quant.quantize_np(score_np)
                        fp.write(f'{idx} {domain} {score_np[0]} {score_quant[0]}\n')
            if args.dpi_enabled :
                model_dpi = train_dpi(source, normalize_solution, args)
                dpi_savefile = os.path.join(save_dir, 'dpi.pth')
                torch.save(model_dpi.state_dict(), dpi_savefile)

if __name__ == '__main__' :
    main()
