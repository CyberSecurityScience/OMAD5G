
from collections import Counter
import copy
from functools import partial
import glob
import math
import os
import pickle
import secrets
import sys
import time
from typing import Dict, List, Optional, Tuple
import einops
import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score
import numpy as np

from dataclasses import dataclass

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import preprocess_flow_stats
from utils import *
import os

@dataclass
class SinglePcap :
    pcap: str
    duration: float
    malicious: bool

@dataclass
class MixingSolution :
    sample_id: str
    pcaps: List[SinglePcap]
    duration: float
    malicious: bool
    malicious_segments: List[int]
    used_segments: List[int] = None

    def create(self, save_path) :
        # step 1: merge pcaps
        # run ./merge_pcap_linear_cic /mnt/tmpfs/<sample_id>.pcap <pcap1> <pcap2> ...
        args = ['./merge_pcap_linear_cic', f'/mnt/tmpfs/{self.sample_id}.pcap']
        args[1] = f'\"{args[1]}\"'
        for pcap in self.pcaps :
            args.append(f'\"{pcap.pcap}\"')
        os.system(' '.join(args))
        # step 2: split into 30 minutes segments
        # run ./split_pcap_linear_cic /mnt/tmpfs/split-<sample_id> /mnt/tmpfs/<sample_id>.pcap
        args = ['./split_pcap_linear_cic', f'/mnt/tmpfs/split-{self.sample_id}', f'/mnt/tmpfs/{self.sample_id}.pcap']
        os.system(' '.join(args))
        # step 3: save to csv
        n_useful_segments = 0
        for segment_id in self.used_segments :
            label = 'mal' if self.malicious else 'ben'
            filename_csv = os.path.join(save_path, f'{label}-{self.sample_id}-{segment_id}.csv')
            try :
                #filename_dns = os.path.join(save_path, f'{label}-{self.sample_id}-{segment_id}.txt')
                # run ./traffic_collector --mode pcap --pcap-filename /mnt/tmpfs/split-{self.sample_id}-{seg_id}.pcap --out-csv-filename {filename_csv} --ue-ip-range 10.42.0.0/16
                args = ['./traffic_collector', '--mode', 'pcap', '--pcap-filename', f'/mnt/tmpfs/split-{self.sample_id}-{segment_id}.pcap', '--out-csv-filename', filename_csv, '--ue-ip-range', '10.42.0.0/16']
                os.system(' '.join(args))
                with open(filename_csv + '.dns.txt', 'r') as f :
                    n_lines = len(f.readlines())
                # read csv
                df = pd.read_csv(filename_csv)
                if df.shape[0] >= 100 and n_lines >= 1 :
                    n_useful_segments += 1
                else :
                    os.system(f'rm {filename_csv}')
                    os.system(f'rm {filename_csv}.dns.txt')
            except Exception as e :
                print(e)
                os.system(f'rm {filename_csv}')
                os.system(f'rm {filename_csv}.dns.txt')
        # step 4: delete pcaps
        os.system(f'rm /mnt/tmpfs/{self.sample_id}.pcap')
        os.system(f'rm /mnt/tmpfs/split-{self.sample_id}*')
        return n_useful_segments
    
    def create_test(self, save_path) :
        # step 1: merge pcaps
        # run ./merge_pcap_linear_cic /mnt/tmpfs/<sample_id>.pcap <pcap1> <pcap2> ...
        args = ['./merge_pcap_linear_cic', f'/mnt/tmpfs/{self.sample_id}.pcap']
        args[1] = f'\"{args[1]}\"'
        for pcap in self.pcaps :
            args.append(f'\"{pcap.pcap}\"')
        os.system(' '.join(args))
        # step 2: split into 30 minutes segments
        # run ./split_pcap_linear_cic /mnt/tmpfs/split-<sample_id> /mnt/tmpfs/<sample_id>.pcap
        args = ['./split_pcap_linear_cic', f'/mnt/tmpfs/split-{self.sample_id}', f'/mnt/tmpfs/{self.sample_id}.pcap']
        os.system(' '.join(args))
        # step 3: save to csv
        n_useful_segments = 0
        for segment_id in self.used_segments :
            label = 'mal' if self.malicious else 'ben'
            filename_csv = os.path.join(save_path, f'{label}-{self.sample_id}-{segment_id}.csv')
            try :
                #filename_dns = os.path.join(save_path, f'{label}-{self.sample_id}-{segment_id}.txt')
                # run ./traffic_collector --mode pcap --pcap-filename /mnt/tmpfs/split-{self.sample_id}-{seg_id}.pcap --out-csv-filename {filename_csv} --ue-ip-range 10.42.0.0/16
                args = ['./traffic_collector', '--mode', 'pcap', '--pcap-filename', f'/mnt/tmpfs/split-{self.sample_id}-{segment_id}.pcap', '--out-csv-filename', filename_csv, '--ue-ip-range', '10.42.0.0/16']
                os.system(' '.join(args))
                if not os.path.exists(filename_csv) :
                    continue
                os.system(' '.join(args))
                with open(filename_csv + '.dns.txt', 'r') as f :
                    n_lines = len(f.readlines())
                # read csv
                df = pd.read_csv(filename_csv)
                if df.shape[0] >= 100 and n_lines >= 1 :
                    n_useful_segments += 1
            except Exception as e :
                print(e)
                os.system(f'rm {filename_csv}')
                os.system(f'rm {filename_csv}.dns.txt')
        # step 4: delete pcaps
        os.system(f'rm /mnt/tmpfs/{self.sample_id}.pcap')
        os.system(f'rm /mnt/tmpfs/split-{self.sample_id}*')
        if n_useful_segments != len(self.used_segments) :
            for segment_id in self.used_segments :
                label = 'mal' if self.malicious else 'ben'
                filename_csv = os.path.join(save_path, f'{label}-{self.sample_id}-{segment_id}.csv')
                os.system(f'rm {filename_csv}')
                os.system(f'rm {filename_csv}.dns.txt')
            return 0
        return 1

    def delete(self, save_path) :
        os.system(f'rm /mnt/tmpfs/{self.sample_id}.pcap')
        os.system(f'rm /mnt/tmpfs/split-{self.sample_id}*')
        for segment_id in self.used_segments :
            label = 'mal' if self.malicious else 'ben'
            filename_csv = os.path.join(save_path, f'{label}-{self.sample_id}-{segment_id}.csv')
            os.system(f'rm {filename_csv}')
            os.system(f'rm {filename_csv}.dns.txt')

def overlap_range(start_a, end_a, start_b, end_b):
    # Calculate the start and end of the overlap
    overlap_start = max(start_a, start_b)
    overlap_end = min(end_a, end_b)

    # Check if there is an overlap and calculate its length
    if overlap_start <= overlap_end:
        return overlap_end - overlap_start  # Return the length of the overlap
    else:
        return 0  # No overlap
    
class TrainTestSplit :
    def __init__(self, benign_train: List[SinglePcap], benign_test: List[SinglePcap], malicious_train: List[SinglePcap], malicious_test: List[SinglePcap]) :
        self.benign_train = benign_train
        self.benign_test = benign_test
        self.malicious_train = malicious_train
        self.malicious_test = malicious_test

    def to_fold(self, fold_id, n_folds) :
        n_per_fold_ben = len(self.benign_train) // n_folds
        n_per_fold_mal = len(self.malicious_train) // n_folds
        test_ben = self.benign_train[fold_id * n_per_fold_ben : (fold_id + 1) * n_per_fold_ben]
        test_mal = self.malicious_train[fold_id * n_per_fold_mal : (fold_id + 1) * n_per_fold_mal]
        train_ben = self.benign_train[:fold_id * n_per_fold_ben] + self.benign_train[(fold_id + 1) * n_per_fold_ben :]
        train_mal = self.malicious_train[:fold_id * n_per_fold_mal] + self.malicious_train[(fold_id + 1) * n_per_fold_mal :]
        return TrainTestSplit(train_ben, test_ben, train_mal, test_mal)

class PcapDataset :
    def __init__(self, txt_path: str, seed: int, benign_key = '-be-') :
        with open(txt_path, 'r') as f :
            lines = f.readlines()
        self.pcaps: List[SinglePcap] = []
        for line in lines :
            filename, duration = line.strip().split('\t')
            duration = float(duration)
            if duration < 1 * 60 or duration > 70 * 60 :
                #print(f'skipping {filename} due to duration {duration}')
                continue
            self.pcaps.append(SinglePcap(filename, duration, benign_key not in filename))
        np.random.seed(seed)
        np.random.shuffle(self.pcaps)
        self.benign_samples = [p for p in self.pcaps if not p.malicious]
        self.malicious_samples = [p for p in self.pcaps if p.malicious]

    def to_fold(self, fold_id, n_folds) -> TrainTestSplit :
        n_benign = len(self.benign_samples)
        n_malicious = len(self.malicious_samples)
        n_per_fold_ben = n_benign // n_folds
        n_per_fold_mal = n_malicious // n_folds
        test_ben = self.benign_samples[fold_id * n_per_fold_ben : (fold_id + 1) * n_per_fold_ben]
        test_mal = self.malicious_samples[fold_id * n_per_fold_mal : (fold_id + 1) * n_per_fold_mal]
        train_ben = self.benign_samples[:fold_id * n_per_fold_ben] + self.benign_samples[(fold_id + 1) * n_per_fold_ben :]
        train_mal = self.malicious_samples[:fold_id * n_per_fold_mal] + self.malicious_samples[(fold_id + 1) * n_per_fold_mal :]
        return TrainTestSplit(train_ben, test_ben, train_mal, test_mal)

def create_single_mix(benign_samples: List[SinglePcap], malicious_samples: List[SinglePcap], n_hours: int, overlap_ratio: float, n_ben: int, n_mal: int) -> MixingSolution :
    cur_length = 0
    used_pcaps = []
    mal_ctr = n_mal
    ben_ctr = n_ben
    total_length = n_hours * 3600 + 10 * 60
    while cur_length < total_length :
        is_pcap_mal = False
        if mal_ctr > 0 :
            mal_ctr -= 1
            source = malicious_samples
            is_pcap_mal = True
        elif ben_ctr > 0 :
            ben_ctr -= 1
            source = benign_samples
        else :
            # reset counters
            mal_ctr = n_mal
            ben_ctr = n_ben
            continue
        pcap = secrets.choice(source)
        cur_length += pcap.duration
        used_pcaps.append((is_pcap_mal, pcap))
    np.random.seed(int.from_bytes(secrets.token_bytes(4), 'big'))
    np.random.shuffle(used_pcaps)
    used_pcaps_with_offset = []
    offset = 0
    for is_pcap_mal, pcap in used_pcaps :
        used_pcaps_with_offset.append((is_pcap_mal, pcap, offset))
        offset += pcap.duration + 0.1 # 100ms offset
    malicious_segments = []
    # each segment is 30 minutes
    seg_length = 1800
    for segment_id in range(n_hours * 2) :
        total_overlap_length = 0
        is_seg_mal = False
        for is_pcap_mal, pcap, offset in used_pcaps_with_offset :
            if not is_pcap_mal :
                continue
            start = segment_id * seg_length
            end = (segment_id + 1) * seg_length
            overlap = overlap_range(start, end, offset, offset + pcap.duration)
            if overlap >= pcap.duration * overlap_ratio :
                is_seg_mal = True
            total_overlap_length += overlap
        if total_overlap_length >= seg_length * overlap_ratio :
            is_seg_mal = True
        if is_seg_mal :
            malicious_segments.append(segment_id)
    return MixingSolution(secrets.token_hex(14), [p for _, p, _ in used_pcaps_with_offset], cur_length, len(malicious_segments) > 0, malicious_segments)

def create_training_mix_benign_train(path: str, n_segments: int, benign_samples: List[SinglePcap], malicious_samples: List[SinglePcap], n_hours: int, overlap_ratio: float, n_ben: int, n_mal: int) -> List[MixingSolution] :
    cur_segments = 0
    selected_mixes = []
    while cur_segments < n_segments :
        mix = create_single_mix(benign_samples, malicious_samples, n_hours, overlap_ratio, n_ben, n_mal)
        mix.used_segments = []
        for i in range(n_hours * 2) :
            mix.used_segments.append(i)
        try :
            cur_segments += mix.create(path)
        except Exception as e :
            print(e)
            mix.delete(path)
            continue
        selected_mixes.append(mix)
    return selected_mixes

def create_training_mix_malicious_train(path: str, n_segments: int, benign_samples: List[SinglePcap], malicious_samples: List[SinglePcap], n_hours: int, overlap_ratio: float, n_ben: int, n_mal: int) -> List[MixingSolution] :
    cur_segments = 0
    selected_mixes = []
    while cur_segments < n_segments :
        mix = create_single_mix(benign_samples, malicious_samples, n_hours, overlap_ratio, n_ben, n_mal)
        mix.used_segments = []
        for mal_seg in mix.malicious_segments :
            mix.used_segments.append(mal_seg)
        try :
            cur_segments += mix.create(path)
        except Exception as e :
            print(e)
            mix.delete(path)
            continue
        selected_mixes.append(mix)
    return selected_mixes

def create_test_mix_bengin(path: str, n_samples: int, benign_samples: List[SinglePcap], malicious_samples: List[SinglePcap], n_hours: int, overlap_ratio: float, n_ben: int, n_mal: int) -> List[MixingSolution] :
    selected_mixes = []
    cur_samples = 0
    while cur_samples < n_samples :
        mix = create_single_mix(benign_samples, malicious_samples, n_hours, overlap_ratio, n_ben, n_mal)
        mix.used_segments = [i for i in range(n_hours * 2)]
        try :
            cur_samples += mix.create_test(path)
        except Exception as e :
            print(e)
            mix.delete(path)
            continue
        selected_mixes.append(mix)
    return selected_mixes

def create_test_mix_malicious(path: str, n_samples: int, benign_samples: List[SinglePcap], malicious_samples: List[SinglePcap], n_hours: int, overlap_ratio: float, n_ben: int, n_mal: int) -> List[MixingSolution] :
    selected_mixes = []
    cur_samples = 0
    while cur_samples < n_samples :
        mix = create_single_mix(benign_samples, malicious_samples, n_hours, overlap_ratio, n_ben, n_mal)
        mix.used_segments = [i for i in range(n_hours * 2)]
        try :
            cur_samples += mix.create_test(path)
        except Exception as e :
            print(e)
            mix.delete(path)
            continue
        selected_mixes.append(mix)
    return selected_mixes

def create_normal_solution(train_folder: str) -> dict :
    #return {}
    # list all csv files
    csv_files = glob.glob(os.path.join(train_folder, '*.csv'))
    # randomly sample 100 files
    csv_files = np.random.choice(csv_files, 100, replace=False)
    all_dfs = []
    for csv_file in csv_files :
        df = pd.read_csv(csv_file)
        all_dfs.append(df)
    doc = pd.concat(all_dfs)
    doc = doc.drop(['Flow ID', 'Src IP','Src Port','Dst IP','Dst Port','Timestamp', 'DNS Resp'], axis=1)
    doc = doc.fillna(0)
    cols = [i for i in doc.columns if i not in ["Content", "Host", 'DNS Query']]
    for col in cols:
        doc[col] = pd.to_numeric(doc[col])
    doc = doc.dropna()
    normalize_solution = {}
    for column in doc.columns :
        if column not in ['pcap_file', 'Content', 'Start TS', 'End TS', 'Host', 'DNS Query'] :
            normalize_solution[column] = preprocess_flow_stats.create_solution(doc, column)
    return normalize_solution

def worker_impl(seed, rank = 0, world_size = 1) :
    ds = PcapDataset('cic_with_length.txt', seed)
    name = f'cic-2'
    n_train_half = 5100
    assert n_train_half % world_size == 0
    n_train_half_worker = n_train_half // world_size
    n_test_ben = 1080
    n_test_mal = 120
    n_hours = 4
    n_ben = 3
    n_mal = 1
    overlap_ratio = 0.7
    save_dir = f'/mnt/nvme0/malware_csv/{name}'
    os.makedirs(save_dir, exist_ok=True)
    n_folds = 5
    for fold_id in range(n_folds) :
        outter = ds.to_fold(fold_id, n_folds)
        outter_path = os.path.join(save_dir, f'fold-{fold_id}')
        outter_path_train = os.path.join(outter_path, 'train')
        outter_path_test = os.path.join(outter_path, 'test')
        os.makedirs(outter_path_train, exist_ok=True)
        os.makedirs(outter_path_test, exist_ok=True)
        if rank == 0 :
            print(f'fold {fold_id}')
            print('test benign:', len(outter.benign_test))
            print('test malicious:', len(outter.malicious_test))
        outter_train_benign_mixes = create_training_mix_benign_train(outter_path_train, n_train_half_worker, outter.benign_train, outter.malicious_train, n_hours, overlap_ratio, 1, 0)
        outter_train_malicious_mixes = create_training_mix_malicious_train(outter_path_train, n_train_half_worker, outter.benign_train, outter.malicious_train, n_hours, overlap_ratio, n_ben, n_mal)
        worker_n_test_ben = (n_test_ben - 1) // world_size + 1
        worker_n_test_ben -= 1 if rank >= (n_test_ben % world_size) and n_test_ben % world_size != 0 else 0
        outter_test_benign_mixes = create_test_mix_bengin(outter_path_test, worker_n_test_ben, outter.benign_test, outter.malicious_test, n_hours, overlap_ratio, 1, 0)
        worker_n_test_mal = (n_test_mal - 1) // world_size + 1
        worker_n_test_mal -= 1 if rank >= (n_test_mal % world_size) and n_test_mal % world_size != 0 else 0
        outter_test_malicious_mixes = create_test_mix_malicious(outter_path_test, worker_n_test_mal, outter.benign_test, outter.malicious_test, n_hours, overlap_ratio, n_ben, n_mal)
        # for mix in outter_train_benign_mixes + outter_train_malicious_mixes :
        #     mix.create(outter_path_train)
        # for mix in outter_test_benign_mixes + outter_test_malicious_mixes :
        #     mix.create(outter_path_test)
        if rank == 0 :
            # create normal solution
            normal_solution = create_normal_solution(outter_path_train)
            with open(os.path.join(outter_path, 'normalize_solution.pkl'), 'wb') as f :
                pickle.dump(normal_solution, f)
        for i in range(n_folds) :
            inner = outter.to_fold(i, n_folds)
            inner_path = os.path.join(outter_path, f'fold-{i}')
            inner_path_train = os.path.join(inner_path, 'train')
            inner_path_test = os.path.join(inner_path, 'test')
            os.makedirs(inner_path_train, exist_ok=True)
            os.makedirs(inner_path_test, exist_ok=True)
            if rank == 0 :
                print(f'  inner fold {i}')
                print('  train benign:', len(inner.benign_train))
                print('  test benign:', len(inner.benign_test))
                print('  train malicious:', len(inner.malicious_train))
                print('  test malicious:', len(inner.malicious_test))
            inner_train_benign_mixes = create_training_mix_benign_train(inner_path_train, n_train_half_worker, inner.benign_train, inner.malicious_train, n_hours, overlap_ratio, 1, 0)
            inner_train_malicious_mixes = create_training_mix_malicious_train(inner_path_train, n_train_half_worker, inner.benign_train, inner.malicious_train, n_hours, overlap_ratio, n_ben, n_mal)
            worker_n_test_ben = (n_test_ben - 1) // world_size + 1
            worker_n_test_ben -= 1 if rank >= (n_test_ben % world_size) and n_test_ben % world_size != 0 else 0
            inner_test_benign_mixes = create_test_mix_bengin(inner_path_test, worker_n_test_ben, inner.benign_test, inner.malicious_test, n_hours, overlap_ratio, 1, 0)
            worker_n_test_mal = (n_test_mal - 1) // world_size + 1
            worker_n_test_mal -= 1 if rank >= (n_test_mal % world_size) and n_test_mal % world_size != 0 else 0
            inner_test_malicious_mixes = create_test_mix_malicious(inner_path_test, worker_n_test_mal, inner.benign_test, inner.malicious_test, n_hours, overlap_ratio, n_ben, n_mal)
            # for mix in inner_train_benign_mixes + inner_train_malicious_mixes :
            #     mix.create(inner_path_train)
            # for mix in inner_test_benign_mixes + inner_test_malicious_mixes :
            #     mix.create(inner_path_test)
            if rank == 0 :
                # create normal solution
                normal_solution = create_normal_solution(inner_path_train)
                with open(os.path.join(inner_path, 'normalize_solution.pkl'), 'wb') as f :
                    pickle.dump(normal_solution, f)
        
def worker(a, b) :
    try :
        worker_impl(a, b)
    except Exception :
        for i in range(100) :
            import traceback
            traceback.print_exc()
        os._exit(0)

def main() :
    seed = secrets.randbits(32)
    # worker(0, 1)
    import multiprocessing as mp
    procs = []
    n_workers = 50
    for i in range(n_workers) :
        p = mp.Process(target = worker_impl, args = (seed, i, n_workers, ))
        p.start()
        procs.append(p)
    for p in procs :
        p.join()

if __name__ == '__main__' :
    main()
