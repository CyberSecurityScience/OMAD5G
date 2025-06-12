
import json
import struct
import time
import fastapi
from trainer import Args
import uvicorn
from fastapi import FastAPI, Request, Response, status, Form, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel
from aiohttp import ClientSession

from preprocess_flow_stats import NormalizeSolution
import pandas as pd
import pickle
import torch
import numpy as np
import os
from natsort import natsorted
from models import DPIModel, all_android_domains

from contextlib import asynccontextmanager

TMP_DIR = 'domainwatcher_tmp'
app = fastapi.FastAPI()

def create_dpi_model(args: Args) :
    model = DPIModel(65, num_category_inputs = 2, num_cat_per_category_input = 16, dpi_bytes = args.dpi_n_dpi_bytes)
    model = model.cuda()
    model.eval()
    return model

MODEL_DPI = None
NORAMLIZE_SOLUTION = None
UE_IP_RANGE = '10.42.0.0/16' # for CIC
ALL_INFER_TIME = []
ALL_FEAT_TIME = []
ALL_PREPROCESS_TIME = []

class UeRequest(BaseModel):
    ue_ip: str

@app.post('/add_monitored_ue') # given an UE IP, add it to the list of monitored UEs
async def add_monitored_ue(req: UeRequest):
    # call POST 127.0.0.1:5185/add_ue with ue_ip
    async with ClientSession() as session:
        async with session.post('http://127.0.0.1:5185/add_ue', json={'ue_ip': req.ue_ip}) as response:
            return response.status

TRAFFIC_COLLECTOR_PROCESS = None

def extract_normal_solution(dst_filename):
    global NORAMLIZE_SOLUTION
    keys = list(natsorted((NORAMLIZE_SOLUTION.keys())))
    ns = []
    for k in keys :
        ns.append(NORAMLIZE_SOLUTION[k].to_json())
    with open(dst_filename, 'w') as f :
        json.dump(ns, f, indent = 4)

@app.post("/upload_model") # upload a preprocessing pkl file and a checkpoint pth file
async def upload_model(
    request: Request,
    preprocessing_pkl: UploadFile = File(None),
    checkpoint_pth: UploadFile = File(None),
):
    # save to TMP_DIR
    if preprocessing_pkl:
        with open(f'{TMP_DIR}/normalize_solution.pkl', 'wb') as f:
            f.write(preprocessing_pkl.file.read())
    global NORAMLIZE_SOLUTION
    NORAMLIZE_SOLUTION = pickle.load(open(f'{TMP_DIR}/normalize_solution.pkl', 'rb'))
    if checkpoint_pth:
        with open(f'{TMP_DIR}/model.pth', 'wb') as f:
            f.write(checkpoint_pth.file.read())
    global MODEL_DPI
    MODEL_DPI = create_dpi_model(Args())
    MODEL_DPI.load_state_dict(torch.load(f'{TMP_DIR}/model.pth', map_location='cpu'))
    print('Extracting normal solution...')
    extract_normal_solution(f'{TMP_DIR}/normalize_solution.json')
    print('Starting traffic collector...')
    import subprocess
    global TRAFFIC_COLLECTOR_PROCESS
    TRAFFIC_COLLECTOR_PROCESS = subprocess.Popen(['traffic_collector', '--mode', 'if', '--if-name', 'ens193f0np0', '--normal-solution', f'{TMP_DIR}/normalize_solution.json', '--ue-ip-range', UE_IP_RANGE])
    return {"status": "success"}

dpi_max_flows = 3000
flow_dropout_ratio = 0.0
dpi_n_dpi_bytes = 160
category_columns = ['Feat 0', 'Feat 1']

def collate_fn(data) :
    x, xc, col_start_ts, col_end_ts, dpi_bytes, y, n_flows, n_bytes = zip(*data)
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
    mask = torch.ones(N, max_flows, dtype = torch.bool)
    for i in range(N) :
        n_flow = x[i].shape[0]
        x_num[i, : n_flow, :] = torch.tensor(x[i])
        x_cat[i, : n_flow, :] = torch.tensor(xc[i])
        x_start_ts[i, : n_flow] = torch.tensor(col_start_ts[i])
        x_end_ts[i, : n_flow] = torch.tensor(col_end_ts[i])
        x_dpi_bytes[i, : n_flow, :] = torch.tensor(dpi_bytes[i])
        mask[i, : n_flow] = False
    return x_num.float(), x_cat, x_start_ts, x_end_ts, x_dpi_bytes, mask, torch.tensor(y, dtype = torch.int64), list(n_flows), list(n_bytes)


def preprocess(df) :
    # preprocess
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', format='ISO8601')
    df = df.dropna(subset = ['Timestamp'])  # Drop rows with NaT values
    df['Start TS'] = (df['Timestamp'] - df['Timestamp'].min()).dt.total_seconds()
    df['End TS'] = df['Start TS'] + df['Feat 2']
    print('df[\'Feat 2\']', df['Feat 2'])
    df = df.drop(['Flow ID', 'Src IP','Src Port','Dst IP','Dst Port','Timestamp', 'DNS Resp'], axis=1)
    df = df.fillna(0)
    n_bytes = 0
    n = df.shape[0]
    if df.shape[0] > dpi_max_flows :
        drop_indices = np.random.choice(df.index, df.shape[0] - dpi_max_flows, replace = False)
        df = df.drop(drop_indices)
    n_bytes += int(df['Feat 3'].sum())
    dpi_bytes = [bytearray.fromhex(h)[: dpi_n_dpi_bytes] for h in df['Content']]
    dpi_bytes = [h + b'\00' * (dpi_n_dpi_bytes - len(h)) for h in dpi_bytes]
    dpi_bytes = [np.array(h, dtype = np.uint8) for h in dpi_bytes]
    dpi_bytes = np.stack(dpi_bytes, axis = 0)
    col_start_ts = np.array(df['Start TS']).astype(np.int64)
    print('np.array(df[\'Start TS\'])', np.array(df['Start TS']))
    col_end_ts = np.array(df['End TS']).astype(np.int64)
    print('np.array(df[\'End TS\'])', np.array(df['End TS']))

    df = df.drop(['Content', 'Start TS', 'End TS', 'Host', 'DNS Query'], axis = 1)
    if NORAMLIZE_SOLUTION is not None :
        for col in df.columns :
            df[col] = NORAMLIZE_SOLUTION[col].apply(df[col])
    cat_cols = np.stack([df[col].to_numpy() for col in category_columns], axis = -1)
    label = 1
    df_clean = df.drop(category_columns, axis = 1)
    df_clean = df_clean.replace([np.inf, -np.inf], 0)
    df_clean = df_clean.to_numpy()
    return df_clean, cat_cols, col_start_ts, col_end_ts, dpi_bytes, label, n, n_bytes

def decode_features_and_preprocess(features_binary: bytes) -> torch.Tensor :
    pd_columns = ['Flow ID', 'Host', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Content', 'DNS Query', 'DNS Resp']
    for i in range(0, 65 + 2) :
        pd_columns.append(f'Feat {i}')
    # create dataframe
    df = pd.DataFrame(columns=pd_columns)
    offset = 0
    # read first 4 bytes big-endian to get the number of flows
    n_flows = int.from_bytes(features_binary[:4], byteorder='big'); offset += 4
    for i in range(n_flows) :
        # read has_dns
        has_dns = features_binary[offset] != 0; offset += 1
        # read client_ip
        client_ip = int.from_bytes(features_binary[offset:offset+4], byteorder='big'); offset += 4
        # read client_port
        client_port = int.from_bytes(features_binary[offset:offset+2], byteorder='big'); offset += 2
        # read server_ip
        server_ip = int.from_bytes(features_binary[offset:offset+4], byteorder='big'); offset += 4
        # read server_port
        server_port = int.from_bytes(features_binary[offset:offset+2], byteorder='big'); offset += 2
        # read first_pkt_ts, microseconds from 1970 UTC
        first_pkt_ts = int.from_bytes(features_binary[offset:offset+8], byteorder='big'); offset += 8
        # read 2 int features
        feat_int = [0, 0]
        for j in range(2) :
            feat_int[j] = int.from_bytes(features_binary[offset:offset+4], byteorder='big'); offset += 4
        # read 65 float features
        feat_fp = [0.0 for _ in range(65)]
        for j in range(65) :
            feat_fp[j] = struct.unpack('>d', features_binary[offset:offset+8])[0]; offset += 8
        # read DPI content
        content_len = dpi_n_dpi_bytes
        dpi_bytes_len = int.from_bytes(features_binary[offset:offset+4], byteorder='big'); offset += 4
        content = features_binary[offset:offset+content_len]; offset += content_len
        # read DNS query
        dns_query = None
        if has_dns :
            query_len = int.from_bytes(features_binary[offset:offset+4], byteorder='big'); offset += 4
            dns_query = features_binary[offset:offset+query_len].decode('utf-8'); offset += query_len
        else :
            dns_query = 'N/A'
        first_pkt_ts = pd.Timestamp(first_pkt_ts, unit='us')
        # convert to pandas row
        row = [
            i, # Flow ID
            None, # Host
            client_ip, # Src IP
            client_port, # Src Port
            server_ip, # Dst IP
            server_port, # Dst Port
            first_pkt_ts, # Timestamp
            content[:dpi_bytes_len].hex(), # Content
            dns_query, # DNS Query
            'N/A', # DNS Resp
        ]
        for i in range(2) :
            row.append(feat_int[i])
        for i in range(65) :
            row.append(feat_fp[i])
        # append to dataframe
        df.loc[len(df)] = row
    df.to_csv(f'{TMP_DIR}/features_raw2.csv')
    # preprocess
    return preprocess(df)

def direct_binary_process(data: bytes) :
    # read 4 bytes for num_flow
    n_flows = int.from_bytes(data[:4], byteorder='big')
    int_feat_offset = 4
    fp_feat_offset = int_feat_offset + 4 * 2 * n_flows
    ts_feat_offset_1 = fp_feat_offset + 4 * 65 * n_flows
    ts_feat_offset_2 = ts_feat_offset_1 + 4 * n_flows
    dpi_bytes_offset = ts_feat_offset_2 + 4 * n_flows
    x_dpi_bytes = torch.zeros(1, n_flows, dpi_n_dpi_bytes, dtype = torch.uint8)
    # read from int_feat_offset into a numpy array [1, n_flows, 2] little-endian
    int_feat = np.frombuffer(data[int_feat_offset:fp_feat_offset], dtype=np.uint32).reshape(1, n_flows, 2)
    # read from fp_feat_offset into a numpy array [1, n_flows, 65] little-endian
    fp_feat = np.frombuffer(data[fp_feat_offset:ts_feat_offset_1], dtype=np.float32).reshape(1, n_flows, 65)
    # read from ts_feat_offset_1 into a numpy array [1, n_flows] little-endian
    ts_feat = np.frombuffer(data[ts_feat_offset_1:ts_feat_offset_2], dtype=np.float32).reshape(1, n_flows)
    # read from ts_feat_offset_2 into a numpy array [1, n_flows] little-endian
    ts_feat_2 = np.frombuffer(data[ts_feat_offset_2:dpi_bytes_offset], dtype=np.float32).reshape(1, n_flows)
    for i in range(n_flows) :
        # read 2 bytes from dpi_bytes_offset as length
        dpi_bytes_len = int.from_bytes(data[dpi_bytes_offset:dpi_bytes_offset+2], byteorder='big'); dpi_bytes_offset += 2
        # read dpi_bytes_len bytes from dpi_bytes_offset
        dpi_bytes = bytearray(data[dpi_bytes_offset:dpi_bytes_offset+dpi_bytes_len]); dpi_bytes_offset += dpi_bytes_len
        x_dpi_bytes[0, i, :dpi_bytes_len] = torch.tensor(dpi_bytes, dtype=torch.uint8)
    mask = torch.zeros(1, n_flows, dtype = torch.bool)[:, :dpi_max_flows]
    int_feat = torch.tensor(int_feat, dtype = torch.int32).long()[:, :dpi_max_flows]
    fp_feat = torch.tensor(fp_feat, dtype = torch.float32)[:,:dpi_max_flows]
    ts_feat = torch.tensor(ts_feat, dtype = torch.float32).long()[:, :dpi_max_flows]
    ts_feat_2 = torch.tensor(ts_feat_2, dtype = torch.float32).long()[:, :dpi_max_flows]
    return fp_feat, int_feat, ts_feat, ts_feat_2, x_dpi_bytes, mask

@app.post('/predict_ue') # gievn an UE IP, predict the score
async def predict_ue(req: UeRequest):
    global MODEL_DPI, NORAMLIZE_SOLUTION
    global ALL_INFER_TIME, ALL_FEAT_TIME, ALL_PREPROCESS_TIME
    if MODEL_DPI is None or NORAMLIZE_SOLUTION is None:
        return {"status": "model not loaded yet"}
    t0 = time.perf_counter()
    # retrieve features
    async with ClientSession() as session:
        async with session.post('http://127.0.0.1:5185/get_ue_stats', json={'ue_ip': req.ue_ip}) as response:
            # check if the response is successful
            if response.status != 200:
                print(f"Error: {response.status}")
                return {"score": 0, "t_features": 0, "t_inference": 0}
            # get features as binary blob
            features_binary = await response.content.read()
    t1 = time.perf_counter()
    t_features = t1 - t0
    t0 = time.perf_counter()
    dpi_x, dpi_xc, dpi_x_start_ts, dpi_x_end_ts, dpi_x_dpi_bytes, dpi_mask = direct_binary_process(features_binary)
    t1 = time.perf_counter()
    t_preprocess = t1 - t0
    # df_clean = features[0]
    # # save to df_clean.csv
    # df_clean.to_csv(f'{TMP_DIR}/df_clean2.csv')
    # return {"score": 1, "t_features": t_features, "t_inference": 0}
    t0 = time.perf_counter()
    with torch.no_grad() :
        score_dpi = MODEL_DPI(
            dpi_x.cuda(),
            dpi_xc.cuda(),
            dpi_x_start_ts,
            dpi_x_end_ts,
            dpi_x_dpi_bytes.cuda(),
            dpi_mask.cuda()
        ).softmax(dim = 1)[:, 1]
        score_dpi = score_dpi.cpu().item()
    t1 = time.perf_counter()
    t_inference = t1 - t0
    ALL_INFER_TIME.append(t_inference)
    ALL_FEAT_TIME.append(t_features)
    ALL_PREPROCESS_TIME.append(t_preprocess)
    if len(ALL_INFER_TIME) >= 100 and len(ALL_INFER_TIME) % 100 == 0 :
        mean_infer_time = np.mean(ALL_INFER_TIME)
        std_infer_time = np.std(ALL_INFER_TIME)
        mean_feat_time = np.mean(ALL_FEAT_TIME)
        std_feat_time = np.std(ALL_FEAT_TIME)
        mean_preprocess_time = np.mean(ALL_PREPROCESS_TIME)
        std_preprocess_time = np.std(ALL_PREPROCESS_TIME)
        print(f'Processed {len(ALL_INFER_TIME)} requests')
        print(f'inference time: {mean_infer_time}±{std_infer_time}s')
        print(f'feature time: {mean_feat_time}±{std_feat_time}s')
        print(f'preprocess time: {mean_preprocess_time}±{std_preprocess_time}s')
        print('GPU mem:', torch.cuda.max_memory_allocated() + torch.cuda.max_memory_reserved())
    return {"score": score_dpi, "t_features": t_features, "t_inference": t_inference, "t_preprocess": t_preprocess}


def test() :
    global NORAMLIZE_SOLUTION
    NORAMLIZE_SOLUTION = pickle.load(open(f'{TMP_DIR}/normalize_solution.pkl', 'rb'))
    df_ref = pd.read_csv('domainwatcher_tmp/ref.csv')
    x = [preprocess(df_ref)]
    t0 = time.perf_counter()
    dpi_x, dpi_xc, dpi_x_start_ts, dpi_x_end_ts, dpi_x_dpi_bytes, dpi_mask, _, _, _ = collate_fn(x)
    t1 = time.perf_counter()
    print('Time to preprocess:', t1 - t0)
    extract_normal_solution('normalize_solution.json')
    with open('test.bin', 'rb') as f :
        data = f.read()
    t0 = time.perf_counter()
    dpi_x2, dpi_xc2, dpi_x_start_ts2, dpi_x_end_ts2, dpi_x_dpi_bytes2, dpi_mask2 = direct_binary_process(data)
    t1 = time.perf_counter()
    print('Time to process binary:', t1 - t0)
    assert torch.allclose(dpi_x, dpi_x2)
    assert torch.allclose(dpi_xc, dpi_xc2)
    assert torch.allclose(dpi_x_start_ts, dpi_x_start_ts2)
    assert torch.allclose(dpi_x_end_ts, dpi_x_end_ts2)
    assert torch.allclose(dpi_x_dpi_bytes, dpi_x_dpi_bytes2)
    print('All tests passed')

if __name__ == '__main__':
    #test()
    uvicorn.run(app, host='0.0.0.0', port=5008)
