
from io import BytesIO
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

import preprocess_flow_stats
import pandas as pd
import pickle
import torch
import numpy as np
import os
import asyncio
import trainer
from evaluate import run_per_chunk_scores, load_dns_model
import yaml
import uuid
import httpx
import tqdm
from contextlib import asynccontextmanager
from trainer import Args

app = fastapi.FastAPI()

# read usf_config.yaml
with open('usf_config.yaml') as f:
    usf_config = yaml.safe_load(f)


SMF_LIST = []
TMP_DIR = usf_config['usf']['tmp_dir']
PCAP_FILES = []

NF_UUID = None
NF_SERVICE_UUID = None

async def nf_register():
    myip = ""
    global NF_UUID, NF_SERVICE_UUID, SMF_LIST
    NF_UUID = str(uuid.uuid4())
    NF_SERVICE_UUID = str(uuid.uuid4())
    data = {
        "nfInstanceId": NF_UUID,
        "nfInstanceName": usf_config['nf']['name'],
        "nfType": "AF",
        "nfStatus": {
            "statusRegistered": 1
        },
        "plmnList": usf_config['nf']['plmns'],
        "sNssais": usf_config['nf']['nssai'],
        "fqdn": usf_config['nf']['name'],
        "ipv4Addresses": [
            myip
        ],
        "allowedPlmns": usf_config['nf']['allowed_plmns'],
        "allowedNssais": usf_config['nf']['allowed_nssai'],
        "nfServicePersistence": False,
        "nfServices": [
            {
                "serviceInstanceId": NF_SERVICE_UUID,
                "serviceName": "nusf-training",
                "versions": [
                    {
                        "apiVersionInUri": "/v1",
                        "apiFullVersion": "/nusf-training/v1"
                    }
                ],
                "scheme": "http",
                "nfServiceStatus": "REGISTERED"
            }
        ],
        "nfProfileChangesSupportInd": False,
        "nfProfileChangesInd": False,
        "lcHSupportInd": False,
        "olcHSupportInd": False
    }
    nrf_uri = usf_config['nf']['nrf_uri']
    put_uri = f'{nrf_uri}/nnrf-nfm/v1/nf-instances/{NF_UUID}'
    # post to nrf using http2
    async with httpx.AsyncClient(http2=True) as client:
        resp = await client.put(put_uri, json=data)
        if resp.status_code != 200:
            raise Exception('failed to register to NRF')
    SMF_LIST = await get_list_of_smfs()

async def get_list_of_smfs() :
    nrf_uri = usf_config['nf']['nrf_uri']
    get_uri = f'{nrf_uri}/nnrf-disc/v1/nf-instances?target-nf-type=SMF&requester-nf-type=AF&service-names=nsmf-c5g'
    async with httpx.AsyncClient(http2=True) as client:
        resp = await client.get(get_uri)
        if resp.status_code != 200:
            raise Exception('failed to get list of SMFs')
        resp_json = resp.json()
        return resp_json['nfInstances']

async def nf_deregister():
    global NF_UUID, NF_SERVICE_UUID
    nrf_uri = usf_config['nf']['nrf_uri']
    delete_uri = f'{nrf_uri}/nnrf-nfm/v1/nf-instances/{NF_UUID}'
    async with httpx.AsyncClient(http2=True) as client:
        resp = await client.delete(delete_uri)
        if resp.status_code != 200:
            raise Exception('failed to deregister to NRF')

@asynccontextmanager
async def lifespan(app: FastAPI):
    #await nf_register()
    yield
    #await nf_deregister()

class AddPcapRequest(BaseModel):
    pcap_filepath: str # pre-mixed and pre-splited pcap file
    is_malicious: bool
    sample_id: str # unique identifier for the sample before splitting
    chunk_idx: int # 0 is the first 30 minutes, 1 is the next 30 minutes, and so on

@app.post('/nusf_training/v1/pcap')
async def add_pcap(req: AddPcapRequest):
    PCAP_FILES.append((req.pcap_filepath, req.chunk_idx, req.sample_id, req.is_malicious))
    return {'status': 'ok'}

TRAINING_PROGRESS = 'not started'
TRAINING_RESULTS = None

# due to a data migration disaster, we can not provide the model parameters that give the result in the paper.
# here is one set of parameters that gives similar results
TRAINING_CONFIGS = {
    'ue_ip_range': '10.10.0.0/16',
    'eval_fpr_target': 0.01,
    'eval_off_path_limit': 0.4,
    'eval_hours': 4, # 4 hours, 8 chunks
    'dns_ben_w': 3, # or other values
    'dns_mal_w': 1, # or other values
    'dns_score_scale': 1, # or other values
    'dns_weight_decay': 0.1, # or other values
    'dns_label_smoothing': 0.05, # or other values
    'dns_temperture': 0.2, # or other values
    'dns_lr': 1e-3, # or other values
    'dns_epochs': '500x400x450', # or other values
    'dns_max_domains': 512, # or other values
    'dns_batch_size': 128, # or other values
    'dns_offset_prob': 1, # or other values
    'dns_offset_ratio': 0.5, # or other values
    'dns_mal_drop': 0.05, # or other values
    'dns_ben_drop': 0.05, # or other values
    'dns_increase_mal_domain_occurence': False,
    'dns_filter_out_android_domain': True,
    'dns_qbits': 16, # or 32 if you want
    'dpi_enabled_fp16': False,
    'dpi_norm_version': 'all',
    'dpi_posweight': 1, # or other values
    'dpi_temperture': 0.1, # or other values
    'dpi_lr': 3e-4, # or other values
    'dpi_label_smoothing': 0.01, # or other values
    'dpi_flow_dropout_ratio': 0.01, # or other values
    'dpi_max_flows': 3000,
    'dpi_n_dpi_bytes': 160,
    'dpi_epochs': '5x3x4', # or other values
    'dpi_batch_size': 24 # or other values
}

class namespace :
    def __init__(self, **kwargs) :
        self.__dict__.update(kwargs)

def read_csv_and_convert(f) :
    df = pd.read_csv(f)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', format='ISO8601')
    df = df.dropna(subset = ['Timestamp'])  # Drop rows with NaT values
    df['Start TS'] = (df['Timestamp'] - df['Timestamp'].min()).dt.total_seconds()
    df['End TS'] = df['Start TS'] + df['Feat 2']
    ts = (df['Timestamp'].max() - df['Timestamp'].min()).total_seconds()
    df = df.drop(['Flow ID', 'Src IP','Src Port','Dst IP','Dst Port','Timestamp', 'DNS Query', 'DNS Resp'], axis=1)
    df = df.fillna(0)
    cols = [i for i in df.columns if i not in ["Content", "Host"]]
    for col in cols:
        df[col] = pd.to_numeric(df[col])
    df = df.dropna()
    return df

async def model_training_main(model_id: int) :
    return
    # not tested
    global TRAINING_PROGRESS, TRAINING_RESULTS
    assert model_id == 0
    TRAINING_PROGRESS = 'started'
    args = namespace(**TRAINING_CONFIGS)
    print('training started for model_id', model_id)
    # step 1: extract features to tmp dir
    TRAINING_PROGRESS = 'extracting features'
    ue_ip_range = TRAINING_CONFIGS['ue_ip_range']
    all_benign_csv = []
    all_malicious_csv = []
    for pcap_file, chunk_idx, sample_id, is_malicious in PCAP_FILES:
        pcap_filename = os.path.basename(pcap_file)
        pcap_filename_no_ext = os.path.splitext(pcap_filename)[0]
        mal_ext = 'mal' if is_malicious else 'ben'
        out_dir = os.path.join(TMP_DIR, pcap_filename_no_ext + f'-{mal_ext}.csv')
        os.system(f"./traffic_collector --mode pcap --pcap-filename \"{pcap_file}\" --out-csv-filename \"{out_dir}\" --ue-ip-range {ue_ip_range}")
        if is_malicious:
            all_malicious_csv.append((out_dir, chunk_idx, sample_id))
        else:
            all_benign_csv.append((out_dir, chunk_idx, sample_id))
    TRAINING_PROGRESS = 'converting save format'
    for f, _, _ in all_benign_csv + all_malicious_csv :
        df = read_csv_and_convert(f)
        df.to_csv(f, index=False)
    all_eval_results = {}
    # step 2: divide into 5 folds of train and test
    for fold in range(5) :
        TRAINING_PROGRESS = f'fold {fold+1}/5 creating normalize solution'
        n_ben_samples_per_fold = len(all_benign_csv) // 5
        n_mal_samples_per_fold = len(all_malicious_csv) // 5
        eval_benign_csv = all_benign_csv[fold * n_ben_samples_per_fold: (fold + 1) * n_ben_samples_per_fold]
        eval_malicious_csv = all_malicious_csv[fold * n_mal_samples_per_fold: (fold + 1) * n_mal_samples_per_fold]
        train_benign_csv = all_benign_csv[:fold * n_ben_samples_per_fold] + all_benign_csv[(fold + 1) * n_ben_samples_per_fold:]
        train_malicious_csv = all_malicious_csv[:fold * n_mal_samples_per_fold] + all_malicious_csv[(fold + 1) * n_mal_samples_per_fold:]
        #   step 3: create normalize solution
        trainset = train_benign_csv + train_malicious_csv
        # read csv files
        TRAINING_PROGRESS = f'fold {fold+1}/5 creating normalize solution > reading csv files'
        train_df = pd.concat([pd.read_csv(f) for f, _, _ in trainset])
        normalize_solution = {}
        TRAINING_PROGRESS = f'fold {fold+1}/5 creating normalize solution > creating solution'
        for column in train_df.columns :
            if column not in ['pcap_file', 'Content', 'Start TS', 'End TS', 'Host'] :
                normalize_solution[column] = preprocess_flow_stats.create_solution(train_df, column)
        # save normalize solution
        TRAINING_PROGRESS = f'fold {fold+1}/5 creating normalize solution > saving solution'
        with open(os.path.join(TMP_DIR, f'normalize_solution_{fold}.pkl'), 'wb') as f :
            pickle.dump(normalize_solution, f)
        #   step 4: train DNS model
        TRAINING_PROGRESS = f'fold {fold+1}/5 training DNS model'
        trainset_dns = [f'{f}.dns.txt' for f, _, _ in trainset]
        model_dns = trainer.train_dns(trainset_dns, args)
        # save dns model
        dns_savefile = os.path.join(TMP_DIR, f'dns_model_{fold}.pth')
        torch.save({'sd': model_dns.state_dict(), 'domains': model_dns.all_domains}, dns_savefile)
        #   step 5: train DPI model
        TRAINING_PROGRESS = f'fold {fold+1}/5 training DPI model'
        model_dpi = trainer.train_dpi([f for f, _, _ in trainset], normalize_solution, args)
        dpi_savefile = os.path.join(TMP_DIR, f'dpi_model_{fold}.pth')
        torch.save(model_dpi.state_dict(), dpi_savefile)
        #   step 6: eval both models
        TRAINING_PROGRESS = f'fold {fold+1}/5 evaluating'
        per_fold_result = run_per_chunk_scores(eval_benign_csv, eval_malicious_csv, model_dpi, model_dns, args, normalize_solution)
        all_eval_results[f'fold-{fold}'] = per_fold_result
    # step 7: threshold search
    TRAINING_PROGRESS = 'threshold search'
    best_thres, best_result = trainer.find_threshold(all_eval_results, args.eval_hours, args.eval_fpr_target, args.eval_off_path_limit)
    if best_thres is None :
        TRAINING_PROGRESS = 'training failed, no threshold found'
        return
    # step 8: train final model
    TRAINING_PROGRESS = 'training final model > creating normalize solution'
    trainset = all_benign_csv + all_malicious_csv
    train_df = pd.concat([pd.read_csv(f) for f, _, _ in trainset])
    normalize_solution = {}
    for column in train_df.columns :
        if column not in ['pcap_file', 'Content', 'Start TS', 'End TS', 'Host'] :
            normalize_solution[column] = preprocess_flow_stats.create_solution(train_df, column)
    # save normalize solution
    with open(os.path.join(TMP_DIR, 'normalize_solution.pkl'), 'wb') as f :
        pickle.dump(normalize_solution, f)
    # train DNS model
    TRAINING_PROGRESS = 'training final model > training DNS model'
    trainset_dns = [f'{f}.dns.txt' for f, _, _ in trainset]
    model_dns = trainer.train_dns(trainset_dns, args)
    # save dns model
    dns_savefile = os.path.join(TMP_DIR, 'dns_model.pth')
    torch.save({'sd': model_dns.state_dict(), 'domains': model_dns.all_domains}, dns_savefile)
    # train DPI model
    TRAINING_PROGRESS = 'training final model > training DPI model'
    model_dpi = trainer.train_dpi([f for f, _, _ in trainset], normalize_solution, args)
    dpi_savefile = os.path.join(TMP_DIR, 'dpi_model.pth')
    torch.save(model_dpi.state_dict(), dpi_savefile)
    # step 10: update TRAINING_RESULTS
    TRAINING_RESULTS = {
        'dns_model': dns_savefile,
        'dpi_model': dpi_savefile,
        'normalize_solution': os.path.join(TMP_DIR, 'normalize_solution.pkl'),
        'threshold': best_thres,
        'best_result': best_result
    }
    TRAINING_PROGRESS = 'not started'

class ModelTrainingRequest(BaseModel):
    model_id: int

@app.post('/nusf_training/v1/train')
async def train_model(req: ModelTrainingRequest):
    global TRAINING_PROGRESS
    if TRAINING_PROGRESS != 'not started':
        return {'status': 'error', 'message': 'training already in progress'}
    asyncio.create_task(model_training_main(req.model_id))
    return {'status': 'ok', 'progress': TRAINING_PROGRESS}

@app.get('/nusf_training/v1/train')
async def get_training_progress():
    global TRAINING_PROGRESS
    return {'status': 'ok', 'progress': TRAINING_PROGRESS}

@app.get('/nusf_training/v1/result')
async def get_training_result():
    global TRAINING_RESULTS
    return {'status': 'ok', 'result': TRAINING_RESULTS}


class UpdateConfigRequest(BaseModel):
    new_config: dict

@app.post('/nusf_training/v1/config')
async def update_training_config(req: UpdateConfigRequest):
    global TRAINING_CONFIGS
    TRAINING_CONFIGS.update(req.new_config)
    return {'status': 'ok'}

@app.get('/nusf_training/v1/config')
async def get_training_config():
    global TRAINING_CONFIGS
    return {'status': 'ok', 'config': TRAINING_CONFIGS}

class DeployModelRequest(BaseModel):
    model_id: int

async def post_multipart_related(url: str, json_data: dict, binary_data: bytes) -> httpx.Response:
    """
    Posts a multipart/related request to `url` with:
      - Part 1: JSON content (application/json)
      - Part 2: Binary blob (application/vnd.c5g)
    using httpx in an async context.
    """

    # Generate a unique boundary string
    boundary = f'Boundary-{uuid.uuid4().hex}'

    # Build the multipart/related body
    body_buffer = BytesIO()

    # -- Part 1: JSON
    body_buffer.write(f'--{boundary}\r\n'.encode('utf-8'))
    body_buffer.write(b'Content-Type: application/json\r\n\r\n')
    body_buffer.write(json.dumps(json_data).encode('utf-8'))
    body_buffer.write(b'\r\n')

    # -- Part 2: Binary
    body_buffer.write(f'--{boundary}\r\n'.encode('utf-8'))
    body_buffer.write(b'Content-Type: application/vnd.c5g\r\n\r\n')
    body_buffer.write(binary_data)
    body_buffer.write(b'\r\n')

    # -- Close boundary --
    body_buffer.write(f'--{boundary}--\r\n'.encode('utf-8'))

    # Prepare headers for multipart/related
    headers = {
        "Content-Type": f"multipart/related; boundary={boundary}"
    }

    # Send the request asynchronously
    async with httpx.AsyncClient(http2=True,http1=False) as client:
        response = await client.post(
            url=url,
            headers=headers,
            content=body_buffer.getvalue()
        )

    return response

async def send_binary_file(filename: str, usage: str, model_id: int, deployment_id: str, target_nf_uri: str) :
    # send file in 8KB chunks
    filesize = os.path.getsize(filename)
    #print('sending', filename, 'to', target_nf_uri, 'size', filesize)
    with open(filename, 'rb') as f:
        offset = 0
        total_chunks = (filesize + 8191) // 8192
        # print('filesize', filesize)
        # print('total chunks', total_chunks)
        chunk_id = 0
        pb = tqdm.tqdm(total=total_chunks)
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            await post_multipart_related(target_nf_uri, {
                'deploymentId': deployment_id,
                'modelId': model_id,
                'usage': usage,
                'offset': offset,
                'size': filesize,
                'length': len(chunk),
                'chunkId': chunk_id,
                'totalChunks': total_chunks
            }, chunk)
            offset += len(chunk)
            chunk_id += 1
            pb.update(1)
        pb.close()

async def deploy_task(req: DeployModelRequest) :
    deployment_id = str(uuid.uuid4())
    normalize_solution_file = TRAINING_RESULTS['normalize_solution']
    dns_model_file = TRAINING_RESULTS['dns_model']
    dpi_model_file = TRAINING_RESULTS['dpi_model']
    #print(SMF_LIST)
    for smf in SMF_LIST:
        smf_ip = smf['ipv4Addresses'][0]
        smf_uri = f'http://{smf_ip}/nsmf-c5g/v1/deploy'
        print('deploying to', smf_uri)
        await send_binary_file(normalize_solution_file, 'normalize_solution', req.model_id, deployment_id, smf_uri)
        await send_binary_file(dpi_model_file, 'dpi_model', req.model_id, deployment_id, smf_uri)

    args = Args()

    dns_model = load_dns_model(dns_model_file, args)
    rep_scores = dns_model.get_domain_reputation_scores()
    with open('android_domains.txt', 'r') as fp :
        all_android_domains = [line.strip() for line in fp]
        all_android_domains = [x for x in all_android_domains if x]
        for domain in all_android_domains :
            if domain not in rep_scores :
                rep_scores[domain] = 0
    rep_scores2 = [{'name': k, 'score': v} for k, v in rep_scores.items()]
    for smf in SMF_LIST:
        smf_ip = smf['ipv4Addresses'][0]
        smf_uri = f'http://{smf_ip}/nsmf-c5g/v1/deploy'
        print('[DNS] deploying to', smf_uri)
        await post_multipart_related(smf_uri, 
            {
                'deployment_id': deployment_id,
                'model_id': req.model_id,
                'usage': 'domains',
                'scores': rep_scores2,
                'dns_on_threshold': TRAINING_RESULTS['threshold'][0],
                'dns_off_threshold': TRAINING_RESULTS['threshold'][1],
                'dpi_threshold': TRAINING_RESULTS['threshold'][2],
                'dns_score_scale': dns_model.quant.scale
            },
            b'\0'
        )
    print('done deploying')

@app.post('/nusf_training/v1/deploy')
async def deploy_model(req: DeployModelRequest):
    global TRAINING_RESULTS
    if TRAINING_RESULTS is None:
        return {'status': 'error', 'message': 'no trained model available'}

    asyncio.create_task(deploy_task(req))

    return {'status': 'ok'}

async def test_deploy() :
    model_id = 0
    deployment_id = str(uuid.uuid4())
    normalize_solution_file = 'normalize_solution.pkl'
    dns_model_file = 'dns.pth'
    dpi_model_file = 'dpi.pth'
    SMF_LIST = ['172.18.0.8']
    print(SMF_LIST)

    args = Args()

    dns_model = load_dns_model(dns_model_file, args)
    rep_scores = dns_model.get_domain_reputation_scores()
    with open('android_domains.txt', 'r') as fp :
        all_android_domains = [line.strip() for line in fp]
        all_android_domains = [x for x in all_android_domains if x]
        for domain in all_android_domains :
            if domain not in rep_scores :
                rep_scores[domain] = 0
    rep_scores2 = [{'name': k, 'score': v} for k, v in rep_scores.items()]
    print(rep_scores2)
    print(len(rep_scores2))
    print(dns_model.quant.scale)
    for smf in SMF_LIST:
        smf_ip = smf
        smf_uri = f'http://{smf_ip}/nsmf-c5g/v1/deploy'
        print('[DNS] deploying to', smf_uri)
        await post_multipart_related(smf_uri, 
            {
                'deploymentId': deployment_id,
                'modelId': model_id,
                'usage': 'domains',
                'scores': rep_scores2,
                'dnsOnThreshold': 0.6,
                'dnsOffThreshold': 0.49,
                'dpiThreshold': 0.55,
                'scoreScale': dns_model.quant.scale
            },
            b'\0'
        )

    for smf in SMF_LIST:
        smf_ip = smf
        smf_uri = f'http://{smf_ip}/nsmf-c5g/v1/deploy'
        print('deploying to', smf_uri)
        await send_binary_file(normalize_solution_file, 'normalize_solution', model_id, deployment_id, smf_uri)
        await send_binary_file(dpi_model_file, 'dpi_model', model_id, deployment_id, smf_uri)

    
    print('done deploying')

if __name__ == '__main__':
    asyncio.run(test_deploy())
    #uvicorn.run(app, host='0.0.0.0', port=80)
