
use libpfcp::{messages::ModelDeploymentRequest, models::ModelWeightsBlob, IDAllocator};
use log::{warn, info};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, Barrier, mpsc::{self, Receiver, Sender}};
use std::{collections::{HashMap, HashSet}, fs::{File, OpenOptions}, io::{self, Seek, SeekFrom, Write}, sync::{Arc, Mutex}};
use lazy_static::lazy_static;
use std::error::Error;
use std::path::Path;
use tokio::io::AsyncReadExt;

use crate::datapath::{BackendReport, Report};

use super::bfruntime::{self, bfrt::{Update, TableKey, KeyField, TableEntry, key_field::{self, Exact}, table_entry, entity::{self}, update, Entity, TableData, DataField, data_field}, P4TableInfo};

const INFERENCE_SERVER: &str = "128.226.117.217:5008";

async fn upload_files(
    url: &str, 
    preprocessing_pkl: &str, 
    checkpoint_pth: &str
) -> Result<(), Box<dyn Error>> {
    // Create a client with a longer timeout for large files
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300)) // 5 minute timeout
        .build()?;
    
    // Create the multipart form
    let mut form = reqwest::multipart::Form::new();
    
    // Handle preprocessing_pkl file
    if !preprocessing_pkl.is_empty() {
        // Open the preprocessing file
        let mut file = tokio::fs::File::open(preprocessing_pkl).await?;
        // Read the file content
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).await?;
        
        // Create a multipart part with the file name extracted from the path
        let file_name = Path::new(preprocessing_pkl)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("normalize_solution.pkl");
        
        // Add the file to the form
        let part = reqwest::multipart::Part::bytes(buffer)
            .file_name(file_name.to_string())
            .mime_str("application/octet-stream")?;
        
        form = form.part("preprocessing_pkl", part);
    }
    
    // Handle checkpoint_pth file
    if !checkpoint_pth.is_empty() {
        // Open the checkpoint file
        let mut file = tokio::fs::File::open(checkpoint_pth).await?;
        // Read the file content
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).await?;
        
        // Create a multipart part with the file name extracted from the path
        let file_name = Path::new(checkpoint_pth)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model.pth");
        
        // Add the file to the form
        let part = reqwest::multipart::Part::bytes(buffer)
            .file_name(file_name.to_string())
            .mime_str("application/octet-stream")?;
        
        form = form.part("checkpoint_pth", part);
    }
    
    // Send the request
    let response = client.post(url)
        .multipart(form)
        .send()
        .await?;
    
    // Check the response
    if response.status().is_success() {
        info!("Normal solution and DPI model uploaded successfully!");
        info!(" === Model is ready ===");
        Ok(())
    } else {
        let error_text = response.text().await?;
        Err(format!("Upload failed: {}", error_text).into())
    }
}

// CRC-ARC implementation
struct CrcArc {
    table: [u16; 256],
}

impl CrcArc {
    fn new() -> Self {
        let mut table = [0u16; 256];
        for i in 0..256 {
            let mut crc = i as u16;
            for _ in 0..8 {
                if (crc & 0x0001) != 0 {
                    crc = (crc >> 1) ^ 0xa001;
                } else {
                    crc >>= 1;
                }
            }
            table[i] = crc;
        }
        CrcArc { table }
    }

    fn calc(&self, data: &[u8]) -> u16 {
        let mut crc = 0;
        for &byte in data {
            let index = (crc as u8 ^ byte) as usize;
            crc = (crc >> 8) ^ self.table[index];
        }
        crc
    }
}

// Calculate CRC32 using Rust's crc32fast crate
fn crc32_calc(data: &[u8]) -> u32 {
    // In a real implementation, you would use the crc32fast crate
    // For demonstration, we'll implement a simple version
    let mut crc = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

// Closest power of 2
fn cl_p2(n: usize) -> usize {
    let lg2 = (n as f64).log2() as usize;
    2usize.pow(lg2 as u32)
}

fn crc32(domain: &str, variant: &str, reverse: bool) -> u64 {
    let max_bytes_per_label = 31;
    let max_bytes = max_bytes_per_label;
    let closest_power2 = cl_p2(max_bytes);
    
    let mut sum_reg: Vec<u8> = Vec::new();
    let mut sum_16: Vec<u8> = Vec::new();
    
    let split_domains: Vec<&str> = if reverse {
        domain.split('.').rev().collect()
    } else {
        domain.split('.').collect()
    };
    
    for label in split_domains {
        let mut i = closest_power2;
        let mut temp_label = label;
        
        while i >= 1 {
            if temp_label.len() >= i {
                if i >= 16 {
                    sum_16.extend_from_slice(&temp_label[..i].as_bytes());
                } else {
                    sum_reg.extend_from_slice(&temp_label[..i].as_bytes());
                }
                temp_label = &temp_label[i..];
            } else {
                if i >= 16 {
                    sum_16.extend(vec![0; i]);
                } else {
                    sum_reg.extend(vec![0; i]);
                }
            }
            i = i / 2;
        }
    }
    
    let crc_arc = CrcArc::new();
    let total = if variant == "16" {
        crc_arc.calc(&sum_reg) as u64
    } else {
        crc32_calc(&sum_reg) as u64
    };
    
    if max_bytes >= 16 {
        let sum_16_crc = if variant == "16" {
            crc_arc.calc(&sum_16) as u64
        } else {
            crc32_calc(&sum_16) as u64
        };
        
        return (total + sum_16_crc) % (2u64.pow(variant.parse::<u32>().unwrap()));
    }
    
    total
}

fn domain_crc(domain: &str) -> u32 {
    crc32(domain, "32", false) as u32
}

fn get_cur_ts() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_micros() as u64
}

fn extract_to_label_vec(dst: &mut Vec<(usize, Vec<u8>)>, domain: &[u8], start_len: usize) {
    if domain.len() > start_len + start_len - 1 {
        warn!("skipping subdomain {}", String::from_utf8(domain.to_vec()).unwrap());
        return;
    }
    let mut domain = domain;
    let mut segment_idx = 0;
    let mut try_length = start_len;
    while try_length != 0 && domain.len() != 0 {
        if domain.len() >= try_length {
            let left_part = &domain[..try_length];
            domain = &domain[try_length..];
            let (l, seg) = dst.get_mut(segment_idx).unwrap();
            assert_eq!(*l, try_length);
            *seg = left_part.to_owned();
        }
        try_length /= 2;
        segment_idx += 1;
    }
    assert_eq!(domain.len(), 0);
}

fn populate_tld_table_impl(tld_list: Vec<String>, table_info: &P4TableInfo) -> Vec<Update> {
    // populate TLD table
    let mut updates = vec![];
    let mut n_added_tlds = 0usize;
    for tld in tld_list {
        // allocate space
        let mut labels = Vec::with_capacity(2);
        for label_id in 0..2usize {
            let mut label_segments = Vec::with_capacity(5);
            for length in [16, 8, 4, 2, 1] {
                label_segments.push((length, vec![0u8; length]));
            }
            labels.push(label_segments);
        }
        
        // TODO:
        let splits = tld.split(".").collect::<Vec<_>>();
        if splits.len() == 2 {
            // for i in 0..splits.len() {
            //     extract_to_label_vec(labels.get_mut(i).unwrap(), splits[i].as_bytes(), 16);
            // }
            extract_to_label_vec(labels.get_mut(0).unwrap(), splits[1].as_bytes(), 16);
            extract_to_label_vec(labels.get_mut(1).unwrap(), splits[0].as_bytes(), 16);
        } else {
            warn!("skipping TLD {}", tld);
            continue;
        }
        
        let mut table_entry = TableKey {
            fields: vec![]
        };
        for (i, label) in labels.iter().enumerate() {
            for (length, value) in label {
                let key_field = KeyField {
                    field_id: table_info.get_key_id_by_name("pipe.Egress.dns_domain_parts_2", &format!("hdr.label{}_{}.l", i + 1, length)),
                    match_type: Some(key_field::MatchType::Exact(Exact {
                        value: value.clone()
                    }))
                };
                table_entry.fields.push(key_field);
            }
        }
        let table_data = TableData {
            action_id: table_info.get_action_id_by_name("pipe.Egress.dns_domain_parts_2", "Egress.dns_split_labels_action_2"),
            fields: vec![]
        };
        let table_entry = TableEntry {
            table_id: table_info.get_table_by_name("pipe.Egress.dns_domain_parts_2"),
            data: Some(table_data),
            is_default_entry: false,
            table_read_flag: None,
            table_mod_inc_flag: None,
            entry_tgt: None,
            table_flags: None,
            value: Some(table_entry::Value::Key(table_entry)),
        };
        let entity = Entity {
            entity: Some(entity::Entity::TableEntry(table_entry))
        };
        let update = Update {
            r#type: update::Type::Insert as _,
            entity: Some(entity)
        };
        updates.push(update);
        n_added_tlds += 1;
    }
    info!("Added {} TLDs", n_added_tlds);
    updates
}

use std::fs::read_to_string;

fn read_lines(filename: &str) -> Vec<String> {
    read_to_string(filename) 
        .unwrap()  // panic on possible file-reading errors
        .lines()  // split the string into an iterator of string slices
        .map(String::from)  // make each slice into a string
        .collect()  // gather them together into a vector
}

pub fn populate_tld_table(tld_list_filename: &str, table_info: &P4TableInfo) -> Vec<Update> {
    // populate TLD table
    let tlds = read_lines(tld_list_filename);
    populate_tld_table_impl(tlds, table_info)
}

pub struct DomainWatcherGlobalContext {
    // global context for domain watcher
}

pub async fn deploy_model(
    req: ModelDeploymentRequest,
    bfrt: Arc<RwLock<Option<bfruntime::BFRuntime>>>
) -> bool {
    // check for existing model
    let mut model_lock = DOMAIN_WATCHER_STATES.model.write().await;
    let model = if model_lock.is_none() {
        // deploy model
        let model = DomainWatcherModel::new(req.model_id.model_id, req.model_id.deployment_name);
        info!("Deploying model {}", model.deployment_name);
        *model_lock = Some(model);
        model_lock.as_mut().unwrap()
    } else {
        let model = model_lock.as_mut().unwrap();
        if model.deployment_name != req.model_id.deployment_name {
            warn!("model deployment name mismatch");
            return false;
        }
        model
    };
    if let Some(scores) = req.scores {
        let incoming_scores = scores.domain_scores.into_iter().map(|(domain, score)| SingleDomain { domain, score }).collect::<Vec<_>>();
        info!("Added {} domain scores", incoming_scores.len());
        model.domain_scores.extend(incoming_scores);
        model.total_domains = scores.total_domains;
        if model.domain_scores.len() == scores.total_domains as usize {
            info!("Received all domain scores");
            // write to dataplane
            info!("Received all domain scores, writing to dataplane for deployment {}", model.deployment_name);
            let mut bfrt_guard = bfrt.write().await;
            let bfrt = bfrt_guard.as_mut().unwrap();
            let table_info = &bfrt.table_info;
            let updates = model.write_to_dataplane(table_info, model.model_id);
            bfrt.write_update_no_transaction(updates).await.unwrap();
        }
    }
    //info!("got action {:?}", req.action);
    if let Some(dns_score_scale) = req.action.dns_score_scale {
        model.dns_score_scale = Some(dns_score_scale);
        info!("DNS score scale set to {}", dns_score_scale);
    }
    if let Some(dns_on_threshold) = req.action.dns_on_threshold {
        model.dns_on_threshold = Some(dns_on_threshold);
        info!("DNS on threshold set to {}", dns_on_threshold);
    }
    if let Some(dns_off_threshold) = req.action.dns_off_threshold {
        model.dns_off_threshold = Some(dns_off_threshold);
        info!("DNS off threshold set to {}", dns_off_threshold);
    }
    if let Some(dpi_threshold) = req.action.dpi_threshold {
        model.dpi_threshold = Some(dpi_threshold);
        info!("DPI threshold set to {}", dpi_threshold);
    }
    if let Some(blob) = req.blob {
        let filename = match blob.flags {
            10 => {
                "normal_solution.pkl"
            }
            11 => {
                "dpi_model.pth"
            }
            _ => unreachable!()
        };
        if filename == "normal_solution.pkl" {
            let manager = model.normal_solution_file.get_or_insert_with(|| {
                FileChunkManager::new(filename.into()).unwrap()
            });
            manager.recv_chunk(&blob).unwrap();
        } else if filename == "dpi_model.pth" {
            let manager = model.dpi_model_file.get_or_insert_with(|| {
                FileChunkManager::new(filename.into()).unwrap()
            });
            manager.recv_chunk(&blob).unwrap();
        }
    }
    model.is_dpi_ready = model.dpi_threshold.is_some() && model.dpi_model_file.is_some() && model.dpi_model_file.as_ref().unwrap().is_complete();
    model.is_normal_solution_ready = model.normal_solution_file.is_some() && model.normal_solution_file.as_ref().unwrap().is_complete();
    model.is_dns_ready = model.domain_scores.len() == model.total_domains as usize && model.dns_score_scale.is_some() && model.dns_on_threshold.is_some() && model.dns_off_threshold.is_some();
    //info!("DNS ready={}, DPI ready={}, Normal solution ready={}", model.is_dns_ready, model.is_dpi_ready, model.is_normal_solution_ready);
    if model.is_dpi_ready && model.is_normal_solution_ready {
        // upload files
        if let Some(normal_solution_file) = &model.normal_solution_file {
            if let Some(dpi_model_file) = &model.dpi_model_file {
                // upload files
                let url = format!("http://{}/upload_model", INFERENCE_SERVER);
                let preprocessing_pkl = "normal_solution.pkl";
                let checkpoint_pth = "dpi_model.pth";
                // spawn a task to upload files, let it upload, don't block
                tokio::spawn(async move {
                    info!("Uploading normal solution and DPI model to inference server");
                    if let Err(e) = upload_files(&url, preprocessing_pkl, checkpoint_pth).await {
                        warn!("Failed to upload files: {}", e);
                    }
                });
            }
        }
    }
    let is_model_ready = model.is_dns_ready && model.is_dpi_ready && model.is_normal_solution_ready;
    if is_model_ready {
        info!("Model is about to be ready");
    } else {
        //info!("DNS ready={}, DPI ready={}, Normal solution ready={}", model.is_dns_ready, model.is_dpi_ready, model.is_normal_solution_ready);
        //info!("Normal progress={}%, dpi progress={}%", model.normal_solution_file.as_ref().map(|f| f.completion_percentage()).unwrap_or(0.0), model.dpi_model_file.as_ref().map(|f| f.completion_percentage()).unwrap_or(0.0));
    }
    is_model_ready
}

pub struct FileChunkManager {
    filename: String,
    file: File,
    received_chunks: HashSet<u64>,
    total_chunks: u64,
    total_size: u64,
    flags: Option<u8>,
}

impl FileChunkManager {
    /// Creates a new FileChunkManager with the specified filename
    pub fn new(filename: String) -> io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&filename)?;
            
        Ok(FileChunkManager {
            filename,
            file,
            received_chunks: HashSet::new(),
            total_chunks: 0,
            total_size: 0,
            flags: None,
        })
    }

    /// Receives a chunk of data and writes it to the file
    /// Returns true if the file is complete after this chunk
    pub fn recv_chunk(&mut self, chunk: &ModelWeightsBlob) -> io::Result<bool> {
        // Initialize or validate metadata on first chunk
        if self.received_chunks.is_empty() {
            self.total_chunks = chunk.total_chunks;
            self.total_size = chunk.total_size;
            self.flags = Some(chunk.flags);
            //info!("setting total_chunks={} total_size={} flags={:?}", self.total_chunks, self.total_size, self.flags);
        } else {
            // Validate that chunk metadata matches our expectations
            if self.total_chunks != chunk.total_chunks ||
               self.total_size != chunk.total_size ||
               self.flags != Some(chunk.flags) {
            
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Inconsistent chunk metadata"
                ));
            }
        }

        // Check if we've already received this chunk
        if self.received_chunks.contains(&chunk.chunk_id) {
            return Ok(self.is_complete());
        }

        // Seek to the position for this chunk
        self.file.seek(SeekFrom::Start(chunk.offset))?;
        
        // Write the chunk data
        self.file.write_all(&chunk.chunk)?;
        
        // Mark this chunk as received
        self.received_chunks.insert(chunk.chunk_id);
        
        // Return whether the file is complete
        Ok(self.is_complete())
    }

    /// Check if the file is complete (all chunks received)
    pub fn is_complete(&self) -> bool {
        self.received_chunks.len() as u64 == self.total_chunks
    }

    /// Get the number of chunks received so far
    pub fn chunks_received(&self) -> usize {
        self.received_chunks.len()
    }

    /// Get the total number of chunks expected
    pub fn total_chunks(&self) -> u64 {
        self.total_chunks
    }

    /// Get the file's complete size
    pub fn total_size(&self) -> u64 {
        self.total_size
    }

    /// Get the file's flags
    pub fn flags(&self) -> Option<u8> {
        self.flags
    }

    /// Get the percentage completion (0.0 to 100.0)
    pub fn completion_percentage(&self) -> f64 {
        if self.total_chunks == 0 {
            return 0.0;
        }
        //println!("received_chunks={} total_chunks={}", self.received_chunks.len(), self.total_chunks);
        (self.received_chunks.len() as f64 / self.total_chunks as f64) * 100.0
    }

    /// Flush any buffered data to disk
    pub fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

pub struct SingleDomain {
    pub domain: String,
    pub score: i16
}

pub struct DomainWatcherModel {
    pub deployment_name: String,
    pub model_id: u8,
    pub domain_scores: Vec<SingleDomain>,
    pub total_domains: u16,
    pub is_dns_ready: bool,
    pub is_dpi_ready: bool,
    pub is_normal_solution_ready: bool,
    pub normal_solution_file: Option<FileChunkManager>,
    pub dpi_model_file: Option<FileChunkManager>,
    pub dns_on_threshold: Option<f64>,
    pub dns_off_threshold: Option<f64>,
    pub dpi_threshold: Option<f64>,
    pub dns_score_scale: Option<f64>,
}

impl DomainWatcherModel {
    pub fn new(model_id: u8, deployment_name: String) -> DomainWatcherModel {
        Self {
            domain_scores: vec![],
            total_domains: 0,
            deployment_name,
            model_id,
            is_dns_ready: false,
            is_dpi_ready: false,
            is_normal_solution_ready: false,
            normal_solution_file: None,
            dpi_model_file: None,
            dns_on_threshold: None,
            dns_off_threshold: None,
            dpi_threshold: None,
            dns_score_scale: None,
        }
    }

    fn to_entities(
        &self,
        table_info: &P4TableInfo,
        model_id: u8
    ) -> Vec<Entity> {
        let mut table_entities = Vec::with_capacity(self.domain_scores.len());
        for single_domain in &self.domain_scores {
            if single_domain.domain == "<PAD>" {
                continue;
            } else if single_domain.domain == "<UNKNOWN>" {
                let table_key = TableKey {
                    fields: vec![
                        KeyField {
                            field_id: table_info.get_key_id_by_name("pipe.Egress.unknown_domain_score", "model_id"),
                            match_type: Some(key_field::MatchType::Exact(Exact {
                                value: vec![model_id]
                            }))
                        }
                    ]
                };
                let table_data = TableData {
                    action_id: table_info.get_action_id_by_name("pipe.Egress.unknown_domain_score", "Egress.set_domain_scores_UNKNOWN"),
                    fields: vec![
                        DataField {
                            field_id: 1,
                            value: Some(data_field::Value::Stream(single_domain.score.to_be_bytes().to_vec())) // <-- 16bit
                        },
                    ]
                };
                let table_entry = TableEntry {
                    table_id: table_info.get_table_by_name("pipe.Egress.unknown_domain_score"),
                    data: Some(table_data),
                    is_default_entry: false,
                    table_read_flag: None,
                    table_mod_inc_flag: None,
                    entry_tgt: None,
                    table_flags: None,
                    value: Some(table_entry::Value::Key(table_key)),
                };
                let entity = Entity {
                    entity: Some(entity::Entity::TableEntry(table_entry))
                };
                table_entities.push(entity);
            } else {
                let hash = domain_crc(&single_domain.domain);
                //info!("domain={} hash={:x}, score={}", single_domain.domain, hash, single_domain.score);
                let table_key = TableKey {
                    fields: vec![
                        KeyField {
                            field_id: table_info.get_key_id_by_name("pipe.Egress.domain_hash2score", "model_id"),
                            match_type: Some(key_field::MatchType::Exact(Exact {
                                value: vec![model_id]
                            }))
                        },
                        KeyField {
                            field_id: table_info.get_key_id_by_name("pipe.Egress.domain_hash2score", "domain_hash"),
                            match_type: Some(key_field::MatchType::Exact(Exact {
                                value: hash.to_be_bytes().to_vec()
                            }))
                        }
                    ]
                };
                let table_data = TableData {
                    action_id: table_info.get_action_id_by_name("pipe.Egress.domain_hash2score", "Egress.set_domain_scores"),
                    fields: vec![
                        DataField {
                            field_id: 1,
                            value: Some(data_field::Value::Stream(single_domain.score.to_be_bytes().to_vec())) // <-- 16bit
                        },
                    ]
                };
                let table_entry = TableEntry {
                    table_id: table_info.get_table_by_name("pipe.Egress.domain_hash2score"),
                    data: Some(table_data),
                    is_default_entry: false,
                    table_read_flag: None,
                    table_mod_inc_flag: None,
                    entry_tgt: None,
                    table_flags: None,
                    value: Some(table_entry::Value::Key(table_key)),
                };
                let entity = Entity {
                    entity: Some(entity::Entity::TableEntry(table_entry))
                };
                table_entities.push(entity);
            }
        }
        table_entities
    }

    pub fn remove_from_dataplane(
        &self,
        table_info: &P4TableInfo,
        model_id: u8
    ) -> Vec<Update> {
        let mut updates = vec![];
        for e in self.to_entities(table_info, model_id) {
            let update = Update {
				r#type: update::Type::Delete as _,
				entity: Some(e)
			};
            updates.push(update);
        }
        updates
    }
    pub fn write_to_dataplane(
        &self,
        table_info: &P4TableInfo,
        model_id: u8
    ) -> Vec<Update> {
        let mut updates = vec![];
        for e in self.to_entities(table_info, model_id) {
            let update = Update {
				r#type: update::Type::Insert as _,
				entity: Some(e)
			};
            updates.push(update);
        }
        updates
    }
}

enum MonitoredPDUSessionState {
    DNS,
    DPI
}

struct MonitoredPDUSession {
    pub model_id: u8,
    pub detection_id: u32,
    pub ue_ip: u32,
    pub seid: u64,
    pub ma_id: u32,
    pub state: MonitoredPDUSessionState,
    pub next_event_ts: u64,
    pub detection_period: u64
}

struct DomainWatcherSessions {
    sessions: HashMap<u32, MonitoredPDUSession>,
    id_gen: IDAllocator<u32>,
}

impl DomainWatcherSessions {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            id_gen: IDAllocator::new_with_counter(1)
        }
    }
}

pub struct DomainWatcherState {
    pub sessions: Arc<RwLock<DomainWatcherSessions>>,
    pub model: Arc<RwLock<Option<DomainWatcherModel>>>,
}

impl DomainWatcherState {
    pub fn new() -> DomainWatcherState {
        Self {
            sessions: Arc::new(RwLock::new(DomainWatcherSessions::new())),
            model: Arc::new(RwLock::new(None))
        }
    }
}

lazy_static! {
	pub static ref DOMAIN_WATCHER_STATES: DomainWatcherState = DomainWatcherState::new();
}


pub async fn add_monitored_ue(ma_id: u32, model_id: u8, seid: u64, ue_ip: u32, cur_ts: u64, monitor_time_window_size: u64, table_info: &P4TableInfo) -> Option<(u32, Vec<Update>)> {
    // auto assign detection_id
    let mut model_lock = DOMAIN_WATCHER_STATES.model.read().await;
    if model_lock.is_none() {
        warn!("model not deployed");
        return None;
    }
    let cur_ts = get_cur_ts();
    let mut state_lock = DOMAIN_WATCHER_STATES.sessions.write().await;
    if !state_lock.sessions.contains_key(&ma_id) {
        let detection_id = state_lock.id_gen.allocate();
        if detection_id.is_err() {
            return None;
        }
        let detection_id = detection_id.unwrap();
        let session_state = MonitoredPDUSession {
            model_id,
            detection_id: detection_id,
            ue_ip,
            ma_id,
            seid,
            state: MonitoredPDUSessionState::DNS,
            next_event_ts: monitor_time_window_size * 1_000_000 + cur_ts,
            detection_period: monitor_time_window_size * 1_000_000
        };
        info!("adding monitored UE [MAID={}] with detection_id={} with ue_ip={}", ma_id, detection_id, std::net::Ipv4Addr::from(ue_ip.to_be_bytes()));
        state_lock.sessions.insert(ma_id, session_state);
        let mut updates = vec![];
        // step 1: add mapping table
        {
            let table_key = TableKey {
                fields: vec![
                    KeyField {
                        field_id: table_info.get_key_id_by_name("pipe.Egress.set_domainwatcher_id", "meta.bridge.ma_id"),
                        match_type: Some(key_field::MatchType::Exact(Exact {
                            value: ma_id.to_be_bytes()[1..].to_vec()
                        }))
                    }
                ]
            };
            let table_data = TableData {
                action_id: table_info.get_action_id_by_name("pipe.Egress.set_domainwatcher_id", "Egress.set_domain_watcher_detection_id"),
                fields: vec![
                    DataField {
                        field_id: 1,
                        value: Some(data_field::Value::Stream(detection_id.to_be_bytes()[1..].to_vec())) // <-- 18bit
                    },
                    DataField {
                        field_id: 2,
                        value: Some(data_field::Value::Stream(model_id.to_be_bytes().to_vec())) // <-- 2bit
                    },
                ]
            };
            let table_entry = TableEntry {
                table_id: table_info.get_table_by_name("pipe.Egress.set_domainwatcher_id"),
                data: Some(table_data),
                is_default_entry: false,
                table_read_flag: None,
                table_mod_inc_flag: None,
                entry_tgt: None,
                table_flags: None,
                value: Some(table_entry::Value::Key(table_key)),
            };
            let entity = Entity {
				entity: Some(entity::Entity::TableEntry(table_entry))
			};
			let update = Update {
				r#type: update::Type::Insert as _,
				entity: Some(entity)
			};
            updates.push(update);
        }
        Some((detection_id, updates))
    } else {
        None
    }
}

pub async fn read_scores(bfrt: Arc<RwLock<Option<bfruntime::BFRuntime>>>) {
    info!("reading scores");
    let mut bfrt_lock_opt = bfrt.write().await;
    let mut bfrt_lock = bfrt_lock_opt.as_mut().unwrap();
    // sync table
    bfrt_lock.sync_registers(
        vec![
            "pipe.Egress.domain_scores_1"
        ]
    ).await;
    let table_info = &bfrt_lock.table_info;
    let mut pull_entries = vec![];
    for detection_id in [0, 1, 2, 3u32] {
        let table_key = TableKey {
            fields: vec![
                KeyField {
                    field_id: table_info.get_key_id_by_name("pipe.Egress.domain_scores_1", "$REGISTER_INDEX"),
                    match_type: Some(key_field::MatchType::Exact(Exact {
                        value: detection_id.to_be_bytes().to_vec()
                    }))
                }
            ]
        };
        let table_entry = TableEntry {
            table_id: table_info.get_table_by_name("pipe.Egress.domain_scores_1"),
            data: None,
            is_default_entry: false,
            table_read_flag: None,
            table_mod_inc_flag: None,
            entry_tgt: None,
            table_flags: None,
            value: Some(table_entry::Value::Key(table_key)),
        };
        pull_entries.push(table_entry);
    }
    let resp_result = bfrt_lock.read_register_i16("pipe.Egress.domain_scores_1", "Egress.domain_scores_1.f1", 10000).await;
    for detection_id in [0, 1, 2, 3u32] {
        println!("detection_id={} score={}", detection_id, resp_result[detection_id as usize]);
    }
}

pub async fn reset_scores(table_info: &P4TableInfo, detection_ids: Vec<u32>) -> Vec<Update> {
    let mut updates = vec![];
    for detection_id in detection_ids {
        let table_key = TableKey {
            fields: vec![
                KeyField {
                    field_id: table_info.get_key_id_by_name("pipe.Egress.domain_scores_1", "$REGISTER_INDEX"),
                    match_type: Some(key_field::MatchType::Exact(Exact {
                        value: detection_id.to_be_bytes().to_vec()
                    }))
                }
            ]
        };
        let table_data = TableData {
            action_id: 0,
            fields: vec![
                DataField {
                    field_id: table_info.get_data_id_by_name("pipe.Egress.domain_scores_1", "Egress.domain_scores_1.f1"),
                    value: Some(data_field::Value::Stream(0i16.to_be_bytes().to_vec())) // <-- 16bit
                },
            ]
        };
        let table_entry = TableEntry {
            table_id: table_info.get_table_by_name("pipe.Egress.domain_scores_1"),
            data: Some(table_data),
            is_default_entry: false,
            table_read_flag: None,
            table_mod_inc_flag: None,
            entry_tgt: None,
            table_flags: None,
            value: Some(table_entry::Value::Key(table_key)),
        };
        let entity = Entity {
            entity: Some(entity::Entity::TableEntry(table_entry))
        };
        let update = Update {
            r#type: update::Type::Modify as _,
            entity: Some(entity)
        };
        updates.push(update);
        }
    updates
}

#[derive(Serialize, Deserialize)]
struct UERequest {
    pub ue_ip: String
}

#[derive(Serialize, Deserialize)]
struct UEResponse {
    pub score: f64,
    pub t_features: f64,
    pub t_inference: f64
}


async fn send_ue_request(ue_ip: u32, endpint: &str) -> Option<UEResponse> {
    let client = reqwest::Client::new();
    let request_body = UERequest {
        ue_ip: format!("{}", std::net::Ipv4Addr::from(ue_ip.to_be_bytes()))
    };
    let endpint = format!("http://{}/{}", INFERENCE_SERVER, endpint);
    let response = client.post(&endpint)
        .json(&request_body)
        .send()
        .await;
    match response {
        Ok(resp) => {
            if resp.status().is_success() {
                //info!("UE request {} sent successfully for IP {}", endpint, request_body.ue_ip);
                // check if response is JSON
                if let Ok(uer) = resp.json::<UEResponse>().await {
                    return Some(uer);
                }
            } else {
                warn!("Failed to send UE request {} for IP {}: {:?}", endpint, request_body.ue_ip, resp.status());
            }
        }
        Err(e) => {
            warn!("Error sending UE request {} for IP {}: {}", endpint, request_body.ue_ip, e);
        }
    };
    None
}

async fn dwd_monitor(bfrt: Arc<RwLock<Option<bfruntime::BFRuntime>>>, ur_tx: Sender<BackendReport>) {
    // get thresholds
    let model_lock = DOMAIN_WATCHER_STATES.model.read().await;
    if model_lock.is_none() {
        warn!("model not deployed");
        return;
    }
    let model = model_lock.as_ref().unwrap();
    if !model.is_dns_ready {
        warn!("model not ready (DNS)");
        return;
    }
    let dns_on_threshold = model.dns_on_threshold.unwrap();
    let dns_off_threshold = model.dns_off_threshold.unwrap();
    let dpi_threshold = model.dpi_threshold.unwrap();
    let dns_score_scale = model.dns_score_scale.unwrap();

    // ((thres_dns_on / (1f64 - thres_dns_on)).ln() * dns_quant_scale).round();
    let dns_on_threshold_i16 = ((dns_on_threshold / (1.0f64 - dns_on_threshold)).ln() * dns_score_scale).round().clamp(-32768.0f64, 32767.0f64) as i16;
    let dns_off_threshold_i16 = ((dns_off_threshold / (1.0f64 - dns_off_threshold)).ln() * dns_score_scale).round().clamp(-32768.0f64, 32767.0f64) as i16;
    

    // read scores
    let detection_id2score = {
        let mut bfrt_lock_opt = bfrt.write().await;
        let bfrt_lock = bfrt_lock_opt.as_mut().unwrap();
        // sync table
        bfrt_lock.sync_registers(
            vec![
                "pipe.Egress.domain_scores_1"
            ]
        ).await;
        bfrt_lock.read_register_i16("pipe.Egress.domain_scores_1", "Egress.domain_scores_1.f1", 6000).await
    };
    println!("detection_id2score={:?}", detection_id2score);
    let mut to_dpi = vec![];
    let mut to_reset_dns = vec![];
    let mut to_remove_dpi = vec![];
    let mut to_report = vec![];
    // list all sessions
    let cur_ts = get_cur_ts();
    {
        let mut state_lock = DOMAIN_WATCHER_STATES.sessions.write().await;
        for (ma_id, session) in state_lock.sessions.iter_mut() {
            if cur_ts >= session.next_event_ts {
                match session.state {
                    MonitoredPDUSessionState::DNS => {
                        let score = detection_id2score[session.detection_id as usize];
                        //info!("UE IP {} score={} in DNS state with on_threshold={} and off_threshold={}", std::net::Ipv4Addr::from(session.ue_ip.to_be_bytes()), score, dns_on_threshold_i16, dns_off_threshold_i16);
                        if score >= dns_on_threshold_i16 {
                            to_report.push((session.seid, session.ue_ip));
                        } else if score >= dns_off_threshold_i16 {
                            session.state = MonitoredPDUSessionState::DPI;
                            to_dpi.push(session.ue_ip);
                        }
                    }
                    MonitoredPDUSessionState::DPI => {
                        // TODO: spawn another task to query DPI prediction and send report.
                        if let Some(resp) = send_ue_request(session.ue_ip, "predict_ue").await {
                            //info!("UE IP {} score={} in DPI state with threshold={}", std::net::Ipv4Addr::from(session.ue_ip.to_be_bytes()), resp.score, dpi_threshold);
                            if resp.score >= dpi_threshold {
                                to_report.push((session.seid, session.ue_ip));
                            }
                        } else {
                            unreachable!()
                        }
                        to_remove_dpi.push(session.ue_ip);
                        session.state = MonitoredPDUSessionState::DNS;
                        //info!("UE IP {} moved to DNS", std::net::Ipv4Addr::from(session.ue_ip.to_be_bytes()));
                    }
                }
                to_reset_dns.push(session.detection_id);
                session.next_event_ts = session.detection_period + cur_ts;
            } else {
                let until = session.next_event_ts - cur_ts;
            }
        }
    }
    for to_dpi_ue_ip in to_dpi.iter() {
        //info!("UE IP {} moved to DPI", std::net::Ipv4Addr::from(to_dpi_ue_ip.to_be_bytes()));
        send_ue_request(*to_dpi_ue_ip, "add_monitored_ue").await;
    }
    {
        let mut bfrt_lock_opt = bfrt.write().await;
        let bfrt_lock = bfrt_lock_opt.as_mut().unwrap();
        let table_info = &bfrt_lock.table_info;
        let mut updates = reset_scores(&table_info, to_reset_dns).await;
        updates.extend(remove_offpath_mirroring(to_remove_dpi, &table_info));
        updates.extend(add_offpath_mirroring(to_dpi, &table_info));
        bfrt_lock.write_update_no_transaction(updates).await.unwrap();
    }
    let mut reports = vec![];
    for (seid, ue_ip) in to_report.iter() {
        //info!("Reporting UE with malware UE_IP={}", std::net::Ipv4Addr::from(ue_ip.to_be_bytes()));
        let mut usage_report_trigger = libpfcp::models::UsageReportTrigger(0);
        usage_report_trigger.setREC5G(1);
        let usage_report = libpfcp::messages::UsageReport {
			urr_id: libpfcp::models::URR_ID { rule_id : u32::MAX },
			ur_seqn: libpfcp::models::UR_SEQN { sqn : 0 },
            usage_report_trigger,
            start_time: None,
            end_time: None,
            volume_measurement: None,
            duration_measurement: None,
            ue_ip_address: None,
            network_instance: None,
            time_of_first_packet: None,
            time_of_last_packet: None,
            usage_information: None,
            predefined_rules_name: None,
        };
        reports.push(BackendReport {
            seid: *seid,
            report: Report::UsageReports(vec![usage_report])
        });
    }
    for report in reports {
        ur_tx.send(report).await.unwrap();
    }
}

fn dwd_monitor_task_thread(bfrt: Arc<RwLock<Option<bfruntime::BFRuntime>>>, ur_tx: Sender<BackendReport>)  {
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().thread_name("dwd_monitor_task_thread").build().unwrap();
    runtime.block_on(async move {
        loop {
            // each dwd_monitor call may take on average >700ms, to avoid contention between new UE registration and domain watcher monitoring (which locks DOMAIN_WATCHER_STATES.sessions)
            // you need to register UEs one by one with a 300ms gap between them
            // and finish registrations of all UEs before the first 30 minutes mark
            // for 5000 UEs it would take 5000*300ms=1500 seconds which is less than 30 minutes
            let start_ts = std::time::Instant::now();
            dwd_monitor(bfrt.clone(), ur_tx.clone()).await;
            //read_scores(bfrt.clone()).await;
            let elapsed = start_ts.elapsed();
            if elapsed < std::time::Duration::from_millis(1000) {
                tokio::time::sleep(std::time::Duration::from_millis(1000) - elapsed).await;
            }
        }
    });
}

pub fn add_offpath_mirroring(ue_ips: Vec<u32>, table_info: &P4TableInfo) -> Vec<Update> {
    let mut updates = vec![];
    for ue_ip in ue_ips {
        let table_key = TableKey {
            fields: vec![
                KeyField {
                    field_id: table_info.get_key_id_by_name("pipe.Ingress.forward_to_offpath_table", "forward_to_offpath_table_match_ip"),
                    match_type: Some(key_field::MatchType::Exact(Exact {
                        value: ue_ip.to_be_bytes().to_vec()
                    }))
                }
            ]
        };
        let table_data = TableData {
            action_id: table_info.get_action_id_by_name("pipe.Ingress.forward_to_offpath_table", "Ingress.forward_to_offpath_table_action"),
            fields: vec![]
        };
        let table_entry = TableEntry {
            table_id: table_info.get_table_by_name("pipe.Ingress.forward_to_offpath_table"),
            data: Some(table_data),
            is_default_entry: false,
            table_read_flag: None,
            table_mod_inc_flag: None,
            entry_tgt: None,
            table_flags: None,
            value: Some(table_entry::Value::Key(table_key)),
        };
        let entity = Entity {
            entity: Some(entity::Entity::TableEntry(table_entry))
        };
        let update = Update {
            r#type: update::Type::Insert as _,
            entity: Some(entity)
        };
        updates.push(update);
    }
    updates
}

pub fn remove_offpath_mirroring(ue_ips: Vec<u32>, table_info: &P4TableInfo) -> Vec<Update> {
    let mut updates = vec![];
    for ue_ip in ue_ips {
        let table_key = TableKey {
            fields: vec![
                KeyField {
                    field_id: table_info.get_key_id_by_name("pipe.Ingress.forward_to_offpath_table", "forward_to_offpath_table_match_ip"),
                    match_type: Some(key_field::MatchType::Exact(Exact {
                        value: ue_ip.to_be_bytes().to_vec()
                    }))
                }
            ]
        };
        let table_entry = TableEntry {
            table_id: table_info.get_table_by_name("pipe.Ingress.forward_to_offpath_table"),
            data: None,
            is_default_entry: false,
            table_read_flag: None,
            table_mod_inc_flag: None,
            entry_tgt: None,
            table_flags: None,
            value: Some(table_entry::Value::Key(table_key)),
        };
        let entity = Entity {
            entity: Some(entity::Entity::TableEntry(table_entry))
        };
        let update = Update {
            r#type: update::Type::Delete as _,
            entity: Some(entity)
        };
        updates.push(update);
    }
    updates
}

pub fn create_dwd_monitor_task(bfrt: Arc<RwLock<Option<bfruntime::BFRuntime>>>, ur_tx: Sender<BackendReport>) {
    info!("creating dwd monitor task");
    std::thread::spawn(move || {
        dwd_monitor_task_thread(bfrt, ur_tx);
    });
}

pub async fn del_monitored_ue(ma_id: u32, table_info: &P4TableInfo) -> Option<(u32, Vec<Update>)> {
    let mut state_lock = DOMAIN_WATCHER_STATES.sessions.write().await;
    if let Some(old_session) = state_lock.sessions.remove(&ma_id) {
        state_lock.id_gen.free(old_session.detection_id);
        let mut updates = vec![];
        // step 1: remove from mapping table
        {
            let table_key = TableKey {
                fields: vec![
                    KeyField {
                        field_id: table_info.get_key_id_by_name("pipe.Egress.set_domainwatcher_id", "meta.bridge.ma_id"),
                        match_type: Some(key_field::MatchType::Exact(Exact {
                            value: ma_id.to_be_bytes()[1..].to_vec()
                        }))
                    }
                ]
            };
            let table_entry = TableEntry {
                table_id: table_info.get_table_by_name("pipe.Egress.set_domainwatcher_id"),
                data: None,
                is_default_entry: false,
                table_read_flag: None,
                table_mod_inc_flag: None,
                entry_tgt: None,
                table_flags: None,
                value: Some(table_entry::Value::Key(table_key)),
            };
            let entity = Entity {
				entity: Some(entity::Entity::TableEntry(table_entry))
			};
			let update = Update {
				r#type: update::Type::Delete as _,
				entity: Some(entity)
			};
            updates.push(update);
        }
        // step 2: set Register at detection_id to 0
        {
            for i in [1] {
                let table_key = TableKey {
                    fields: vec![
                        KeyField {
                            field_id: table_info.get_key_id_by_name(&format!("pipe.Egress.domain_scores_{}", i), "$REGISTER_INDEX"),
                            match_type: Some(key_field::MatchType::Exact(Exact {
                                value: (old_session.detection_id as u32).to_be_bytes().to_vec()
                            }))
                        }
                    ]
                };
                let table_data = TableData {
                    action_id: 0,
                    fields: vec![
                        DataField {
                            field_id: table_info.get_data_id_by_name(&format!("pipe.Egress.domain_scores_{}", i), &format!("Egress.domain_scores_{}.f1", i)),
                            value: Some(data_field::Value::Stream(0u16.to_be_bytes().to_vec())) // <-- 16bit
                        },
                    ]
                };
                let table_entry = TableEntry {
                    table_id: table_info.get_table_by_name(&format!("pipe.Egress.domain_scores_{}", i)),
                    data: Some(table_data),
                    is_default_entry: false,
                    table_read_flag: None,
                    table_mod_inc_flag: None,
                    entry_tgt: None,
                    table_flags: None,
                    value: Some(table_entry::Value::Key(table_key)),
                };
                let entity = Entity {
                    entity: Some(entity::Entity::TableEntry(table_entry))
                };
                let update = Update {
                    r#type: update::Type::Modify as _,
                    entity: Some(entity)
                };
                updates.push(update);
            }
        }
        Some((old_session.detection_id, updates))
    } else {
        None
    }
}
