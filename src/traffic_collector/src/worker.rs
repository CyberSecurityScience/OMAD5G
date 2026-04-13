use std::sync::{Arc, Mutex};
use core_affinity::CoreId;
use crossbeam::channel::unbounded;
use crossbeam::queue::{self, ArrayQueue};

use crate::flow_stats::FlowStatsResult;
use crate::{ue_stats, NormalizeSolution};


pub struct Worker {
    pub id: u32,
    pub ue_stats: dashmap::DashMap<u32, ue_stats::UEStats>,
    pub failed_pkts: u32,
}

fn worker_thread(id: usize, worker: Arc<Mutex<Worker>>, recv: Arc<ArrayQueue<(u32, Vec<u8>, u64, u16, bool, Arc<Option<Vec<NormalizeSolution>>>)>>) {
    core_affinity::set_for_current(CoreId { id: id * 2 + 3 });
    loop {
        if let Some((ue_ip, pkt_ipv4, ts, orig_len, stop, normal_sol)) = recv.pop() {
            if stop {
                break;
            }
            let mut worker = worker.lock().unwrap();
            worker.process_packet(ue_ip, &pkt_ipv4, orig_len, ts, normal_sol);
        }
    }
}

impl Worker {
    pub fn new(id: u32) -> Worker {
        Worker {
            id,
            ue_stats: dashmap::DashMap::new(),
            failed_pkts: 0,
        }
    }

    pub fn process_packet(&mut self, ue_ip: u32, pkt_ipv4: &[u8], orig_len: u16, ts: u64, normal_sol: Arc<Option<Vec<NormalizeSolution>>>) {
        if !self.ue_stats.contains_key(&ue_ip) {
            self.failed_pkts += 1;
            //println!("Worker {} received packet for unknown UE {}, {}", self.id, std::net::Ipv4Addr::from(ue_ip), self.failed_pkts);
            return;
        }
        let mut ue_stats = self.ue_stats.get_mut(&ue_ip).unwrap();
        ue_stats.new_packet(pkt_ipv4, orig_len, ts, normal_sol);
    }

    pub fn add_ue(&self, ue_ip: u32) {
        self.ue_stats.insert(ue_ip, ue_stats::UEStats::new(ue_ip));
    }

    pub fn remove_ue_with_stats(&self, ue_ip: u32, normal_sol: Arc<Option<Vec<NormalizeSolution>>>) -> Option<(u64, Vec<u8>)> {
        let (_, mut ret) = self.ue_stats.remove(&ue_ip)?;
        ret.end_and_encode_stats(normal_sol)
    }

    pub fn remove_ue_with_stats_direct(&self, ue_ip: u32, normal_sol: Arc<Option<Vec<NormalizeSolution>>>) -> Option<(u64, u64, Vec<FlowStatsResult>)> {
        let (_, mut ret) = self.ue_stats.remove(&ue_ip)?;
        ret.end_and_get_stats(normal_sol)
    }
}

pub struct WorkerThreadPool {
    pub worker_handles: Vec<std::thread::JoinHandle<()>>,
    //pub send_channels: Vec<crossbeam::channel::Sender<(u32, Vec<u8>, u64, bool)>>,
    pub queues: Vec<Arc<ArrayQueue<(u32, Vec<u8>, u64, u16, bool, Arc<Option<Vec<NormalizeSolution>>>)>>>,
    pub workers: Vec<Arc<Mutex<Worker>>>,
}

impl WorkerThreadPool {
    pub fn new() -> WorkerThreadPool {
        let num_cpus = 20;
        // start threads
        let mut worker_handles = Vec::with_capacity(num_cpus);
        let mut workers = Vec::with_capacity(num_cpus);
        //let mut send_channels = Vec::with_capacity(num_cpus);
        let mut queues = Vec::with_capacity(num_cpus);
        for i in 0..num_cpus {
            let worker = Worker::new(i as u32);
            let worker = Arc::new(Mutex::new(worker));
            workers.push(worker.clone());
            //let (send, recv) = unbounded();
            let q = Arc::new(ArrayQueue::new(4096));
            let recv = q.clone();
            let handle = std::thread::spawn(move || worker_thread(i + 1, worker, recv));
            //send_channels.push(send);
            queues.push(q);
            worker_handles.push(handle);
        }
        WorkerThreadPool {
            worker_handles,
            //send_channels,
            queues,
            workers,
        }
    }

    pub fn process_packet(&self, ue_ip: u32, pkt_ipv4: &[u8], orig_len: u16, ts: u64, realtime: bool, normal_sol: Arc<Option<Vec<NormalizeSolution>>>) {
        let worker_index = ue_ip as usize % self.workers.len();
        let data = (ue_ip, pkt_ipv4.to_vec(), ts, orig_len, false, normal_sol);
        //self.send_channels[worker_index].send((ue_ip, pkt_ipv4.to_vec(), ts, false)).unwrap();
        if realtime {
            self.queues[worker_index].force_push(data);
        } else {
            loop {
                if let Err(_) = self.queues[worker_index].push(data.clone()) {
                    continue;
                }
                break;
            }
        }
    }

    pub fn is_all_queues_empty(&self) -> bool {
        for q in &self.queues {
            if !q.is_empty() {
                return false;
            }
        }
        true
    }

    pub fn add_ue(&self, ue_ip: u32) {
        let worker_index = ue_ip as usize % self.workers.len();
        self.workers[worker_index].lock().unwrap().add_ue(ue_ip);
    }

    pub fn remove_ue_with_stats(&self, ue_ip: u32, normal_sol: Arc<Option<Vec<NormalizeSolution>>>) -> Option<(u64, Vec<u8>)> {
        let worker_index = ue_ip as usize % self.workers.len();
        self.workers[worker_index].lock().unwrap().remove_ue_with_stats(ue_ip, normal_sol)
    }

    pub fn remove_ue_with_stats_direct(&self, ue_ip: u32, normal_sol: Arc<Option<Vec<NormalizeSolution>>>) -> Option<(u64, u64, Vec<FlowStatsResult>)> {
        let worker_index = ue_ip as usize % self.workers.len();
        self.workers[worker_index].lock().unwrap().remove_ue_with_stats_direct(ue_ip, normal_sol)
    }

    pub fn stop(&self, normal_sol: Arc<Option<Vec<NormalizeSolution>>>) {
        // for send in &self.send_channels {
        //     send.send((0, vec![], 0, true)).unwrap();
        // }
        for q in &self.queues {
            q.force_push((0, vec![], 0, 0, true, normal_sol.clone()));
        }
    }
}

pub struct GlobalThreadPoolContext {
    pub thread_pool: WorkerThreadPool,
}

impl GlobalThreadPoolContext {
    pub fn new() -> GlobalThreadPoolContext {
        GlobalThreadPoolContext {
            thread_pool: WorkerThreadPool::new(),
        }
    }
}

lazy_static! {
	pub static ref GLOBAL_TP_CONTEXT: GlobalThreadPoolContext = GlobalThreadPoolContext::new();
}
