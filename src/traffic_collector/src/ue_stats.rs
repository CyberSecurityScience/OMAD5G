
use std::{collections::HashMap, sync::Arc, u64};

use crate::{flow5tuple::Flow5Tuple, flow_stats::{self, FlowStats, FlowStatsResult}, NormalizeSolution};

pub struct UEStats {
    pub ue_ip: u32,
    pub flow_stats: HashMap<Flow5Tuple, FlowStats>,
    pub num_pkt: u64,
    pub num_bytes: u64,
    pub dns_queries: Vec<String>,
    pub finished_flow_stats: Vec<FlowStatsResult>,
}

impl UEStats {
    pub fn new(ue_ip: u32) -> UEStats {
        UEStats {
            ue_ip: ue_ip,
            flow_stats: HashMap::with_capacity(4000),
            num_pkt: 0,
            num_bytes: 0,
            dns_queries: Vec::with_capacity(500),
            finished_flow_stats: Vec::with_capacity(4000),
        }
    }

    pub fn reset(&mut self, ue_ip: u32) {
        self.ue_ip = ue_ip;
        self.flow_stats.clear();
        self.num_pkt = 0;
        self.num_bytes = 0;
    }

    pub fn new_packet(&mut self, pkt_ipv4: &[u8], orig_len: u16, ts: u64, normal_sol: Arc<Option<Vec<NormalizeSolution>>>) {
        self.num_pkt += 1;
        self.num_bytes += orig_len as u64;
        if orig_len < 20 + 8 {
            return;
        }
        let src_ip = u32::from_be_bytes([pkt_ipv4[12], pkt_ipv4[13], pkt_ipv4[14], pkt_ipv4[15]]);
        let dst_ip = u32::from_be_bytes([pkt_ipv4[16], pkt_ipv4[17], pkt_ipv4[18], pkt_ipv4[19]]);
        let protocol = pkt_ipv4[9];
        if protocol != 17 && protocol != 6 {
            return;
        }
        if protocol == 6 && orig_len < 20 + 20 {
            return;
        }
        let is_tcp_syn = protocol == 6 && pkt_ipv4[20 + 13] & flow_stats::TCP_SYN != 0;
        let src_port = u16::from_be_bytes([pkt_ipv4[20], pkt_ipv4[21]]);
        let dst_port = u16::from_be_bytes([pkt_ipv4[22], pkt_ipv4[23]]);
        let flow = Flow5Tuple {
            src_ip: src_ip,
            src_port: src_port,
            dst_ip: dst_ip,
            dst_port: dst_port,
            protocol: protocol,
        };
        let mut to_del_flow_key = None;
        let mut to_add_flow_key = None;
        if let Some(existing_flow) = self.flow_stats.get_mut(&flow) {
            let is_expired = existing_flow.is_flow_expired(ts);
            if is_expired {
                let flow_result = existing_flow.get_result(normal_sol);
                self.finished_flow_stats.push(flow_result);
                if let Some(dns_query) = existing_flow.dns_query.clone() {
                    self.dns_queries.push(dns_query);
                }
                to_del_flow_key = Some(flow);
                if protocol == 17 || is_tcp_syn {
                    to_add_flow_key = Some(flow);
                }
            } else {
                let flow_end = existing_flow.new_packet(pkt_ipv4, orig_len, ts);
                if flow_end {
                    let flow_result = existing_flow.get_result(normal_sol);
                    self.finished_flow_stats.push(flow_result);
                    if let Some(dns_query) = existing_flow.dns_query.clone() {
                        self.dns_queries.push(dns_query);
                    }
                    to_del_flow_key = Some(flow);
                }
            }
        } else {
            if protocol == 17 || is_tcp_syn {
                to_add_flow_key = Some(flow);
            }
        }
        if let Some(flow_key) = to_del_flow_key {
            self.flow_stats.remove(&flow_key);
        }
        if let Some(flow_key) = to_add_flow_key {
            let mut flow_stats = FlowStats::new(self.ue_ip);
            flow_stats.new_packet(pkt_ipv4, orig_len, ts);
            self.flow_stats.insert(flow_key, flow_stats);
        }
    }

    pub fn flush_unfinished_flows(&mut self, normal_sol: Arc<Option<Vec<NormalizeSolution>>>) {
        for (flow_key, flow_stats) in &mut self.flow_stats {
            let flow_result = flow_stats.get_result(normal_sol.clone());
            self.finished_flow_stats.push(flow_result);
            if let Some(dns_query) = flow_stats.dns_query.clone() {
                self.dns_queries.push(dns_query);
            }
        }
        self.flow_stats.clear();
    }

    pub fn end_and_encode_stats(&mut self, normal_sol: Arc<Option<Vec<NormalizeSolution>>>) -> Option<(u64, Vec<u8>)> {
        self.flush_unfinished_flows(normal_sol);
        if self.finished_flow_stats.is_empty() {
            return None;
        }
        let mut ret = Vec::with_capacity(self.finished_flow_stats.len() * 300 * 8);
        let flow_count = self.finished_flow_stats.len() as u32;
        ret.extend_from_slice(&flow_count.to_be_bytes());
        let mut first_flow_ts = u64::MAX;
        for flow_result in &self.finished_flow_stats {
            let (ts, flow_result_bytes) = flow_result.encode();
            first_flow_ts = first_flow_ts.min(ts);
            ret.extend_from_slice(&flow_result_bytes);
        }
        Some((first_flow_ts, ret))
    }

    pub fn end_and_get_stats(&mut self, normal_sol: Arc<Option<Vec<NormalizeSolution>>>) -> Option<(u64, u64, Vec<FlowStatsResult>)> {
        self.flush_unfinished_flows(normal_sol);
        if self.finished_flow_stats.is_empty() {
            return None;
        }
        let mut first_flow_ts = u64::MAX;
        let mut dpi_bytes = 0;
        for flow_result in &self.finished_flow_stats {
            dpi_bytes += (flow_result.dpi_bytes_len as u64) + 2;
            first_flow_ts = first_flow_ts.min(flow_result.first_pkt_ts);
        }
        Some((first_flow_ts, dpi_bytes, std::mem::take(&mut self.finished_flow_stats)))
    }
}
