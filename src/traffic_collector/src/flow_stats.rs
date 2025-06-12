use std::sync::Arc;

use crate::{stat_tracker::StatTracker, NormalizeSolution};

pub const DPI_BYTES: usize = 160;
pub const FLOW_TIMEOUT: u64 = 120_000_000u64; // 120 seconds

pub const TCP_FIN: u8 = 0b0000_0001;
pub const TCP_SYN: u8 = 0b0000_0010;
pub const TCP_RST: u8 = 0b0000_0100;
pub const TCP_PSH: u8 = 0b0000_1000;
pub const TCP_ACK: u8 = 0b0001_0000;
pub const TCP_URG: u8 = 0b0010_0000;
pub const TCP_ECE: u8 = 0b0100_0000;
pub const TCP_CWR: u8 = 0b1000_0000;

pub struct FlowStats {
    pub start_ts: u64,
    pub last_pkt_ts: u64,
    pub last_pkt_fwd_ts: u64,
    pub last_pkt_bwd_ts: u64,

    pub client_ip: u32,
    pub server_ip: u32,
    pub client_port: u16,
    pub server_port: u16,
    pub protocol: u8,
    pub fwd_bytes: u64,
    pub bwd_bytes: u64,
    pub fwd_pkts: u64,
    pub bwd_pkts: u64,

    pub tcp_fin_flag_count_fwd: u32,
    pub tcp_fin_flag_count_bwd: u32,
    pub tcp_syn_flag_count_fwd: u32,
    pub tcp_syn_flag_count_bwd: u32,
    pub tcp_rst_flag_count_fwd: u32,
    pub tcp_rst_flag_count_bwd: u32,
    pub tcp_psh_flag_count_fwd: u32,
    pub tcp_psh_flag_count_bwd: u32,
    pub tcp_ack_flag_count_fwd: u32,
    pub tcp_ack_flag_count_bwd: u32,
    pub tcp_urg_flag_count_fwd: u32,
    pub tcp_urg_flag_count_bwd: u32,
    pub tcp_cwr_flag_count_fwd: u32,
    pub tcp_cwr_flag_count_bwd: u32,
    pub tcp_ece_flag_count_fwd: u32,
    pub tcp_ece_flag_count_bwd: u32,

    pub tcp_init_win_bytes_fwd: u32,
    pub tcp_init_win_bytes_bwd: u32,
    pub tcp_num_pkt_1byte_fwd: u64,
    pub tcp_num_pkt_1byte_bwd: u64,

    pub pkt_len_stats_fwd: StatTracker<u64>,
    pub pkt_len_stats_bwd: StatTracker<u64>,
    pub pkt_len_stats: StatTracker<u64>,

    pub iat_fwd: StatTracker<u64>,
    pub iat_bwd: StatTracker<u64>,
    pub iat: StatTracker<u64>,

    pub dpi_bytes: [u8; DPI_BYTES],
    pub dpi_bytes_len: u16,

    pub fin_count: u32,

    pub dns_query: Option<String>,
}

pub struct FlowStatsResult {
    pub client_ip: u32,
    pub server_ip: u32,
    pub client_port: u16,
    pub server_port: u16,
    pub first_pkt_ts: u64,
    pub flow_dur_sec: f32,
    pub stats_int: [u32; 2],
    pub stats: [f64; 65],
    pub dpi_bytes_len: u16,
    pub dpi_bytes: [u8; DPI_BYTES],
    pub dns_query: Option<String>,
}

impl FlowStatsResult {
    pub fn encode(&self) -> (u64, Vec<u8>) {
        let mut ans = Vec::with_capacity(8 + 65 * 8 + DPI_BYTES + 400);
        if self.dns_query.is_some() {
            ans.push(1);
        } else {
            ans.push(0);
        }
        // encode port and ip
        ans.extend_from_slice(&self.client_ip.to_be_bytes());
        ans.extend_from_slice(&self.client_port.to_be_bytes());
        ans.extend_from_slice(&self.server_ip.to_be_bytes());
        ans.extend_from_slice(&self.server_port.to_be_bytes());
        ans.extend_from_slice(&self.first_pkt_ts.to_be_bytes());
        for i in 0..2 {
            ans.extend_from_slice(&self.stats_int[i].to_be_bytes());
        }
        for i in 0..65 {
            ans.extend_from_slice(&self.stats[i].to_be_bytes());
        }
        ans.extend_from_slice(&self.dpi_bytes_len.to_be_bytes());
        ans.extend_from_slice(&self.dpi_bytes[..self.dpi_bytes_len as usize]);
        if let Some(dns_query) = &self.dns_query {
            let query_len = dns_query.len() as u32;
            ans.extend_from_slice(&query_len.to_be_bytes());
            ans.extend_from_slice(dns_query.as_bytes());
        }
        (self.first_pkt_ts, ans)
    }
}

#[derive(FromPrimitive, ToPrimitive, PartialEq, Eq)]
enum KnownProtocol {
    UNKNOWN,
    HTTP,
    HTTP_81,
    HTTP_8080,
    TLS,
    FTP,
    SSH,
    DNS_OVER_TLS,
    QUIC,
    DNS,
    SMP2P,
    IRC
}


fn identify_known_protocol(
    src_port: u16,
    dst_port: u16,
    protocol: u8
) -> KnownProtocol {
    match (src_port, dst_port, protocol) {
        (_, 80, 6) | (80, _, 6) => KnownProtocol::HTTP,
        (_, 81, 6) | (81, _, 6) => KnownProtocol::HTTP_81,
        (_, 8080, 6) | (8080, _, 6) => KnownProtocol::HTTP_8080,
        (_, 443, 6) | (443, _, 6) => KnownProtocol::TLS,
        (_, 21, 6) | (21, _, 6) => KnownProtocol::FTP,
        (_, 20, 6) | (20, _, 6) => KnownProtocol::FTP,
        (_, 22, 6) | (22, _, 6) => KnownProtocol::SSH,
        (_, 853, 6) | (853, _, 6) => KnownProtocol::DNS_OVER_TLS,
        (_, 443, 17) | (443, _, 17) => KnownProtocol::QUIC,
        (_, 53, 17) | (53, _, 17) => KnownProtocol::DNS,
        // (_, 10001, 6) | (10001, _, 6) => KnownProtocol::SMP2P,
        // (_, 6667, 6) | (6667, _, 6) => KnownProtocol::IRC,
        _ => KnownProtocol::UNKNOWN
    }
}


impl FlowStats {
    pub fn new(client_ip: u32) -> FlowStats {
        FlowStats {
            start_ts: 0,
            last_pkt_ts: 0,
            last_pkt_fwd_ts: 0,
            last_pkt_bwd_ts: 0,

            client_ip: client_ip,
            server_ip: 0,
            client_port: 0,
            server_port: 0,
            protocol: 0,
            fwd_bytes: 0,
            bwd_bytes: 0,
            fwd_pkts: 0,
            bwd_pkts: 0,

            tcp_fin_flag_count_fwd: 0,
            tcp_fin_flag_count_bwd: 0,
            tcp_syn_flag_count_fwd: 0,
            tcp_syn_flag_count_bwd: 0,
            tcp_rst_flag_count_fwd: 0,
            tcp_rst_flag_count_bwd: 0,
            tcp_psh_flag_count_fwd: 0,
            tcp_psh_flag_count_bwd: 0,
            tcp_ack_flag_count_fwd: 0,
            tcp_ack_flag_count_bwd: 0,
            tcp_urg_flag_count_fwd: 0,
            tcp_urg_flag_count_bwd: 0,
            tcp_cwr_flag_count_fwd: 0,
            tcp_cwr_flag_count_bwd: 0,
            tcp_ece_flag_count_fwd: 0,
            tcp_ece_flag_count_bwd: 0,

            tcp_init_win_bytes_fwd: 0,
            tcp_init_win_bytes_bwd: 0,
            tcp_num_pkt_1byte_fwd: 0,
            tcp_num_pkt_1byte_bwd: 0,

            pkt_len_stats_fwd: StatTracker::new(),
            pkt_len_stats_bwd: StatTracker::new(),
            pkt_len_stats: StatTracker::new(),

            iat_fwd: StatTracker::new(),
            iat_bwd: StatTracker::new(),
            iat: StatTracker::new(),

            dpi_bytes: [0; 160],
            dpi_bytes_len: 0,

            fin_count: 0,

            dns_query: None,
        }
    }

    pub fn reset(&mut self, client_ip: u32) {
        self.start_ts = 0;
        self.last_pkt_ts = 0;

        self.client_ip = client_ip;
        self.server_ip = 0;
        self.client_port = 0;
        self.server_port = 0;
        self.protocol = 0;
        self.fwd_bytes = 0;
        self.bwd_bytes = 0;
        self.fwd_pkts = 0;
        self.bwd_pkts = 0;

        self.tcp_fin_flag_count_fwd = 0;
        self.tcp_fin_flag_count_bwd = 0;
        self.tcp_syn_flag_count_fwd = 0;
        self.tcp_syn_flag_count_bwd = 0;
        self.tcp_rst_flag_count_fwd = 0;
        self.tcp_rst_flag_count_bwd = 0;
        self.tcp_psh_flag_count_fwd = 0;
        self.tcp_psh_flag_count_bwd = 0;
        self.tcp_ack_flag_count_fwd = 0;
        self.tcp_ack_flag_count_bwd = 0;
        self.tcp_urg_flag_count_fwd = 0;
        self.tcp_urg_flag_count_bwd = 0;
        self.tcp_cwr_flag_count_fwd = 0;
        self.tcp_cwr_flag_count_bwd = 0;
        self.tcp_ece_flag_count_fwd = 0;
        self.tcp_ece_flag_count_bwd = 0;

        self.tcp_init_win_bytes_fwd = 0;
        self.tcp_init_win_bytes_bwd = 0;

        self.pkt_len_stats_fwd.reset();
        self.pkt_len_stats_bwd.reset();
        self.pkt_len_stats.reset();

        self.iat_fwd.reset();
        self.iat_bwd.reset();
        self.iat.reset();
    }

    pub fn new_packet(&mut self, pkt_ipv4: &[u8], orig_len: u16, ts: u64) -> bool {
        // Parse the packet

        let is_first_pkt = self.start_ts == 0;
        let iat_diff = if is_first_pkt {
            self.start_ts = ts;
            0
        } else {
            let diff = ts - self.last_pkt_ts;
            diff
        };
        self.last_pkt_ts = ts;

        let src_ip = u32::from_be_bytes([pkt_ipv4[12], pkt_ipv4[13], pkt_ipv4[14], pkt_ipv4[15]]);
        let dst_ip = u32::from_be_bytes([pkt_ipv4[16], pkt_ipv4[17], pkt_ipv4[18], pkt_ipv4[19]]);

        let mut tcp_disconnect = false;

        let is_fwd = src_ip == self.client_ip; // Forward(Uplink) direction
        if is_first_pkt {
            self.server_ip = if is_fwd { dst_ip } else { src_ip };
        }
        let (iat_fwd_diff, iat_bwd_diff) = if is_fwd {
            let diff = if self.last_pkt_fwd_ts != 0 { ts - self.last_pkt_fwd_ts } else { 0 };
            self.last_pkt_fwd_ts = ts;
            (diff, 0)
        } else {
            let diff = if self.last_pkt_bwd_ts != 0 { ts - self.last_pkt_bwd_ts } else { 0 };
            self.last_pkt_bwd_ts = ts;
            (0, diff)
        };
        let protocol = pkt_ipv4[9];
        if protocol == 17 || protocol == 6 {
            self.protocol = protocol;
            // Parse the UDP or TCP packet
            if orig_len < 20 + 8 {
                // Ignore the packet
                return false;
            }

            let dpi_remaining = DPI_BYTES - self.dpi_bytes_len as usize;
            if dpi_remaining > 0 {
                let pkt_len = orig_len as usize;
                let copy_len = std::cmp::min(dpi_remaining, pkt_len);
                self.dpi_bytes[self.dpi_bytes_len as usize..self.dpi_bytes_len as usize + copy_len].copy_from_slice(&pkt_ipv4[0..copy_len]);
                // mask IP address and port
                let mask_start = std::cmp::min(self.dpi_bytes_len as usize + 12, DPI_BYTES - 12);
                let mask_end = std::cmp::min(self.dpi_bytes_len as usize + 24, DPI_BYTES);
                let empty_slice = [0u8; 12];
                self.dpi_bytes[mask_start..mask_end].copy_from_slice(&empty_slice[0..mask_end - mask_start]);
                self.dpi_bytes_len += copy_len as u16;
            }

            let src_port = u16::from_be_bytes([pkt_ipv4[20], pkt_ipv4[21]]);
            let dst_port = u16::from_be_bytes([pkt_ipv4[22], pkt_ipv4[23]]);
            if is_first_pkt {
                self.protocol = protocol;
                self.client_port = if is_fwd { src_port } else { dst_port };
                self.server_port = if is_fwd { dst_port } else { src_port };
            }
            if iat_diff != 0 {
                self.iat.insert(iat_diff);
            }
            if is_fwd {
                if iat_fwd_diff != 0 {
                    self.iat_fwd.insert(iat_fwd_diff);
                }
            } else {
                if iat_bwd_diff != 0 {
                    self.iat_bwd.insert(iat_bwd_diff);
                }
            }
            self.pkt_len_stats.insert(orig_len as u64);
            if is_fwd {
                self.pkt_len_stats_fwd.insert(orig_len as u64);
            } else {
                self.pkt_len_stats_bwd.insert(orig_len as u64);
            }
            if is_fwd {
                self.fwd_bytes += orig_len as u64;
                self.fwd_pkts += 1;
            } else {
                self.bwd_bytes += orig_len as u64;
                self.bwd_pkts += 1;
            }
            if protocol == 6 {
                if orig_len < 20 + 20 {
                    // Ignore the packet
                    return false;
                }
                // Parse the TCP packet
                let tcp_flags = pkt_ipv4[20 + 13];
                if tcp_flags & TCP_FIN != 0 {
                    self.fin_count += 1;
                    if self.fin_count >= 2 {
                        tcp_disconnect = true;
                    }
                }
                if tcp_flags & TCP_RST != 0 {
                    tcp_disconnect = true;
                }
                if is_fwd {
                    if tcp_flags & TCP_FIN != 0 {
                        self.tcp_fin_flag_count_fwd += 1;
                    }
                    if tcp_flags & TCP_SYN != 0 {
                        self.tcp_syn_flag_count_fwd += 1;
                    }
                    if tcp_flags & TCP_RST != 0 {
                        self.tcp_rst_flag_count_fwd += 1;
                    }
                    if tcp_flags & TCP_PSH != 0 {
                        self.tcp_psh_flag_count_fwd += 1;
                    }
                    if tcp_flags & TCP_ACK != 0 {
                        self.tcp_ack_flag_count_fwd += 1;
                    }
                    if tcp_flags & TCP_URG != 0 {
                        self.tcp_urg_flag_count_fwd += 1;
                    }
                    if tcp_flags & TCP_CWR != 0 {
                        self.tcp_cwr_flag_count_fwd += 1;
                    }
                    if tcp_flags & TCP_ECE != 0 {
                        self.tcp_ece_flag_count_fwd += 1;
                    }
                } else {
                    if tcp_flags & TCP_FIN != 0 {
                        self.tcp_fin_flag_count_bwd += 1;
                    }
                    if tcp_flags & TCP_SYN != 0 {
                        self.tcp_syn_flag_count_bwd += 1;
                    }
                    if tcp_flags & TCP_RST != 0 {
                        self.tcp_rst_flag_count_bwd += 1;
                    }
                    if tcp_flags & TCP_PSH != 0 {
                        self.tcp_psh_flag_count_bwd += 1;
                    }
                    if tcp_flags & TCP_ACK != 0 {
                        self.tcp_ack_flag_count_bwd += 1;
                    }
                    if tcp_flags & TCP_URG != 0 {
                        self.tcp_urg_flag_count_bwd += 1;
                    }
                    if tcp_flags & TCP_CWR != 0 {
                        self.tcp_cwr_flag_count_bwd += 1;
                    }
                    if tcp_flags & TCP_ECE != 0 {
                        self.tcp_ece_flag_count_bwd += 1;
                    }
                }
                let tcp_content_len = orig_len - 20 - 20;
                if tcp_content_len != 0 {
                    if is_fwd {
                        self.tcp_num_pkt_1byte_fwd += 1;
                    } else {
                        self.tcp_num_pkt_1byte_bwd += 1;
                    }
                }
            }
        } else {
            // Ignore the packet
            self.protocol = protocol;
            return true;
        }
        if self.server_port == 53 && self.fwd_pkts == 1 && self.protocol == 17 {
            // fisrt DNS packet
            let dns_bytes = &pkt_ipv4[28..];
            if dns_bytes.len() > 12 {
                let n_query = u16::from_be_bytes(dns_bytes[4..6].try_into().unwrap());
                //println!("n_query: {}", n_query);
                if n_query == 1 {
                    let mut domain_start = &dns_bytes[12..];
                    //println!("domain_start: {}", domain_start[0]);
                    let mut seglen = domain_start[0];
                    let mut domain = "".to_string();
                    while domain_start.len() != 0 && seglen != 0 && seglen <= 31 {
                        domain.push_str(String::from_utf8(domain_start[1..(seglen + 1) as usize].to_vec()).unwrap().as_str());
                        domain_start = &domain_start[(seglen + 1) as usize..];
                        seglen = domain_start[0];
                        if seglen != 0 && seglen <= 31 {
                            domain.push_str(".");
                        }
                    }
                    if domain.len() != 0 {
                        let parts: Vec<&str> = domain.split('.').collect::<Vec<_>>();
                        let parts = if parts.len() >= 4 {
                            parts[..4].to_vec()
                        } else {
                            parts
                        };
                        let mut domain = "".to_string();
                        if parts.len() <= 3 {
                            domain = parts.join(".");
                        } else {
                            domain = parts[parts.len() - 3..].join(".");
                        }
                        //println!("q: {}", domain);
                        if domain.len() != 0 {
                            self.dns_query = Some(domain);
                        }
                    }
                }
            }
        }
        if self.server_port == 53 && self.bwd_pkts == 1 && self.protocol == 17 {
            tcp_disconnect = true;
        }
        tcp_disconnect
    }

    pub fn is_flow_expired(&self, ts: u64) -> bool {
        ts - self.last_pkt_ts > FLOW_TIMEOUT
    }

    pub fn get_result(&self, normal_sol: Arc<Option<Vec<NormalizeSolution>>>) -> FlowStatsResult {
        let flow_duration_sec = ((self.last_pkt_ts - self.start_ts) as f64) / 1_000_000.0;
        let mut ret = FlowStatsResult {
            client_ip: self.client_ip,
            server_ip: self.server_ip,
            client_port: self.client_port,
            server_port: self.server_port,
            first_pkt_ts: self.start_ts,
            flow_dur_sec: flow_duration_sec as f32,
            stats_int: [
                self.protocol as u32,
                identify_known_protocol(self.client_port, self.server_port, self.protocol) as u32
            ],
            stats: [
                flow_duration_sec,
                (self.fwd_bytes + self.bwd_bytes) as f64,
                (self.fwd_pkts + self.bwd_pkts) as f64,
                self.fwd_pkts as f64,
                self.bwd_pkts as f64,
                self.fwd_bytes as f64,
                self.bwd_bytes as f64,
                if self.bwd_pkts == 0 { 0.0 } else { self.fwd_pkts as f64 / self.bwd_pkts as f64 },
                if self.bwd_bytes == 0 { 0.0 } else { self.fwd_bytes as f64 / self.bwd_bytes as f64 },
                self.fwd_bytes as f64 / flow_duration_sec,
                self.fwd_pkts as f64 / flow_duration_sec,
                self.bwd_bytes as f64 / flow_duration_sec,
                self.bwd_pkts as f64 / flow_duration_sec,
                (self.fwd_bytes + self.bwd_bytes) as f64 / flow_duration_sec,
                (self.fwd_pkts + self.bwd_pkts) as f64 / flow_duration_sec,
                self.tcp_fin_flag_count_fwd as f64,
                self.tcp_syn_flag_count_fwd as f64,
                self.tcp_rst_flag_count_fwd as f64,
                self.tcp_psh_flag_count_fwd as f64,
                self.tcp_ack_flag_count_fwd as f64,
                self.tcp_urg_flag_count_fwd as f64,
                self.tcp_ece_flag_count_fwd as f64,
                self.tcp_cwr_flag_count_fwd as f64,
                self.tcp_fin_flag_count_bwd as f64,
                self.tcp_syn_flag_count_bwd as f64,
                self.tcp_rst_flag_count_bwd as f64,
                self.tcp_psh_flag_count_bwd as f64,
                self.tcp_ack_flag_count_bwd as f64,
                self.tcp_urg_flag_count_bwd as f64,
                self.tcp_ece_flag_count_bwd as f64,
                self.tcp_cwr_flag_count_bwd as f64,
                (self.tcp_fin_flag_count_fwd + self.tcp_fin_flag_count_bwd) as f64,
                (self.tcp_syn_flag_count_fwd + self.tcp_syn_flag_count_bwd) as f64,
                (self.tcp_rst_flag_count_fwd + self.tcp_rst_flag_count_bwd) as f64,
                (self.tcp_psh_flag_count_fwd + self.tcp_psh_flag_count_bwd) as f64,
                (self.tcp_ack_flag_count_fwd + self.tcp_ack_flag_count_bwd) as f64,
                (self.tcp_urg_flag_count_fwd + self.tcp_urg_flag_count_bwd) as f64,
                (self.tcp_ece_flag_count_fwd + self.tcp_ece_flag_count_bwd) as f64,
                (self.tcp_cwr_flag_count_fwd + self.tcp_cwr_flag_count_bwd) as f64,
                self.tcp_init_win_bytes_fwd as f64,
                self.tcp_init_win_bytes_bwd as f64,
                self.tcp_num_pkt_1byte_fwd as f64,
                self.tcp_num_pkt_1byte_bwd as f64,
                if self.fwd_pkts == 0 { 0.0 } else { self.tcp_num_pkt_1byte_fwd as f64 / self.fwd_pkts as f64 },
                if self.bwd_pkts == 0 { 0.0 } else { self.tcp_num_pkt_1byte_bwd as f64 / self.bwd_pkts as f64 },
                (self.tcp_num_pkt_1byte_fwd + self.tcp_num_pkt_1byte_bwd) as f64,
                (self.tcp_num_pkt_1byte_fwd + self.tcp_num_pkt_1byte_bwd) as f64 / (self.fwd_pkts + self.bwd_pkts) as f64,
                self.pkt_len_stats_fwd.min() as f64,
                self.pkt_len_stats_fwd.max() as f64,
                self.pkt_len_stats_fwd.mean(),
                self.pkt_len_stats_bwd.min() as f64,
                self.pkt_len_stats_bwd.max() as f64,
                self.pkt_len_stats_bwd.mean(),
                self.pkt_len_stats.min() as f64,
                self.pkt_len_stats.max() as f64,
                self.pkt_len_stats.mean(),
                self.iat_fwd.min() as f64,
                self.iat_fwd.max() as f64,
                self.iat_fwd.mean(),
                self.iat_bwd.min() as f64,
                self.iat_bwd.max() as f64,
                self.iat_bwd.mean(),
                self.iat.min() as f64,
                self.iat.max() as f64,
                self.iat.mean(),
            ],
            dpi_bytes_len: self.dpi_bytes_len,
            dpi_bytes: self.dpi_bytes,
            dns_query: self.dns_query.clone(),
        };
        let normal_sol = normal_sol.as_ref().as_ref();
        if let Some(normal_sol) = normal_sol {
            let mut stats_int_normalized = [0u32; 2];
            let mut stats_normalized = [0.0; 65];
            for i in 0..2 {
                stats_int_normalized[i] = normal_sol[i].apply_int(ret.stats_int[i]);
            }
            for i in 0..65 {
                stats_normalized[i] = normal_sol[i + 2].apply_f64(ret.stats[i]);
            }
            ret.stats_int = stats_int_normalized;
            ret.stats = stats_normalized;
        }
        return ret;
    }

    pub fn is_dns(&self) -> bool {
        let known_protocol = identify_known_protocol(self.client_port, self.server_port, self.protocol);
        known_protocol == KnownProtocol::DNS
    }
}
