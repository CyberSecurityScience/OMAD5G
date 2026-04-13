use std::net::IpAddr;
use std::hash::{Hash, Hasher};
use std::cmp::Ordering;

/// A 5-tuple representing a network flow.
/// Does not reorder fields; direction-agnostic logic is in the custom impls.
#[derive(Debug, Clone, Copy)]
pub struct Flow5Tuple {
    pub src_ip: u32,
    pub src_port: u16,
    pub dst_ip: u32,
    pub dst_port: u16,
    pub protocol: u8,
}

impl PartialEq for Flow5Tuple {
    fn eq(&self, other: &Self) -> bool {
        // Forward match:
        let forward_match = 
            self.src_ip == other.src_ip
            && self.src_port == other.src_port
            && self.dst_ip == other.dst_ip
            && self.dst_port == other.dst_port
            && self.protocol == other.protocol;

        // Reverse match:
        let reverse_match = 
            self.src_ip == other.dst_ip
            && self.src_port == other.dst_port
            && self.dst_ip == other.src_ip
            && self.dst_port == other.src_port
            && self.protocol == other.protocol;

        forward_match || reverse_match
    }
}

impl Eq for Flow5Tuple {}

impl Hash for Flow5Tuple {
    fn hash<H: Hasher>(&self, state: &mut H) {
        fn flow_to_bytes(ip: &u32, port: u16, ip2: &u32, port2: u16, proto: u8) -> [u8; 13] {
            // 4 bytes for IP, 2 bytes for port, 4 for the other IP, 2 for the other port, 1 for proto = 13 total
            let mut arr = [0u8; 13];

            let ip_bytes = ip.to_be_bytes();
            let ip2_bytes = ip2.to_be_bytes();
            let port_bytes = port.to_be_bytes();
            let port2_bytes = port2.to_be_bytes();
            arr[0..4].copy_from_slice(&ip_bytes);
            arr[4..6].copy_from_slice(&port_bytes);
            arr[6..10].copy_from_slice(&ip2_bytes);
            arr[10..12].copy_from_slice(&port2_bytes);
            arr[12] = proto;

            arr
        }

        let forward = flow_to_bytes(&self.src_ip, self.src_port, &self.dst_ip, self.dst_port, self.protocol);
        let reverse = flow_to_bytes(&self.dst_ip, self.dst_port, &self.src_ip, self.src_port, self.protocol);

        // Choose whichever is lexicographically smaller to ensure
        // the same hash for forward and reverse directions.
        let chosen = match forward.cmp(&reverse) {
            Ordering::Less | Ordering::Equal => forward,
            Ordering::Greater => reverse,
        };

        // Now hash the chosen byte array
        chosen.hash(state);
    }
}
