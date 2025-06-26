#![allow(irrefutable_let_patterns)]

use async_trait::async_trait;

use super::models::PFCPHeader;
use super::ResponseMatchingTuple;
use std::{
	net::{IpAddr, UdpSocket},
	thread::{self, JoinHandle},
};


pub trait SessionRequestHandlers {
	fn handle_session_establishment(
		&self,
		header: PFCPHeader,
		body: Vec<u8>,
		src_ip: IpAddr,
	) -> Vec<u8>;
	fn handle_session_modification(
		&self,
		header: PFCPHeader,
		body: Vec<u8>,
		src_ip: IpAddr,
	) -> Vec<u8>;
	fn handle_session_deletion(&self, header: PFCPHeader, body: Vec<u8>, src_ip: IpAddr)
		-> Vec<u8>;
	fn handle_session_report(&self, header: PFCPHeader, body: Vec<u8>, src_ip: IpAddr) -> Vec<u8>;
}

pub trait NodeRequestHandlers {
	fn handle_heartbeat(&self, header: PFCPHeader, body: Vec<u8>, src_ip: IpAddr) -> Vec<u8>;
	fn handle_pfd_management(&self, header: PFCPHeader, body: Vec<u8>, src_ip: IpAddr) -> Vec<u8>;
	fn handle_association_setup(
		&self,
		header: PFCPHeader,
		body: Vec<u8>,
		src_ip: IpAddr,
	) -> Vec<u8>;
	fn handle_association_update(
		&self,
		header: PFCPHeader,
		body: Vec<u8>,
		src_ip: IpAddr,
	) -> Vec<u8>;
	fn handle_association_release(
		&self,
		header: PFCPHeader,
		body: Vec<u8>,
		src_ip: IpAddr,
	) -> Vec<u8>;
	fn handle_node_report(&self, header: PFCPHeader, body: Vec<u8>, src_ip: IpAddr) -> Vec<u8>;
	fn handle_session_set_deletion(
		&self,
		header: PFCPHeader,
		body: Vec<u8>,
		src_ip: IpAddr,
	) -> Vec<u8>;
	fn handle_model_deployment(
		&self,
		header: PFCPHeader,
		body: Vec<u8>,
		src_ip: IpAddr,
	) -> Vec<u8>;
}

pub fn handler_thread_response<C>(handlers: C, recv_socket: UdpSocket)
where
	C: SessionRequestHandlers + NodeRequestHandlers + Send + Clone,
{
	let mut buf = [0; 65536];
	while let (number_of_bytes, src_addr) = recv_socket
		.recv_from(&mut buf)
		.expect("Didn't receive data")
	{
		//println!("receive {} bytes from {}", number_of_bytes, src_addr);
		let mut content = &buf[..number_of_bytes];
		let mut fo_flag_set = true;
		while content.len() != 0 && fo_flag_set {
			if let Ok((body, next_msg_pointer, header)) = PFCPHeader::decode(content) {
				fo_flag_set = header.flags.getFO() != 0;
				if header.is_request() {
					println!("[!] Request received on response handler thread, proceed to process regardless.");
					unreachable!();
				} else {
					let matching_triplet = ResponseMatchingTuple {
						remote_ip: src_addr.ip(),
						seq: header.seq,
					};
					let mut guard = super::PFCP_NODE_GLOBAL_CONTEXT
						.ongoing_requests
						.write()
						.unwrap();
					{
						match guard.remove(&matching_triplet) {
							Some(shared_state) => {
								let mut shared_state = shared_state.write().unwrap();
								shared_state.response = Some((header, body));
								if let Some(waker) = shared_state.waker.take() {
									waker.wake();
								}
							}
							None => {
								println!(
									"No entry found for incoming response {:?}",
									matching_triplet
								);
							}
						};
					}
				}
				content = next_msg_pointer;
			} else {
				println!("Failed to decode PFCP header, message discarded");
				break;
			}
		}
	}
}

pub fn handler_thread_request<C: 'static>(handlers: C, request_recv_port_override: Option<u16>)
where
	C: SessionRequestHandlers + NodeRequestHandlers + Send + Sync + Clone,
{
	let socket = UdpSocket::bind(&format!(
		"0.0.0.0:{}",
		request_recv_port_override.map_or(8805, |f| f)
	))
	.expect("couldn't bind to address");
	let mut buf = [0; 65536];
	while let (number_of_bytes, src_addr) = socket.recv_from(&mut buf).expect("Didn't receive data")
	{
		//println!("receive {} bytes from {}", number_of_bytes, src_addr);
		let mut content = &buf[..number_of_bytes];
		let mut fo_flag_set = true;
		while content.len() != 0 && fo_flag_set {
			if let Ok((body, next_msg_pointer, header)) = PFCPHeader::decode(content) {
				fo_flag_set = header.flags.getFO() != 0;
				if header.is_request() {
					let mut guard = super::PFCP_NODE_GLOBAL_CONTEXT
						.receive_seq_counters
						.write()
						.unwrap();
					if let Some(seq) = guard.get_mut(&src_addr.ip()) {
						// if header.seq <= *seq && *seq != 0x00_ff_ff_ff {
						// 	println!("Received message whose seq num[{}] <= current seq num[{}], message is considered dupulicated and is discarded", header.seq, *seq);
						// 	break;
						// } else {
						// 	*seq = header.seq;
						// }
						*seq = header.seq;
					} else {
						guard.insert(src_addr.ip(), header.seq);
					}
					let socket2 = socket.try_clone().unwrap();
					// handle received request
					let mut response_header = header.clone();
					let handlers2 = handlers.clone();
					
					std::thread::spawn(move || {
						let mut response_body = match header.msg_type {
							1 => handlers2.handle_heartbeat(header, body, src_addr.ip()),
							3 => handlers2.handle_pfd_management(header, body, src_addr.ip()),
							5 => handlers2.handle_association_setup(header, body, src_addr.ip()),
							7 => handlers2.handle_association_update(header, body, src_addr.ip()),
							9 => handlers2.handle_association_release(header, body, src_addr.ip()),
							12 => handlers2.handle_node_report(header, body, src_addr.ip()),
							200 => handlers2.handle_model_deployment(header, body, src_addr.ip()),
							14 => handlers2.handle_session_set_deletion(header, body, src_addr.ip()),
							50 => handlers2.handle_session_establishment(header, body, src_addr.ip()),
							52 => handlers2.handle_session_modification(header, body, src_addr.ip()),
							54 => handlers2.handle_session_deletion(header, body, src_addr.ip()),
							56 => handlers2.handle_session_report(header, body, src_addr.ip()),
							_ => unreachable!(),
						};
						response_header.msg_type += 1; // request -> response
						let mut length = response_body.len() + 4;
						if response_header.seid.is_some() {
							length += 8;
						}
						assert!(length < 0xffff);
						response_header.length = length as _;
						let mut resp_msg = response_header.encode();
						resp_msg.append(&mut response_body);
						// send response back
						socket2.send_to(resp_msg.as_slice(), src_addr).unwrap();
					});
					
				} else {
					println!("[!] Response received on request handler thread, proceed to process regardless.");
					let matching_triplet = ResponseMatchingTuple {
						remote_ip: src_addr.ip(),
						seq: header.seq,
					};
					let mut guard = super::PFCP_NODE_GLOBAL_CONTEXT
						.ongoing_requests
						.write()
						.unwrap();
					{
						match guard.remove(&matching_triplet) {
							Some(shared_state) => {
								let mut shared_state = shared_state.write().unwrap();
								shared_state.response = Some((header, body));
								if let Some(waker) = shared_state.waker.take() {
									waker.wake();
								}
							}
							None => {
								println!(
									"No entry found for incoming response {:?}",
									matching_triplet
								);
							}
						};
					}
				}
				content = next_msg_pointer;
			} else {
				println!("Failed to decode PFCP header, message discarded");
				break;
			}
		}
	}
}

pub fn create_handler_thread<C: 'static>(
	handlers: C,
	send_request_socket: UdpSocket,
	request_recv_port_override: Option<u16>
) -> Vec<JoinHandle<()>>
where
	C: SessionRequestHandlers + NodeRequestHandlers + Send + Sync + Clone,
{
	let h1 = handlers.clone();
	let handle1 = thread::spawn(move || handler_thread_response(h1, send_request_socket));
	let handle2 =
		thread::spawn(move || handler_thread_request(handlers, request_recv_port_override));
	vec![handle1, handle2]
}
