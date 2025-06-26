use std::{collections::{HashMap, HashSet}, iter::FromIterator, sync::Arc, convert::TryInto};

use domain_watcher::SingleDomain;
use libpfcp::{
	messages::{CreateFAR, CreatePDR, CreateQER, ForwardingParameters, ModelDeploymentRequest, UpdateFAR, UpdatePDR, UpdateQER},
	models::{ApplyAction, MeasurementMethod, ReportingTriggers, VolumeThreshold, VolumeThresholdFlags, FAR_ID, PDR_ID, QER_ID, URR_ID},
};
use log::{info, error, warn};
use tokio::sync::{RwLock, Mutex};

use crate::datapath::{tofino::{bfruntime::bfrt::*, table_templates::PipelineTemplateDirection, action_tables::MatchActionPipeline}, FlattenedURR};

use self::{table_templates::{MatchInstance, match_pipeline}, pfcp_context::PfcpContext, bfruntime::{P4TableInfo, bfrt::key_field::Exact}, usage_reporting::PullingEntryOp};

use super::{BackendInterface, BackendReport, PacketPipeline, FlattenedPacketPipeline, FlattenedQER, OperationPerfStats};
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
mod bfruntime_models;
mod bfruntime;

mod utils;
mod common_tables;
mod routing;
mod pfcp_context;
mod rule_optimization;
mod difference_generator;
mod table_templates;
mod action_tables;
mod match_action_id;
mod usage_reporting;
mod qer;
pub mod qos;

mod domain_watcher;

#[derive(Debug)]
pub enum TofinoBackendError {
	UnsupportedSourceInterface,
	NoForwardingTemplateFound,
	Todo,
	TooManyUpfSelfId,
	TooManyNexthopIPs,
	TeidOutOfRange,
	ConflictingQuotaFAR,
	UnknownQerId,
	InsufficientTableCapacity,
	InsufficientGBRFlowCapacity,
	InsufficientUsageReportingCapacity,
	InsufficientBandwidthCapacity,
	InsufficientActionTableCapacity,
	UnforeseenBfrtError
}
impl std::fmt::Display for TofinoBackendError {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f,"{:?}", *self)
	}
}

impl std::error::Error for TofinoBackendError {
	fn description(&self) -> &str {
		"ERROR TODO"
	}
}

pub struct TofinoBackendSingleThreadContext {
	global_qer_context: qer::GlobalQerContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct URRTestAuxSettings {
	ema_value: f64,
	countdown_ms: i64,
	allow_neg_countdown: bool,
	enable_volume_estimation: bool,
	enable_auto_countdown: bool,
	auto_countdown_update_freq_ms: i64,
	auto_countdown_offset: i64,
	enable_delayed_countdown: bool,
	enter_slow_pull_mode_est_pull_distance: i32,
	enter_slow_pull_mode_rounds: i32,
	slow_pull_mode_delayed_rounds: i32,
	pull_round_time_ms: i64,
	max_update_delay_ms: u64
}

unsafe impl Send for TofinoBackendSingleThreadContext {}
unsafe impl Sync for TofinoBackendSingleThreadContext {}
pub struct TofinoBackend {
	pub bfrt: Arc<RwLock<Option<bfruntime::BFRuntime>>>,
	pfcp_contexts: dashmap::DashMap<u64, PfcpContext>,
	global_contexts: Arc<Mutex<TofinoBackendSingleThreadContext>>,
	global_action_table_context: action_tables::GlobalActionTableContext,
	table_info: P4TableInfo,
	maid_deletion_sender: tokio::sync::mpsc::Sender<u32>,
	enable_deferred_id_del: bool,
	urr_aux: URRTestAuxSettings,
	pull_entry_update_tx: tokio::sync::mpsc::Sender<PullingEntryOp>,
	domain_watcher_android: Option<domain_watcher::DomainWatcherModel>,
	dwd_monitor_period: u64
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct SendEndMarkerRequest {
	pub src_ip: std::net::IpAddr,
	pub dst_ip: std::net::IpAddr,
	pub dst_teid: u32
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct SendBufferedDataRequest {
	pub seid: u64,
	pub pdr_id: u16,
	pub drop: bool
}

impl TofinoBackend {
	async fn next_squence(&mut self) -> u64 {
		0 // TODO: used to ensure table updates are all in order when batching update requests even in the face of out of order TofinoBackend procedure completions
	}
}

#[async_trait]
impl BackendInterface for TofinoBackend {
	async fn new(settings: super::BackendSettings) -> Box<dyn BackendInterface + Sync + Send>
	where
		Self: Sized,
	{
		info!("Backend: TofinoBackend::new");
		// step 1: connect to gRPC
		let mut client_id = 0;
		let mut bfrt = {
			loop {
				match bfruntime::BFRuntime::new(settings.target_addr.clone(), client_id, 0).await {
					Ok(c) => {
						break c;
					}
					Err(e) => {
						warn!("{:?}", e);
					}
				};
				client_id += 1;
				if client_id > 10 {
					panic!("Failed to create BfRuntime after 10 tries");
				}
			}
		};
		if client_id != 0 {
			warn!("Connected to Tofino swich driver using client_id={}", client_id);
		}
		info!("Connected to switch via BFRT");
		// info!("Resetting all tables");
		// bfrt.reset_all_tables().await;
		// step 2: populate common tables
		info!("Populating Tofino tables");
		let mut non_p4_tables = common_tables::populate_cpu_pcie_port(&bfrt.tofino_table_info, settings.cpu_pcie_port);
		bfrt.write_update_no_transaction(non_p4_tables).await.unwrap();
		// step 3: populate UPF table
		let upf_ip = match &settings.upf_ip {
			std::net::IpAddr::V4(x) => *x,
			std::net::IpAddr::V6(_) => panic!("IPv6 not supported!"),
		};
		let upf_ip_updates = common_tables::populate_upf_self_ip(&bfrt.table_info, upf_ip);
		bfrt.write_update_no_transaction(upf_ip_updates).await.unwrap();
		info!("Populating routing tables");
		// step 4: populate routing table
		let routing_updates = routing::populate_routing_table(&bfrt.table_info, &settings.routing);
		bfrt.write_update_no_transaction(routing_updates).await.unwrap();

		info!("Populating DomainWatcher tables");
		let mut domain_watcher_updates = domain_watcher::populate_tld_table("tlds.txt", &bfrt.table_info);
		// let domains = vec![
		// 	SingleDomain {
		// 		domain: "<UNKNOWN>".into(),
		// 		score: 7
		// 	},
		// 	SingleDomain {
		// 		domain: "google.com".into(),
		// 		score: 2
		// 	},
		// 	SingleDomain {
		// 		domain: "google.co.jp".into(),
		// 		score: -5
		// 	},
		// 	SingleDomain {
		// 		domain: "xxax6".into(),
		// 		score: 32
		// 	},
		// 	SingleDomain {
		// 		domain: "xxaxxxaxxxaxxxaxxxaxxxax".into(),
		// 		score: 32
		// 	}
		// ];
		// let mut dw_model_android = domain_watcher::DomainWatcherModel::new(0, "xx".into());
		// dw_model_android.domain_scores = domains;
		// let mut domain_watcher_updates = dw_model_android.write_to_dataplane(&bfrt.table_info, 0);
		// {
		// 	let mut lock = domain_watcher::DOMAIN_WATCHER_STATES.model.write().await;
		// 	lock.replace(dw_model_android);
		// }
		bfrt.write_update_no_transaction(domain_watcher_updates).await.unwrap();

		let mut domain_watcher_updates = vec![];

		domain_watcher_updates.append(&mut common_tables::populate_offpath_detection_egree_mirror(&bfrt.tofino_table_info, settings.offpath_mirrot_port));
		bfrt.write_update_no_transaction(domain_watcher_updates).await.unwrap();


		let table_info = bfrt.table_info.clone();

		let bfrt_arc = Arc::new(RwLock::new(Some(bfrt)));

		domain_watcher::create_dwd_monitor_task(bfrt_arc.clone(), settings.ur_tx.clone());
		
		let (
			action_table,
			initial_maid_updates,
			sender
		) = action_tables::GlobalActionTableContext::new(
			vec![],
			bfrt_arc.clone(),
			&table_info,
			settings.enable_deferred_id_del
		).unwrap();
		let global_ctx = TofinoBackendSingleThreadContext {
			global_qer_context: qer::GlobalQerContext::new()
		};
		{
			let mut bfrt2 = bfrt_arc.write().await;
			let bfrt = bfrt2.as_mut().unwrap();
			bfrt.write_update(initial_maid_updates).await.unwrap();
		}
		info!("Creating usage reporting thread");
		let (pull_entry_update_tx, usage_seid_del_rx) = tokio::sync::mpsc::channel(3000);
		//usage_reporting::create_usage_reporting_threads(bfrt_arc.clone(), settings.ur_tx, sender.clone(), usage_seid_del_rx, false, &settings.est_log_filename, settings.urr_aux.clone());

		info!("TofinoBackend created");
		// step 5: return
		Box::new(TofinoBackend {
			bfrt: bfrt_arc,
			pfcp_contexts: dashmap::DashMap::new(),
			global_contexts: Arc::new(Mutex::new(global_ctx)),
			global_action_table_context: action_table,
			table_info,
			maid_deletion_sender: sender,
			enable_deferred_id_del: settings.enable_deferred_id_del,
			urr_aux: settings.urr_aux,
			pull_entry_update_tx,
			domain_watcher_android: None,//Some(dw_model_android),
			dwd_monitor_period: settings.dwd_monitor_period
		})
	}

	async fn on_to_cpu_packet_received(&mut self, packet: &[u8]) {

	}

	async fn update_or_add_forwarding(&mut self, seid: u64, mut pipelines: Vec<PacketPipeline>) -> Result<OperationPerfStats, Box<dyn std::error::Error + Send + Sync>> {
		//info!("Backend: TofinoBackend::update_or_add_rules [SEID={}]", seid);
		let sqn = self.next_squence().await;

		// mircoseconds since 1 Jan 1970 UTC
		let cur_ts = std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.expect("Time went backwards")
			.as_micros() as u64;

		let perf_timer_1 = std::time::Instant::now();
		if !self.pfcp_contexts.contains_key(&seid) {
			self.pfcp_contexts.insert(seid, PfcpContext::new(seid));
		}
		
		// step 1: Turn pipelines to templates, track which ones require SendEM
		// step 2: figure out which pipelines are updates and which are inserts by tracking know PDR_ID within a given session
		// step 3: find PDRs who require buffered data to be sent
		let mut pfcp_ctx = self.pfcp_contexts.get_mut(&seid).unwrap();

		let mut paired_pipelines = linear_map::LinearMap::with_capacity(pipelines.len());
		for p in pipelines.into_iter() {
			paired_pipelines.insert(p.pdr_id.rule_id, p);
		}

		let to_del = Vec::from_iter(pfcp_ctx.to_del_pdr_ids.iter().cloned());
		
		pfcp_ctx.prepare_update();

 	 	// merge with old rules and update rules PFCP context
		pfcp_ctx.merge_updates(to_del, paired_pipelines)?;
		// reorder MA pipeline
 	 	pfcp_ctx.reorder_ma_rules();
		// find differences
		let (to_del, to_mod, to_add, untouched, untouched_pipelines) = pfcp_ctx.find_difference();
		

		let (
			touched,
			mut table_updates
		) = self.global_action_table_context.update(
			self.bfrt.clone(),
			to_del,
			to_mod,
			to_add,
			seid,
			untouched_pipelines,
			&self.table_info,
			&self.maid_deletion_sender,
			self.enable_deferred_id_del
		).await?;
		let perf_timer_1_ret = perf_timer_1.elapsed().as_micros() as u64;

		pfcp_ctx.activate_pipelines(touched, untouched);


		println!("pfcp_ctx.enable_dwd: {}", pfcp_ctx.enable_dwd);
		let mut updates2 = if pfcp_ctx.enable_dwd {
			if let Some(ul_maid) = pfcp_ctx.get_ul_ma_id() {
				// only uplink direction need monitoring
				if let Some(ue_ip) = pfcp_ctx.get_ue_ip() {
					log::info!("[DWD] Adding monitored UE for uplink MAID: {}, UE_IP: {}", ul_maid, std::net::Ipv4Addr::from(ue_ip.to_be_bytes()));
					let ret = domain_watcher::add_monitored_ue(ul_maid, 0, seid, ue_ip, cur_ts, self.dwd_monitor_period, &self.table_info).await;
					if let Some((_, mut updates)) = ret {
						// let mut updates_mirroring = domain_watcher::add_offpath_mirroring(ue_ip, &self.table_info);
						// updates.append(&mut updates_mirroring);
						updates
					} else {
						vec![]
					}
				} else {
					vec![]
				}
			} else {
				//log::info!("No uplink MAID found, not adding monitored UE");
				vec![]
			}
		} else {
			vec![]
		};


		table_updates.append(&mut updates2);

		pfcp_ctx.to_del_pdr_ids.clear();
		
		let perf_timer_2 = std::time::Instant::now();
		{
			let mut bfrt2 = self.bfrt.write().await;
			let bfrt = bfrt2.as_mut().unwrap();
			bfrt.write_update_reorder(table_updates, sqn).await?;
		}
		let perf_timer_2_ret = perf_timer_2.elapsed().as_micros() as u64;

		Ok(OperationPerfStats { stats1: perf_timer_1_ret, stats2: perf_timer_2_ret, stats3: 0 })
	}

	async fn delete_forwarding(&mut self, seid: u64, pipelines: Vec<PDR_ID>) {
		if pipelines.len() == 0 {
			return;
		}
		//info!("Backend: TofinoBackend::delete_rules [SEID={}] {:?}", seid, pipelines);
		if !self.pfcp_contexts.contains_key(&seid) {
			self.pfcp_contexts.insert(seid, PfcpContext::new(seid));
		}
		let mut pfcp_ctx = self.pfcp_contexts.get_mut(&seid).unwrap();
		// delete_forwarding is always followed by update_or_add_forwarding, we just store it in cache and do the actual deleting in update_or_add_forwarding
		pfcp_ctx.to_del_pdr_ids.extend(pipelines.into_iter().map(|f| f.rule_id));
	}

	async fn delete_session(&mut self, seid: u64) -> (OperationPerfStats, Vec<BackendReport>) {
		//info!("Backend: TofinoBackend::delete_session");
		
		let mut reports = vec![];
		if let Some((_, pfcp_ctx)) = self.pfcp_contexts.remove(&seid) {
			let sqn = self.next_squence().await;

			// step 1: delet reporting
			reports.append(&mut usage_reporting::delete_pfcp_session(self.bfrt.clone(), &self.table_info, &self.maid_deletion_sender, seid, &self.pull_entry_update_tx).await);
			// step 2: delete MAIDs
			let maids = pfcp_ctx
				.pipelines
				.iter()
				.map(|(_, x)| x.maid)
				.filter(|f| f.is_some())
				.map(|f| f.unwrap())
				.collect::<Vec<_>>();


			let t = std::time::Instant::now();
			let mut updates = self.global_action_table_context.remove_maids(maids, &self.table_info, &self.maid_deletion_sender, self.enable_deferred_id_del).await;
			// step 3: delete matching entries
			for (_, p) in pfcp_ctx.pipelines.iter() {
				for m in p.matches.iter() {
					let table_entry = TableEntry {
						table_id: self.table_info.get_table_by_name(m.p4_table_name()),
						data: None,
						is_default_entry: false,
						table_read_flag: None,
						table_mod_inc_flag: None,
						entry_tgt: None,
						table_flags: None,
						value: Some(table_entry::Value::Key(m.generate_table_key(&self.table_info).unwrap())),
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
			}
			// step 4: delete QER_IDs
			let mut global_ctx = self.global_contexts.lock().await;
			let global_qer_ctx = &mut global_ctx.global_qer_context;

			global_qer_ctx.transcation_begin();
			for (_, p) in pfcp_ctx.pipelines.iter() {
				if let Some(global_qer_id) = p.linked_global_qer_id {
					if global_qer_id != 0 && global_qer_id != 1 {
						global_qer_ctx.free_qer_id(&self.table_info, global_qer_id);
					}
				}
			}

			if pfcp_ctx.enable_dwd {
				// remove monitored UE
				if let Some(ul_maid) = pfcp_ctx.get_ul_ma_id() {
					if let Some(ue_ip) = pfcp_ctx.get_ue_ip() {
						log::info!("[DWD] Removing monitored UE for uplink MAID: {}, UE_IP: {}", ul_maid, std::net::Ipv4Addr::from(ue_ip.to_be_bytes()));
						let ret = domain_watcher::del_monitored_ue(ul_maid, &self.table_info).await;
						if let Some((_, mut updates2)) = ret {
							updates.append(&mut updates2);
						}
					}
				}
			}

			updates.append(&mut global_qer_ctx.transcation_commit());
			{
				let mut bfrt2 = self.bfrt.write().await;
				let bfrt = bfrt2.as_mut().unwrap();
				bfrt.write_update_reorder(updates, sqn).await.unwrap();
			}
		}
		//info!("[SEID={}] session deleted", seid);

		(OperationPerfStats { stats1: 0, stats2: 0, stats3: 0 }, reports)
	}

	async fn update_or_add_qer(&mut self, seid: u64, pipelines: Vec<FlattenedQER>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
		if pipelines.len() == 0 {
			return Ok(());
		}

		let sqn = self.next_squence().await;

		if !self.pfcp_contexts.contains_key(&seid) {
			self.pfcp_contexts.insert(seid, PfcpContext::new(seid));
		}
		
		let mut global_ctx = self.global_contexts.lock().await;
		let mut pfcp_ctx = self.pfcp_contexts.get_mut(&seid).unwrap();
		let pfcp_qer_context = &mut pfcp_ctx.qers;
		let global_qer_ctx = &mut global_ctx.global_qer_context;
		global_qer_ctx.transcation_begin();

		for p in pipelines.iter() {
			if let Some(entry) = pfcp_qer_context.get_mut(&p.qer_id) {
				let global_qer_id = entry.0;
				if let Some(new_id) = global_qer_ctx.update_qer(&self.table_info, global_qer_id, *p)? {
					entry.0 = new_id;
				}
			} else {
				if let Some(global_qer_id) = global_qer_ctx.allocate_qer_id(&self.table_info, *p) {
					pfcp_qer_context.insert(p.qer_id, (global_qer_id, *p));
				} else {
					return Err(TofinoBackendError::InsufficientGBRFlowCapacity.into());
				}
			}
		}

		let updates = global_qer_ctx.transcation_commit();
		{
			let mut bfrt2 = self.bfrt.write().await;
			let bfrt = bfrt2.as_mut().unwrap();
			bfrt.write_update_reorder(updates, sqn).await?;
		}
		
		Ok(())
	}

	async fn delete_qer(&mut self, seid: u64, pipelines: Vec<QER_ID>) {
		let sqn = self.next_squence().await;

		let mut global_ctx = self.global_contexts.lock().await;
		let mut pfcp_ctx = self.pfcp_contexts.get_mut(&seid).unwrap();
		let pfcp_qer_context = &mut pfcp_ctx.qers;
		let global_qer_ctx = &mut global_ctx.global_qer_context;
		global_qer_ctx.transcation_begin();

		for p in pipelines.iter() {
			if let Some((global_qer_id, _)) = pfcp_qer_context.remove(&p.rule_id) {
				global_qer_ctx.free_qer_id(&self.table_info, global_qer_id);
			}
		}

		let updates = global_qer_ctx.transcation_commit();
		{
			let mut bfrt2 = self.bfrt.write().await;
			let bfrt = bfrt2.as_mut().unwrap();
			bfrt.write_update_reorder(updates, sqn).await.unwrap();
		}
	}

	async fn update_or_add_urr(&mut self, seid: u64, pipelines: Vec<FlattenedURR>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
		// println!("===update_or_add_urr===");
		// println!("{:#?}", pipelines);
		
		if !self.pfcp_contexts.contains_key(&seid) {
			self.pfcp_contexts.insert(seid, PfcpContext::new(seid));
		}

		usage_reporting::create_pfcp_session(seid, &self.pull_entry_update_tx).await;


		for p in pipelines.into_iter() {
			if p.enable_dwd {
				let mut pfcp_ctx = self.pfcp_contexts.get_mut(&seid).unwrap();
				pfcp_ctx.enable_dwd = true;
			}
			usage_reporting::add_or_update_usage_reporting(seid, p, self.urr_aux.clone(), &self.pull_entry_update_tx).await;
		}

		Ok(())
	}

	async fn delete_urr(&mut self, seid: u64, pipelines: Vec<URR_ID>) -> Vec<Option<BackendReport>> {
		// println!("===delete_urr===");
		// println!("{:#?}", pipelines);

		let mut ret = Vec::with_capacity(pipelines.len());

		for p in pipelines.into_iter() {
			if let Ok(x) = usage_reporting::delete_usage_reporting(seid, p.rule_id, &self.pull_entry_update_tx).await {
				ret.push(x);
			}
		}

		ret
	}

	async fn release_all_sessions(&mut self) {
		info!("Backend: TofinoBackend::release_all_sessions");
	}

	async fn reset_all(&mut self, settings: super::BackendSettings) {
		info!("Backend: TofinoBackend::reset_all");
	}

	async fn stop(&mut self) {
		info!("Backend: TofinoBackend::stop");
		let mut bfrt = self.bfrt.write().await;
		let bfrt2 = bfrt.take();
		drop(bfrt2);
	}

	async fn model_depolyment(&mut self, request: ModelDeploymentRequest) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
		//info!("Backend: TofinoBackend::model_depolyment");
		domain_watcher::deploy_model(request, self.bfrt.clone()).await;
		Ok(())
	}
}
