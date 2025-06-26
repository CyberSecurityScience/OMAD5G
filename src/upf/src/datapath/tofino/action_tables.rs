use std::{collections::{HashSet, HashMap}, sync::Arc, iter::FromIterator, thread::JoinHandle};

use itertools::Itertools;
use linear_map::{LinearMap, set::LinearSet};
use log::info;
use models::QosRequirement;
use tokio::sync::{RwLock, mpsc::{Receiver, Sender}};

use crate::datapath::tofino::{bfruntime::bfrt::{TableEntry, table_entry, Entity, Update, entity, update, key_field::Exact}, usage_reporting::PdrUsageReportingAssociation};

use super::{table_templates::{MatchInstance, PipelineTemplateDirection}, match_action_id::{ActionInstance, MaidTree, ActionTableOperation, ActionInstanceCount}, bfruntime::{P4TableInfo, bfrt::{TableKey, KeyField, key_field::{self, Ternary}, TableData, DataField, data_field}, BFRuntime}, pfcp_context::{pdr_id_t, TofinoPdrPipeline}, difference_generator::{ToDelPipeline, ToModPipeline, ToAddPipeline}};

#[derive(Debug)]
pub enum ActionTableError {
    CannotBreakRangeAnymore,
    TreeTooDeep,
    InsufficientActionTableCapacity,
    InsufficientMAID,
}

impl std::fmt::Display for ActionTableError {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f,"{:?}", *self)
	}
}

impl std::error::Error for ActionTableError {
	fn description(&self) -> &str {
		"ERROR TODO"
	}
}

#[derive(Debug, Clone)]
pub struct MatchActionPipeline {
    pub priority: i32,
    pub pdr_id: u16,
    pub match_instance: MatchInstance,
    pub action_instance: ActionInstance,
    pub urr_ids: Vec<u32>,
    pub linked_qer_id: Option<u16>,
    pub match_action_id: Option<u32>,
}

impl MatchActionPipeline {
    pub fn is_active(&self) -> bool {
        self.match_action_id.is_some()
    }
}

pub struct GlobalActionTableContext {
    maid_tree: Arc<RwLock<MaidTree>>,

    thread: Option<JoinHandle<()>>
}

impl GlobalActionTableContext {
    async fn remove_ids(ids: &Vec<u32>, maid_tree: Arc<RwLock<MaidTree>>, bfrt: Arc<RwLock<Option<BFRuntime>>>) {
        // info!("freeing MAIDS: {:?}", ids);
        let mut tree = maid_tree.write().await;
        let mut bfrt2 = bfrt.write().await;
        let bfrt = bfrt2.as_mut().unwrap();
        tree.action_tables_transcation_begin();
        for id in ids {
            tree.free_maid(*id as _);
        }
        let mut updates = GlobalActionTableContext::transcation_commit(&mut tree, &bfrt.table_info, vec![]).unwrap();
        updates.reserve(ids.len());
        for id in ids {
            let egress_table_entry = TableKey {
				fields: vec![
					KeyField {
						field_id: bfrt.table_info.get_key_id_by_name("pipe.Egress.accounting.accounting_exact", "ma_id"),
						match_type: Some(key_field::MatchType::Exact(Exact {
							value: id.to_be_bytes()[1..].to_vec()
						}))
					}
				]
			};
			let table_entry = TableEntry {
				table_id: bfrt.table_info.get_table_by_name("pipe.Egress.accounting.accounting_exact"),
				data: None,//Some(table_data),
				is_default_entry: false,
				table_read_flag: None,
				table_mod_inc_flag: None,
				entry_tgt: None,
				table_flags: None,
				value: Some(table_entry::Value::Key(egress_table_entry)),
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
        bfrt.write_update(updates).await.unwrap();
    }
    fn maid_deletion_thread(mut receiver: Receiver<u32>, maid_tree: Arc<RwLock<MaidTree>>, bfrt: Arc<RwLock<Option<BFRuntime>>>) {
        info!("maid_deletion_thread started");
        let runtime = tokio::runtime::Builder::new_current_thread().thread_name("Tofino UPF global_PDR_ID deletion thread").enable_time().build().unwrap();
        let start_timestamp = crate::context::BACKEND_TIME_REFERENCE.get().unwrap().clone();
        runtime.block_on(async {
            let mut ids = vec![];
            loop {
                if ids.len() != 0 {
                    GlobalActionTableContext::remove_ids(&ids, maid_tree.clone(), bfrt.clone()).await;
                }
                ids.clear();
                let epoch_timestamp_ms = start_timestamp.elapsed().as_millis() as u64;
                while (start_timestamp.elapsed().as_millis() as i64) - (epoch_timestamp_ms as i64) <= 100 && ids.len() <= 1000 {
                    if let Ok(id) = receiver.try_recv() {
                        ids.push(id);
                    }
                }
            }
        });
    }

    pub fn new(initial_action_tuples: Vec<ActionInstanceCount>, bfrt: Arc<RwLock<Option<BFRuntime>>>, info: &P4TableInfo, enable_deferred_id_del: bool) -> Result<(Self, Vec<Update>, Sender<u32>), ActionTableError> {
        let (mut maid_tree, ops) = MaidTree::new(initial_action_tuples, 24, 2048, 2048, 7168)?;
        GlobalActionTableContext::transcation_begin(&mut maid_tree);
        let table_updates = GlobalActionTableContext::transcation_commit(&mut maid_tree, info, ops)?;
        let tree = Arc::new(RwLock::new(maid_tree));
        let tree2 = tree.clone();
        let (sender, receiver) = tokio::sync::mpsc::channel(4000);
        let thread = if enable_deferred_id_del {
            Some(std::thread::spawn(move || {
                GlobalActionTableContext::maid_deletion_thread(receiver, tree2, bfrt);
            }))
        } else {
            None
        };
        let s = Self {
            maid_tree: tree,
            thread
        };
        Ok((s, table_updates, sender))
    }
    pub fn transcation_begin(maid_tree: &mut MaidTree) {
        maid_tree.action_tables_transcation_begin();
    }
    pub fn transcation_commit(maid_tree: &mut MaidTree, info: &P4TableInfo, mut additional_ops: Vec<ActionTableOperation>) -> Result<Vec<Update>, ActionTableError> {
        let mut ops = maid_tree.action_tables_transcation_commit()?;
        ops.append(&mut additional_ops);
        let mut updates = Vec::with_capacity(ops.len());
        for op in ops.into_iter() {
            let (entry, table_op) = match op {
                ActionTableOperation::Modify(_) => unreachable!(),
                ActionTableOperation::Insert(entry) => {
                    (entry, update::Type::Insert)
                },
                ActionTableOperation::Delete(entry) => {
                    (entry, update::Type::Delete)
                },
            };
            let (key, data) = match entry.action_tuple {
                ActionInstance::Nop => continue,
                ActionInstance::Decap(nocp) => {
                    let table_key = TableKey {
                        fields: vec![
                            KeyField {
                                field_id: info.get_key_id_by_name(entry.action_tuple.which_table().p4_table_name(), "meta.ma_id"),
                                match_type: Some(key_field::MatchType::Ternary(Ternary {
                                    value: (entry.value as u32).to_be_bytes()[1..].to_vec(),
                                    mask: (entry.mask as u32).to_be_bytes()[1..].to_vec()
                                }))
                            }
                        ]
                    };
                    let table_data = TableData {
                        action_id: info.get_action_id_by_name(entry.action_tuple.which_table().p4_table_name(), "Ingress.ul_mark_tos_table_forward_v4"),
                        fields: vec![
                            DataField {
                                field_id: 1,
                                value: Some(data_field::Value::Stream(vec![0u8])) // mark DSCP
                            },
                            DataField {
                                field_id: 2,
                                value: Some(data_field::Value::Stream(vec![0u8])) // DSCP
                            },
                            DataField {
                                field_id: 3,
                                value: Some(data_field::Value::Stream(vec![(nocp as u8)])) // NoCP
                            },
                            DataField {
                                field_id: 4,
                                value: Some(data_field::Value::Stream(vec![0u8])) // Drop
                            }
                        ]
                    };
                    (table_key, table_data)
                },
                ActionInstance::DecapMarkDSCP(nocp, dscp) => {
                    let table_key = TableKey {
                        fields: vec![
                            KeyField {
                                field_id: info.get_key_id_by_name(entry.action_tuple.which_table().p4_table_name(), "meta.ma_id"),
                                match_type: Some(key_field::MatchType::Ternary(Ternary {
                                    value: (entry.value as u32).to_be_bytes()[1..].to_vec(),
                                    mask: (entry.mask as u32).to_be_bytes()[1..].to_vec()
                                }))
                            }
                        ]
                    };
                    let table_data = TableData {
                        action_id: info.get_action_id_by_name(entry.action_tuple.which_table().p4_table_name(), "Ingress.ul_mark_tos_table_forward_v4"),
                        fields: vec![
                            DataField {
                                field_id: 1,
                                value: Some(data_field::Value::Stream(vec![1u8])) // mark DSCP
                            },
                            DataField {
                                field_id: 2,
                                value: Some(data_field::Value::Stream(vec![dscp])) // DSCP
                            },
                            DataField {
                                field_id: 3,
                                value: Some(data_field::Value::Stream(vec![(nocp as u8)])) // NoCP
                            },
                            DataField {
                                field_id: 4,
                                value: Some(data_field::Value::Stream(vec![0u8])) // Drop
                            }
                        ]
                    };
                    (table_key, table_data)
                },
                ActionInstance::EncapDl(nocp, ip, qfi) => {
                    let table_key = TableKey {
                        fields: vec![
                            KeyField {
                                field_id: info.get_key_id_by_name(entry.action_tuple.which_table().p4_table_name(), "meta.ma_id"),
                                match_type: Some(key_field::MatchType::Ternary(Ternary {
                                    value: (entry.value as u32).to_be_bytes()[1..].to_vec(),
                                    mask: (entry.mask as u32).to_be_bytes()[1..].to_vec()
                                }))
                            }
                        ]
                    };
                    let table_data = TableData {
                        action_id: info.get_action_id_by_name(entry.action_tuple.which_table().p4_table_name(), "Ingress.dl_to_N3N9_table_v4"),
                        fields: vec![
                            DataField {
                                field_id: 1,
                                value: Some(data_field::Value::Stream(ip.octets().to_vec())) // IP
                            },
                            DataField {
                                field_id: 2,
                                value: Some(data_field::Value::Stream(vec![qfi])) // QFI
                            },
                            DataField {
                                field_id: 3,
                                value: Some(data_field::Value::Stream(vec![(nocp as u8)])) // NoCP
                            },
                            DataField {
                                field_id: 4,
                                value: Some(data_field::Value::Stream(vec![0u8])) // Buffer
                            },
                            DataField {
                                field_id: 4,
                                value: Some(data_field::Value::Stream(vec![0u8])) // Drop
                            }
                        ]
                    };
                    (table_key, table_data)
                },
                ActionInstance::EncapUl(nocp, ip, qfi) => {
                    let table_key = TableKey {
                        fields: vec![
                            KeyField {
                                field_id: info.get_key_id_by_name(entry.action_tuple.which_table().p4_table_name(), "meta.ma_id"),
                                match_type: Some(key_field::MatchType::Ternary(Ternary {
                                    value: (entry.value as u32).to_be_bytes()[1..].to_vec(),
                                    mask: (entry.mask as u32).to_be_bytes()[1..].to_vec()
                                }))
                            }
                        ]
                    };
                    let table_data = TableData {
                        action_id: info.get_action_id_by_name(entry.action_tuple.which_table().p4_table_name(), "Ingress.ul_to_N3N9_table_table_v4"),
                        fields: vec![
                            DataField {
                                field_id: 1,
                                value: Some(data_field::Value::Stream(0u32.to_be_bytes().to_vec())) // IP
                            },
                            DataField {
                                field_id: 2,
                                value: Some(data_field::Value::Stream(vec![qfi])) // QFI
                            },
                            DataField {
                                field_id: 3,
                                value: Some(data_field::Value::Stream(vec![(nocp as u8)])) // NoCP
                            }
                        ]
                    };
                    (table_key, table_data)
                },
                ActionInstance::Buffer(nocp) => {
                    let table_key = TableKey {
                        fields: vec![
                            KeyField {
                                field_id: info.get_key_id_by_name(entry.action_tuple.which_table().p4_table_name(), "meta.ma_id"),
                                match_type: Some(key_field::MatchType::Ternary(Ternary {
                                    value: (entry.value as u32).to_be_bytes()[1..].to_vec(),
                                    mask: (entry.mask as u32).to_be_bytes()[1..].to_vec()
                                }))
                            }
                        ]
                    };
                    let table_data = TableData {
                        action_id: info.get_action_id_by_name(entry.action_tuple.which_table().p4_table_name(), "Ingress.dl_to_N3N9_table_v4"),
                        fields: vec![
                            DataField {
                                field_id: 1,
                                value: Some(data_field::Value::Stream(0u32.to_be_bytes().to_vec())) // IP
                            },
                            DataField {
                                field_id: 2,
                                value: Some(data_field::Value::Stream(vec![0u8])) // QFI
                            },
                            DataField {
                                field_id: 3,
                                value: Some(data_field::Value::Stream(vec![(nocp as u8)])) // NoCP
                            },
                            DataField {
                                field_id: 4,
                                value: Some(data_field::Value::Stream(vec![1u8])) // Buffer
                            },
                            DataField {
                                field_id: 4,
                                value: Some(data_field::Value::Stream(vec![0u8])) // Drop
                            }
                        ]
                    };
                    (table_key, table_data)
                },
                ActionInstance::DropDl(nocp) => {
                    let table_key = TableKey {
                        fields: vec![
                            KeyField {
                                field_id: info.get_key_id_by_name(entry.action_tuple.which_table().p4_table_name(), "meta.ma_id"),
                                match_type: Some(key_field::MatchType::Ternary(Ternary {
                                    value: (entry.value as u32).to_be_bytes()[1..].to_vec(),
                                    mask: (entry.mask as u32).to_be_bytes()[1..].to_vec()
                                }))
                            }
                        ]
                    };
                    let table_data = TableData {
                        action_id: info.get_action_id_by_name(entry.action_tuple.which_table().p4_table_name(), "Ingress.dl_to_N3N9_table_v4"),
                        fields: vec![
                            DataField {
                                field_id: 1,
                                value: Some(data_field::Value::Stream(0u32.to_be_bytes().to_vec())) // IP
                            },
                            DataField {
                                field_id: 2,
                                value: Some(data_field::Value::Stream(vec![0u8])) // QFI
                            },
                            DataField {
                                field_id: 3,
                                value: Some(data_field::Value::Stream(vec![(nocp as u8)])) // NoCP
                            },
                            DataField {
                                field_id: 4,
                                value: Some(data_field::Value::Stream(vec![0u8])) // Buffer
                            },
                            DataField {
                                field_id: 4,
                                value: Some(data_field::Value::Stream(vec![1u8])) // Drop
                            }
                        ]
                    };
                    (table_key, table_data)
                },
                ActionInstance::DropUl(nocp) => {
                    let table_key = TableKey {
                        fields: vec![
                            KeyField {
                                field_id: info.get_key_id_by_name(entry.action_tuple.which_table().p4_table_name(), "meta.ma_id"),
                                match_type: Some(key_field::MatchType::Ternary(Ternary {
                                    value: (entry.value as u32).to_be_bytes()[1..].to_vec(),
                                    mask: (entry.mask as u32).to_be_bytes()[1..].to_vec()
                                }))
                            }
                        ]
                    };
                    let table_data = TableData {
                        action_id: info.get_action_id_by_name(entry.action_tuple.which_table().p4_table_name(), "Ingress.ul_mark_tos_table_forward_v4"),
                        fields: vec![
                            DataField {
                                field_id: 1,
                                value: Some(data_field::Value::Stream(vec![0u8])) // mark DSCP
                            },
                            DataField {
                                field_id: 2,
                                value: Some(data_field::Value::Stream(vec![0u8])) // DSCP
                            },
                            DataField {
                                field_id: 3,
                                value: Some(data_field::Value::Stream(vec![(nocp as u8)])) // NoCP
                            },
                            DataField {
                                field_id: 4,
                                value: Some(data_field::Value::Stream(vec![1u8])) // Drop
                            }
                        ]
                    };
                    (table_key, table_data)
                },
            };
            let table_entry = TableEntry {
                table_id: info.get_table_by_name(entry.action_tuple.which_table().p4_table_name()),
                data: Some(data),
                is_default_entry: false,
                table_read_flag: None,
                table_mod_inc_flag: None,
                entry_tgt: None,
                table_flags: None,
                value: Some(table_entry::Value::Key(key)),
            };
            let entity = Entity {
                entity: Some(entity::Entity::TableEntry(table_entry))
            };
            let update = Update {
                r#type: table_op as _,
                entity: Some(entity)
            };
            updates.push(update);
        }
        Ok(updates)
    }
    pub async fn update<'a>(
        &'a mut self,
        bfrt: Arc<RwLock<Option<BFRuntime>>>,
        to_del: Vec<ToDelPipeline<'a>>,
        mut to_mod: Vec<ToModPipeline<'a>>,
        mut to_add: Vec<ToAddPipeline<'a>>,
        seid: u64,
        untouched_pipelines: Vec<TofinoPdrPipeline>,
        info: &P4TableInfo,
        maid_del_tx: &Sender<u32>,
        enable_deferred_id_del: bool
    ) -> Result<(LinearMap<u16, u32>, Vec<Update>), ActionTableError> {
        // println!("=======Rule update for [SEID={}]=======", seid);
        // println!("To Delete");
        // println!("{:#?}", to_del);
        // println!("To Modify");
        // println!("{:#?}", to_mod);
        // println!("To Insert");
        // println!("{:#?}", to_add);

        let mut need_immediate_pulling = {
            // if assoicateion or matching field is updated or delete
            if to_del.len() != 0 {
                true
            } else if to_mod.len() != 0 {
                let mut ret = false;
                for m in to_mod.iter() {
                    ret |= m.update_matches.is_some();
                    ret |= m.update_urr.is_some();
                }
                ret
            } else {
                false
            }
        };
        if !enable_deferred_id_del {
            need_immediate_pulling = true;
        }

        let mut modified_pdr_id_to_maid_map = LinearMap::with_capacity(to_add.len() + to_mod.len());

        // Lits of MAIDs to be freed after a pull
        let mut list_of_to_free_maid_with_pdr_id = LinearMap::with_capacity(to_mod.len());

        // step 1: update tree
        let mut table_updates = {
            // hold MAID tree lock as short as possible so other threads can update the tree
            let mut maid_tree = self.maid_tree.write().await;
            Self::transcation_begin(&mut maid_tree);
            for item in to_del.iter() {
                maid_tree.free_maid(item.maid as _);
            }
            for item in to_mod.iter_mut() {
                if let Some(action_update) = &item.update_action {
                    if need_immediate_pulling {
                        maid_tree.free_maid(item.maid as _);
                    } else {
                        list_of_to_free_maid_with_pdr_id.insert(item.maid, item.pdr_id);
                    }
                    let new_id = maid_tree.allocate_maid(action_update.new_action)? as u32;
                    modified_pdr_id_to_maid_map.insert(item.pdr_id, new_id);
                    item.allocated_maid = Some(new_id);
                }
            }
            for item in to_add.iter_mut() {
                let new_id = maid_tree.allocate_maid(*item.action)? as u32;
                item.allocated_maid = Some(new_id);
                modified_pdr_id_to_maid_map.insert(item.pdr_id, new_id);
            }
            Self::transcation_commit(&mut maid_tree, info, vec![])?
        };
        table_updates.reserve(16);

        // step 1.1 generate usage reporting assocaitions
        let mut all_associations = Vec::with_capacity(10);
        for pipe in untouched_pipelines.into_iter() {
            if pipe.maid.is_some() {
                all_associations.push(PdrUsageReportingAssociation {
                    pdr_id: pipe.pdr_id,
                    maid: pipe.maid.unwrap(),
                    ingress_entries: pipe.matches.iter().map(|f| {
                        TableEntry {
                            table_id: info.get_table_by_name(f.p4_table_name()),
                            data: None,
                            is_default_entry: false,
                            table_read_flag: None,
                            table_mod_inc_flag: None,
                            entry_tgt: None,
                            table_flags: None,
                            value: Some(table_entry::Value::Key(f.generate_table_key(info).unwrap())),
                        }
                    }).collect_vec(),
                    is_uplink: pipe.matches[0].get_direction() == PipelineTemplateDirection::Uplink,
                    urr_ids: LinearSet::from_iter(pipe.linked_urrs.iter().cloned()),
                });
            }
        }
        for item in to_mod.iter() {
            let maid = item.allocated_maid.unwrap_or(item.maid);
            all_associations.push(PdrUsageReportingAssociation {
                pdr_id: item.pdr_id,
                maid: maid,
                ingress_entries: item.matches.iter().map(|f| {
                    TableEntry {
                        table_id: info.get_table_by_name(f.p4_table_name()),
                        data: None,
                        is_default_entry: false,
                        table_read_flag: None,
                        table_mod_inc_flag: None,
                        entry_tgt: None,
                        table_flags: None,
                        value: Some(table_entry::Value::Key(f.generate_table_key(info).unwrap())),
                    }
                }).collect_vec(),
                is_uplink: item.matches[0].get_direction() == PipelineTemplateDirection::Uplink,
                urr_ids: LinearSet::from_iter(item.urr_ids.iter().cloned()),
            });
        }
        for item in to_add.iter() {
            let maid = item.allocated_maid.unwrap();
            all_associations.push(PdrUsageReportingAssociation {
                pdr_id: item.pdr_id,
                maid: maid,
                ingress_entries: item.matches.iter().map(|f| {
                    TableEntry {
                        table_id: info.get_table_by_name(f.p4_table_name()),
                        data: None,
                        is_default_entry: false,
                        table_read_flag: None,
                        table_mod_inc_flag: None,
                        entry_tgt: None,
                        table_flags: None,
                        value: Some(table_entry::Value::Key(f.generate_table_key(info).unwrap())),
                    }
                }).collect_vec(),
                is_uplink: item.matches[0].get_direction() == PipelineTemplateDirection::Uplink,
                urr_ids: LinearSet::from_iter(item.urr_ids.iter().cloned()),
            });
        }

        // step 2: pull usage
        table_updates.append(&mut super::usage_reporting::action_table_update(bfrt, info, seid, all_associations, need_immediate_pulling, list_of_to_free_maid_with_pdr_id, maid_del_tx).await);

        // step 3: update match table
        for item in to_del.into_iter() {
            for m in item.matches {
                let table_entry = TableEntry {
                    table_id: info.get_table_by_name(m.p4_table_name()),
                    data: None,
                    is_default_entry: false,
                    table_read_flag: None,
                    table_mod_inc_flag: None,
                    entry_tgt: None,
                    table_flags: None,
                    value: Some(table_entry::Value::Key(m.generate_table_key(info).unwrap())),
                };
                let entity = Entity {
                    entity: Some(entity::Entity::TableEntry(table_entry))
                };
                let update = Update {
                    r#type: update::Type::Delete as _,
                    entity: Some(entity)
                };
                table_updates.push(update);
            }
        }

        for item in to_mod.into_iter() {
            let maid = item.allocated_maid.unwrap_or(item.maid);
            if (item.update_qer.is_some() || item.update_action.is_some()) && item.update_matches.is_none() {
                // if we only perform (QER ID update OR MAID update)
                // we modify match entries with new global_qer_id
                let global_qer_id = if let Some(update_qer) = item.update_qer {
                    update_qer.new_global_qer_id
                } else {
                    item.global_qer_id
                };
                for m in item.old_matches {
                    let table_entry = TableEntry {
                        table_id: info.get_table_by_name(m.p4_table_name()),
                        data: Some(m.generate_table_action_data(info, maid, global_qer_id).unwrap()),
                        is_default_entry: false,
                        table_read_flag: None,
                        table_mod_inc_flag: None,
                        entry_tgt: None,
                        table_flags: None,
                        value: Some(table_entry::Value::Key(m.generate_table_key(info).unwrap())),
                    };
                    let entity = Entity {
                        entity: Some(entity::Entity::TableEntry(table_entry))
                    };
                    let update = Update {
                        r#type: update::Type::Modify as _,
                        entity: Some(entity)
                    };
                    table_updates.push(update);
                }
            }
            if let Some(match_update) = item.update_matches {
                // if we perform match update
                // we delete old then insert new
                let global_qer_id = item.global_qer_id;
                for m in match_update.old_matches {
                    let table_entry = TableEntry {
                        table_id: info.get_table_by_name(m.p4_table_name()),
                        data: None,
                        is_default_entry: false,
                        table_read_flag: None,
                        table_mod_inc_flag: None,
                        entry_tgt: None,
                        table_flags: None,
                        value: Some(table_entry::Value::Key(m.generate_table_key(info).unwrap())),
                    };
                    let entity = Entity {
                        entity: Some(entity::Entity::TableEntry(table_entry))
                    };
                    let update = Update {
                        r#type: update::Type::Delete as _,
                        entity: Some(entity)
                    };
                    table_updates.push(update);
                }
                for m in match_update.new_matches {
                    let table_entry = TableEntry {
                        table_id: info.get_table_by_name(m.p4_table_name()),
                        data: Some(m.generate_table_action_data(info, maid, global_qer_id).unwrap()),
                        is_default_entry: false,
                        table_read_flag: None,
                        table_mod_inc_flag: None,
                        entry_tgt: None,
                        table_flags: None,
                        value: Some(table_entry::Value::Key(m.generate_table_key(info).unwrap())),
                    };
                    let entity = Entity {
                        entity: Some(entity::Entity::TableEntry(table_entry))
                    };
                    let update = Update {
                        r#type: update::Type::Insert as _,
                        entity: Some(entity)
                    };
                    table_updates.push(update);
                }
            }
            // URR is updated by the usage reporting module, not here
            // Action is already updated in step 1
        }
        for item in to_add.into_iter() {
            let maid = item.allocated_maid.unwrap();
            for m in item.matches {
                let table_entry = TableEntry {
                    table_id: info.get_table_by_name(m.p4_table_name()),
                    data: Some(m.generate_table_action_data(info, maid, item.global_qer_id).unwrap()),
                    is_default_entry: false,
                    table_read_flag: None,
                    table_mod_inc_flag: None,
                    entry_tgt: None,
                    table_flags: None,
                    value: Some(table_entry::Value::Key(m.generate_table_key(info).unwrap())),
                };
                let entity = Entity {
                    entity: Some(entity::Entity::TableEntry(table_entry))
                };
                let update = Update {
                    r#type: update::Type::Insert as _,
                    entity: Some(entity)
                };
                table_updates.push(update);
            }
        }

        // println!("{:#?}", table_updates);

        //tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
        
        Ok((modified_pdr_id_to_maid_map, table_updates))
    }
    pub async fn remove_maids(&mut self, maids: Vec<u32>, info: &P4TableInfo, maid_del_tx: &Sender<u32>, enable_deferred_id_del: bool) -> Vec<Update> {
        let mut updates;
        
        if !enable_deferred_id_del {
            let mut maid_tree = self.maid_tree.write().await;
            Self::transcation_begin(&mut maid_tree);
            for item in maids.iter() {
                maid_tree.free_maid(*item as _);
            }
            updates = Self::transcation_commit(&mut maid_tree, info, vec![]).unwrap();
        } else {
            updates = vec![]
        };
        updates.reserve(maids.len());
        for maid in maids.iter() {
            if enable_deferred_id_del {
                maid_del_tx.send(*maid).await.unwrap();
            } else {
                let egress_table_entry = TableKey {
                    fields: vec![
                        KeyField {
                            field_id: info.get_key_id_by_name("pipe.Egress.accounting.accounting_exact", "ma_id"),
                            match_type: Some(key_field::MatchType::Exact(Exact {
                                value: maid.to_be_bytes()[1..].to_vec()
                            }))
                        }
                    ]
                };
                let table_entry = TableEntry {
                    table_id: info.get_table_by_name("pipe.Egress.accounting.accounting_exact"),
                    data: None,//Some(table_data),
                    is_default_entry: false,
                    table_read_flag: None,
                    table_mod_inc_flag: None,
                    entry_tgt: None,
                    table_flags: None,
                    value: Some(table_entry::Value::Key(egress_table_entry)),
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
        updates
    }
}
