#include <core.p4>
#include <tna.p4>

#include "headers.p4"
#include "settings.p4"
#include "pdr.p4"
#include "xgboost.p4"

parser TofinoIngressParser(
			packet_in                                   pkt,
	inout   ig_metadata_t                               meta,
	out     ingress_intrinsic_metadata_t                ig_intr_md
) {
	state start {
		pkt.extract(ig_intr_md);
		transition select(ig_intr_md.resubmit_flag) {
			1 : parse_resubmit;
			0 : parse_port_metadata;
		}
	}

	state parse_resubmit {
		// Parse resubmitted packet here.
		pkt.advance(64);
		transition accept;
	}

	state parse_port_metadata {
		pkt.advance(64);  //tofino 1 port metadata size
		transition accept;
	}
}

parser IngressParser(
			packet_in                                   pkt,
	/* User */
	out     ig_header_t                                 hdr,
	out     ig_metadata_t                               meta,
	out     ingress_intrinsic_metadata_t                ig_intr_md
) {
    ParserCounter()         len_limit;
	TofinoIngressParser() tofino_parser;

	state start {
		tofino_parser.apply(pkt, meta, ig_intr_md);
		transition init_metadata;
	}

	state init_metadata {
		meta.uplink = false;
		meta.l4_protocol = 0;
		meta.l4_src_port = 0;
		meta.l4_dst_port = 0;	
		meta.individual_gbr_meter_color = MeterColor_t.GREEN;
		//meta.gbr_meter_color = MeterColor_t.GREEN;
		transition parse_ethernet;
	}

	state parse_ethernet {
		pkt.extract(hdr.ethernet);
		transition select(hdr.ethernet.etherType) {
            DETECTION_TRIGGER:  parse_detection_trigger;
            default :           parse_ip_gateway;
        }
	}

    state parse_detection_trigger {
        pkt.extract(hdr.detection_trigger);
        transition accept;
    }

	state parse_ip_gateway {
		transition select(hdr.ethernet.src_part2) {
			N6_MAC_MAPPING
			default: parse_underlay;
		}
	}

	state parse_underlay {
		transition select(hdr.ethernet.etherType) {
			0x0800: parse_underlay_ipv4;
			default: reject;
		}
	}

	state parse_underlay_ipv4 {
		pkt.extract(hdr.underlay_ipv4);
		transition select(hdr.underlay_ipv4.ihl) {
				  5 : parse_underlay_ipv4_no_options;
			default : reject;
		}
	}

	state parse_underlay_ipv4_no_options {
		transition select(hdr.underlay_ipv4.protocol) {
			17: parse_underlay_udp;
			default: reject;
		}
	}


	state parse_underlay_udp {
		pkt.extract(hdr.underlay_udp);
		transition select(hdr.underlay_udp.dstPort) {
			2152: parse_gtpu;
			default: reject;
		}
	}

	state parse_gtpu {
		pkt.extract(hdr.gtpu);
		meta.extracted_teid = hdr.gtpu.teid[(MA_ID_BITS  - COMPRESSED_QFI_BITS - 1):0];
		transition select(hdr.gtpu.extensionHeaderFlag, hdr.gtpu.sequenceNumberFlag, hdr.gtpu.npduNumberFlag) {
			(0, 0, 0): parse_overlay_gateway;
			default: parse_gtpu_optional;
		}
	}

	state parse_gtpu_optional {
		pkt.extract(hdr.gtpu_optional);
		transition select(hdr.gtpu_optional.nextExtensionHeaderType) {
			8w0b10000101: parse_gtpu_psc;
			default: reject; // not handled we can only reject
		}
	}

	state parse_gtpu_psc {
		pkt.extract(hdr.gtpu_ext_psc);
		meta.tunnel_qfi = hdr.gtpu_ext_psc.qfi;
		meta.uplink = (bool)hdr.gtpu_ext_psc.pduType[0:0];
		transition select(hdr.gtpu_ext_psc.extHdrLength) {
			0: reject;
			1: parse_gtpu_psc_optional_1;
			2: parse_gtpu_psc_optional_2;
			3: parse_gtpu_psc_optional_3;
			4: parse_gtpu_psc_optional_4;
			5: parse_gtpu_psc_optional_5;
			6: parse_gtpu_psc_optional_6;
			7: parse_gtpu_psc_optional_7;
			8: parse_gtpu_psc_optional_8;
			9: parse_gtpu_psc_optional_9;
			10: parse_gtpu_psc_optional_10;
			11: parse_gtpu_psc_optional_11;
			12: parse_gtpu_psc_optional_12;
			default: reject;
		}
	}

	state parse_gtpu_psc_optional_1 {
		// skip 8 bit nextHdr
		pkt.advance(8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_2 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (2 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_3 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (3 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_4 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (4 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_5 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (5 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_6 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (6 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_7 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (7 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_8 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (8 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_9 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (9 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_10 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (10 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_11 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (11 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_12 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (12 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}

	state parse_overlay_gateway {
		// handle cases where no more data after GTP-U mandatory header
		transition select(hdr.gtpu.messageType) {
			255: parse_overlay; // G-PDU type with overlay data
			default: accept; // other GTP-U control messages
		}
	}

	state parse_overlay {
		bit<4> ip_ver = pkt.lookahead<bit<4>>();
		transition select(ip_ver) {
			4w4: parse_overlay_ipv4;
			default: reject; // other L3 protocol not supported
		}
	}

	state parse_overlay_ipv4 {
		pkt.extract(hdr.overlay_ipv4);
		meta.l4_protocol = hdr.overlay_ipv4.protocol;
		meta.overlay_length = hdr.overlay_ipv4.totalLen;
		transition select(hdr.overlay_ipv4.ihl) {
				  5 : parse_overlay_ipv4_no_options;
			default : reject; // invalid IHL value
		}
	}

	state parse_overlay_ipv4_no_options {
		transition select(hdr.overlay_ipv4.protocol) {
			6: parse_overlay_tcp;
			17: parse_overlay_udp;
			default: accept; // ICMP and other L4 protocols
		}
	}

	state parse_overlay_udp {
		pkt.extract(hdr.overlay_udp);
		meta.l4_src_port = hdr.overlay_udp.srcPort;
		meta.l4_dst_port = hdr.overlay_udp.dstPort;
		transition select(hdr.overlay_udp.srcPort, hdr.overlay_udp.dstPort) {
            (_,53) :      parse_dns_h;
            //(53,_) :      parse_dns_h;
            default :   accept;
        }
	}

	state parse_overlay_tcp {
		pkt.extract(hdr.overlay_tcp);
		meta.l4_src_port = hdr.overlay_tcp.srcPort;
		meta.l4_dst_port = hdr.overlay_tcp.dstPort;
		transition accept; // Done
	}


    state parse_dns_h {
        pkt.extract(hdr.dns);
        len_limit.set(8w60);

        transition parse_len1;
    }

    
    state parse_len1 {
        bit<8> len = pkt.lookahead<bit<8>>();
        len_limit.decrement(8w1);

        transition select(len) {
            1 .. 31:           preparse_label1;
            0:                                      postparse_dns;
            default:                                accept;
        }
    }
    
    state parse_len2 {
        bit<8> len = pkt.lookahead<bit<8>>();
        len_limit.decrement(8w1);

        transition select(len) {
            1 .. 31:           preparse_label2;
            0:                                      postparse_dns;
            default:                                accept;
        }
    }
    
    state parse_len3 {
        bit<8> len = pkt.lookahead<bit<8>>();
        len_limit.decrement(8w1);

        transition select(len) {
            1 .. 31:           preparse_label3;
            0:                                      postparse_dns;
            default:                                accept;
        }
    }
    
    state parse_len4 {
        bit<8> len = pkt.lookahead<bit<8>>();
        len_limit.decrement(8w1);

        transition select(len) {
            1 .. 31:           preparse_label4;
            0:                                      postparse_dns;
            default:                                accept;
        }
    }
    
    state parse_len5 {
        bit<8> len = pkt.lookahead<bit<8>>();
        len_limit.decrement(8w1);

        transition select(len) {
            
            0:                                      postparse_dns;
            default:                                accept;
        }
    }
    

    
    state preparse_label1 {
        pkt.extract(hdr.len1);
        
        transition select(hdr.len1.l) { 
            16 &&& 16:          parse_label1_16;
            8 &&& 8:          parse_label1_8;
            4 &&& 4:          parse_label1_4;
            2 &&& 2:          parse_label1_2;
            1 &&& 1:          parse_label1_1;
        }
    }

    state postparse_label1 {
        transition select(len_limit.is_negative()) {
            false:                    parse_len2;
            true:                     accept;
        }
    }
    
    state parse_label1_16 {
        pkt.extract(hdr.label1_16);
        len_limit.decrement(8w16);

        transition select(hdr.len1.l) { 
            8 &&& 8:          parse_label1_8;
            4 &&& 4:          parse_label1_4;
            2 &&& 2:          parse_label1_2;
            1 &&& 1:          parse_label1_1;
            default:                  postparse_label1;
        }
    }
    
    state parse_label1_8 {
        pkt.extract(hdr.label1_8);
        len_limit.decrement(8w8);

        transition select(hdr.len1.l) { 
            4 &&& 4:          parse_label1_4;
            2 &&& 2:          parse_label1_2;
            1 &&& 1:          parse_label1_1;
            default:                  postparse_label1;
        }
    }
    
    state parse_label1_4 {
        pkt.extract(hdr.label1_4);
        len_limit.decrement(8w4);

        transition select(hdr.len1.l) { 
            2 &&& 2:          parse_label1_2;
            1 &&& 1:          parse_label1_1;
            default:                  postparse_label1;
        }
    }
    
    state parse_label1_2 {
        pkt.extract(hdr.label1_2);
        len_limit.decrement(8w2);

        transition select(hdr.len1.l) { 
            1 &&& 1:          parse_label1_1;
            default:                  postparse_label1;
        }
    }
    
    state parse_label1_1 {
        pkt.extract(hdr.label1_1);
        len_limit.decrement(8w1);

        transition postparse_label1;
    }
    
    state preparse_label2 {
        pkt.extract(hdr.len2);
        
        transition select(hdr.len2.l) { 
            16 &&& 16:          parse_label2_16;
            8 &&& 8:          parse_label2_8;
            4 &&& 4:          parse_label2_4;
            2 &&& 2:          parse_label2_2;
            1 &&& 1:          parse_label2_1;
        }
    }

    state postparse_label2 {
        transition select(len_limit.is_negative()) {
            false:                    parse_len3;
            true:                     accept;
        }
    }
    
    state parse_label2_16 {
        pkt.extract(hdr.label2_16);
        len_limit.decrement(8w16);

        transition select(hdr.len2.l) { 
            8 &&& 8:          parse_label2_8;
            4 &&& 4:          parse_label2_4;
            2 &&& 2:          parse_label2_2;
            1 &&& 1:          parse_label2_1;
            default:                  postparse_label2;
        }
    }
    
    state parse_label2_8 {
        pkt.extract(hdr.label2_8);
        len_limit.decrement(8w8);

        transition select(hdr.len2.l) { 
            4 &&& 4:          parse_label2_4;
            2 &&& 2:          parse_label2_2;
            1 &&& 1:          parse_label2_1;
            default:                  postparse_label2;
        }
    }
    
    state parse_label2_4 {
        pkt.extract(hdr.label2_4);
        len_limit.decrement(8w4);

        transition select(hdr.len2.l) { 
            2 &&& 2:          parse_label2_2;
            1 &&& 1:          parse_label2_1;
            default:                  postparse_label2;
        }
    }
    
    state parse_label2_2 {
        pkt.extract(hdr.label2_2);
        len_limit.decrement(8w2);

        transition select(hdr.len2.l) { 
            1 &&& 1:          parse_label2_1;
            default:                  postparse_label2;
        }
    }
    
    state parse_label2_1 {
        pkt.extract(hdr.label2_1);
        len_limit.decrement(8w1);

        transition postparse_label2;
    }
    
    state preparse_label3 {
        pkt.extract(hdr.len3);
        
        transition select(hdr.len3.l) { 
            16 &&& 16:          parse_label3_16;
            8 &&& 8:          parse_label3_8;
            4 &&& 4:          parse_label3_4;
            2 &&& 2:          parse_label3_2;
            1 &&& 1:          parse_label3_1;
        }
    }

    state postparse_label3 {
        transition select(len_limit.is_negative()) {
            false:                    parse_len4;
            true:                     accept;
        }
    }
    
    state parse_label3_16 {
        pkt.extract(hdr.label3_16);
        len_limit.decrement(8w16);

        transition select(hdr.len3.l) { 
            8 &&& 8:          parse_label3_8;
            4 &&& 4:          parse_label3_4;
            2 &&& 2:          parse_label3_2;
            1 &&& 1:          parse_label3_1;
            default:                  postparse_label3;
        }
    }
    
    state parse_label3_8 {
        pkt.extract(hdr.label3_8);
        len_limit.decrement(8w8);

        transition select(hdr.len3.l) { 
            4 &&& 4:          parse_label3_4;
            2 &&& 2:          parse_label3_2;
            1 &&& 1:          parse_label3_1;
            default:                  postparse_label3;
        }
    }
    
    state parse_label3_4 {
        pkt.extract(hdr.label3_4);
        len_limit.decrement(8w4);

        transition select(hdr.len3.l) { 
            2 &&& 2:          parse_label3_2;
            1 &&& 1:          parse_label3_1;
            default:                  postparse_label3;
        }
    }
    
    state parse_label3_2 {
        pkt.extract(hdr.label3_2);
        len_limit.decrement(8w2);

        transition select(hdr.len3.l) { 
            1 &&& 1:          parse_label3_1;
            default:                  postparse_label3;
        }
    }
    
    state parse_label3_1 {
        pkt.extract(hdr.label3_1);
        len_limit.decrement(8w1);

        transition postparse_label3;
    }
    
    state preparse_label4 {
        pkt.extract(hdr.len4);
        
        transition select(hdr.len4.l) { 
            16 &&& 16:          parse_label4_16;
            8 &&& 8:          parse_label4_8;
            4 &&& 4:          parse_label4_4;
            2 &&& 2:          parse_label4_2;
            1 &&& 1:          parse_label4_1;
        }
    }

    state postparse_label4 {
        transition select(len_limit.is_negative()) {
            false:                    parse_len5;
            true:                     accept;
        }
    }
    
    state parse_label4_16 {
        pkt.extract(hdr.label4_16);
        len_limit.decrement(8w16);

        transition select(hdr.len4.l) { 
            8 &&& 8:          parse_label4_8;
            4 &&& 4:          parse_label4_4;
            2 &&& 2:          parse_label4_2;
            1 &&& 1:          parse_label4_1;
            default:                  postparse_label4;
        }
    }
    
    state parse_label4_8 {
        pkt.extract(hdr.label4_8);
        len_limit.decrement(8w8);

        transition select(hdr.len4.l) { 
            4 &&& 4:          parse_label4_4;
            2 &&& 2:          parse_label4_2;
            1 &&& 1:          parse_label4_1;
            default:                  postparse_label4;
        }
    }
    
    state parse_label4_4 {
        pkt.extract(hdr.label4_4);
        len_limit.decrement(8w4);

        transition select(hdr.len4.l) { 
            2 &&& 2:          parse_label4_2;
            1 &&& 1:          parse_label4_1;
            default:                  postparse_label4;
        }
    }
    
    state parse_label4_2 {
        pkt.extract(hdr.label4_2);
        len_limit.decrement(8w2);

        transition select(hdr.len4.l) { 
            1 &&& 1:          parse_label4_1;
            default:                  postparse_label4;
        }
    }
    
    state parse_label4_1 {
        pkt.extract(hdr.label4_1);
        len_limit.decrement(8w1);

        transition postparse_label4;
    }
    

    state postparse_dns {
        pkt.extract(hdr.dns_extra);

        transition accept;
    }
}

control ACL(
	inout   ig_header_t                                 hdr,
	inout   ig_metadata_t                               meta,
	/* Intrinsic */
	in      ingress_intrinsic_metadata_t                ig_intr_md,
	in      ingress_intrinsic_metadata_from_parser_t    ig_prsr_md,
	inout   ingress_intrinsic_metadata_for_deparser_t   ig_dprsr_md,
	inout   ingress_intrinsic_metadata_for_tm_t         ig_tm_md
) {
	
	apply {

	}
}

control Accounting(
	/* Flow Identifiers */
	in      ma_id_t                               ma_id
)(bit<32> adj) {
	DirectCounter<bit<36>>(CounterType_t.PACKETS_AND_BYTES) usage_counters;
	action inc_counter() {
		usage_counters.count(adj);
	}

	table accounting_exact {
		key = {
			ma_id : exact;
		}
		actions = {
			inc_counter;
			@defaultonly NoAction;
		}
		const default_action = NoAction();
		counters = usage_counters;
		const size = TABLE_SIZE_ACCOUNTING;
	}
	apply {
		accounting_exact.apply();
	}
}

control RoutingIPv4(
	/* User */
	inout   ig_header_t                                 hdr,
	in      ipv4_addr_t                                 dst_ip,
	/* Intrinsic */
	inout   ingress_intrinsic_metadata_for_tm_t         ig_tm_md
) {
	// -----------------------------------------------------------------
	// step 7 : routing

	action send(PortId_t port, mac_addr_t src_mac, mac_addr_t dst_mac) {
		ig_tm_md.ucast_egress_port = port;
		hdr.ethernet.dst = dst_mac;
		hdr.ethernet.src_part1 = src_mac[31:0];
		hdr.ethernet.src_part2 = src_mac[47:32];
	}

	table ipv4_lpm {
		key = {
			dst_ip           : lpm;
		}
		actions = {
			send;
			@defaultonly NoAction;
		}
		const default_action = NoAction();
		const size = TABLE_SIZE_IPV4_LPM;
	}

	apply {
		ipv4_lpm.apply();
	}
}


control Ingress(
	/* User */
	inout   ig_header_t                                 hdr,
	inout   ig_metadata_t                               meta,
	/* Intrinsic */
	in      ingress_intrinsic_metadata_t                ig_intr_md,
	in      ingress_intrinsic_metadata_from_parser_t    ig_prsr_md,
	inout   ingress_intrinsic_metadata_for_deparser_t   ig_dprsr_md,
	inout   ingress_intrinsic_metadata_for_tm_t         ig_tm_md
) {
	// -----------------------------------------------------------------
	//                         Common actions

	action mark_for_drop() {
		ig_dprsr_md.drop_ctl = ig_dprsr_md.drop_ctl | 0b001;
	}

	action actual_drop() {
		mark_for_drop();
		exit;
	}

	action arp_reply(mac_addr_t request_mac) {
		//update operation code from request to reply
		hdr.arp.op_code = ARP_REPLY;
		
		hdr.arp.dst_mac = hdr.arp.src_mac;
		
		hdr.arp.src_mac = request_mac;

		ipv4_addr_t tmp = hdr.arp.src_ip;
		hdr.arp.src_ip = hdr.arp.dst_ip;
		hdr.arp.dst_ip = tmp;

		//update ethernet header
		hdr.ethernet.dst = hdr.ethernet.src_part1 ++ hdr.ethernet.src_part2;
		hdr.ethernet.src_part1 = request_mac[31:0];
		hdr.ethernet.src_part2 = request_mac[47:32];

		//send it back to the same port
		ig_tm_md.ucast_egress_port = ig_intr_md.ingress_port;
		ig_tm_md.bypass_egress = 1;
	}

	action gtpu_echo_response_ipv4() {
		ipv4_addr_t tmp = hdr.underlay_ipv4.srcAddr;
		hdr.underlay_ipv4.srcAddr = hdr.underlay_ipv4.dstAddr;
		hdr.underlay_ipv4.dstAddr = tmp;
	}

	table handle_gtpu_control_msg_table {
		key = {
			hdr.gtpu.isValid()          : exact;
			hdr.gtpu.messageType        : exact;
			hdr.underlay_ipv4.isValid() : exact;
		}
		actions = {
			gtpu_echo_response_ipv4;
			@defaultonly NoAction;
		}
		const default_action = NoAction();
		const entries = {
			(true, 1, true) : gtpu_echo_response_ipv4();
		}
		const size = 1;
	}

	// UPF constants
	ipv4_addr_t             underlay_src_ipv4 = 0;

	action set_upf_ip_table_set_ip(ipv4_addr_t upf_ipv4) {
		underlay_src_ipv4 = upf_ipv4;
	}

	table set_upf_ip_table {
		key = {
		}
		actions = {
			set_upf_ip_table_set_ip;
		}
		const size = 1;
	}

	// common actions

	bool do_nocp = false;
	bool do_buffer = false;

	action decap() { // used by UL_DECAP_TO_DN
		hdr.underlay_ipv4.setInvalid();
		hdr.underlay_ipv4_options.setInvalid();
		hdr.underlay_ipv6.setInvalid();
		hdr.underlay_udp.setInvalid();
		hdr.gtpu.setInvalid();
		hdr.gtpu_optional.setInvalid();
		hdr.gtpu_ext_psc.setInvalid();
		hdr.gtpu_ext_psc_optional.setInvalid();
		hdr.gtpu_ext_psc_next_header_type.setInvalid();
	}

    PDR_builtin_counters() pdr;

	// post match tables

	// UL
	action ul_no_mark_tos_table_forward(bool nocp) {
		decap();
		do_nocp = nocp;
	}

	action ul_no_mark_tos_table_drop() {
		mark_for_drop();
	}

	table ul_no_mark_tos_table {
		key = {
			meta.ma_id : ternary;
		}
		actions = {
			@defaultonly NoAction;
			ul_no_mark_tos_table_forward;
			ul_no_mark_tos_table_drop;
		}
		const default_action = NoAction();
		const size = 512 * 2;
	}

	bool do_drop = false;

	action ul_mark_tos_table_forward_v4(bool mark_tos, bit<8> tos_value, bool nocp, bool drop) {
		decap();
		do_drop = drop;
		if (mark_tos)
			hdr.overlay_ipv4.diffserv = tos_value;
		do_nocp = nocp;
	}

	table ul_mark_tos_table {
		key = {
			meta.ma_id : ternary;
		}
		actions = {
			@defaultonly NoAction;
			ul_mark_tos_table_forward_v4;
			mark_for_drop;
		}
		const default_action = NoAction();
		const size = 512 * 4;
	}

	action ul_to_N3N9_table_table_v4(ipv4_addr_t gnb_ipv4, qfi_t qfi_v, bool nocp) {
		do_nocp = nocp;
		hdr.underlay_ipv4.setValid();
		hdr.underlay_ipv4.srcAddr = underlay_src_ipv4;
		hdr.underlay_ipv4.dstAddr = gnb_ipv4;
		// other IPv4 fields
		hdr.underlay_ipv4.version = 4;
		hdr.underlay_ipv4.ihl = 5;
		hdr.underlay_ipv4.diffserv = 0;
		hdr.underlay_ipv4.identification = 16w0x1145;
		hdr.underlay_ipv4.flags = 3w0b010;
		hdr.underlay_ipv4.fragOffset = 0;
		hdr.underlay_ipv4.ttl = 65;
		hdr.underlay_ipv4.protocol = 17;

		hdr.underlay_udp.setValid();
		hdr.underlay_udp.srcPort = 2152;
		hdr.underlay_udp.dstPort = 2152;
		hdr.underlay_udp.len =
			8 + // UDP length
			8 + // GTP-U header
			4 + // GTP-U optional header
			4 + // GTP-U PSC ext header
			meta.overlay_length;
		// TODO: UDP checksum
		hdr.underlay_udp.checksum = 0;


		hdr.underlay_ipv4.totalLen =
			20 + // IP header
			8 +  // UDP length
			8 +  // GTP-U header
			4 +  // GTP-U optional header
			4 +  // GTP-U PSC ext header
			meta.overlay_length;

		hdr.gtpu.setValid();
		hdr.gtpu.version = 3w1;
		hdr.gtpu.protocolType = 1;
		hdr.gtpu.spare = 0;
		hdr.gtpu.extensionHeaderFlag = 1;
		hdr.gtpu.sequenceNumberFlag = 0;
		hdr.gtpu.npduNumberFlag = 0;
		hdr.gtpu.messageType = 255;
		hdr.gtpu.payloadLength =
			4 + // GTP-U optional header
			4 + // GTP-U PSC ext header
			meta.overlay_length;
		hdr.gtpu.teid = meta.teid;

		hdr.gtpu_optional.setValid();
		hdr.gtpu_optional.sequenceNumber = 0;
		hdr.gtpu_optional.npduNumber = 0;
		hdr.gtpu_optional.nextExtensionHeaderType = 8w0b10000101;

		hdr.gtpu_ext_psc.setValid();
		hdr.gtpu_ext_psc.extHdrLength = 1;
		hdr.gtpu_ext_psc.pduType = 0;
		hdr.gtpu_ext_psc.dontCare = 0;
		hdr.gtpu_ext_psc.qfi = qfi_v;
		meta.tunnel_qfi = qfi_v;

		hdr.gtpu_ext_psc_next_header_type.setValid();
		hdr.gtpu_ext_psc_next_header_type.content = 0;
	}

	table ul_to_N3N9_table {
		key = {
			meta.ma_id : ternary;
		}
		actions = {
			@defaultonly NoAction;
			ul_to_N3N9_table_table_v4;
		}
		const default_action = NoAction();
		const size = 512 * 4;
	}

	action dl_to_N3N9_table_v4(ipv4_addr_t gnb_ipv4, qfi_t qfi_v, bool nocp, bool buf, bool drop) {
		do_drop = drop;
		do_nocp = nocp;
		do_buffer = buf;
		hdr.underlay_ipv4.setValid();
		hdr.underlay_ipv4.srcAddr = underlay_src_ipv4;
		hdr.underlay_ipv4.dstAddr = gnb_ipv4;
		// other IPv4 fields
		hdr.underlay_ipv4.version = 4;
		hdr.underlay_ipv4.ihl = 5;
		hdr.underlay_ipv4.diffserv = 0;
		hdr.underlay_ipv4.identification = 16w0x1145;
		hdr.underlay_ipv4.flags = 3w0b010;
		hdr.underlay_ipv4.fragOffset = 0;
		hdr.underlay_ipv4.ttl = 65;
		hdr.underlay_ipv4.protocol = 17;

		hdr.underlay_udp.setValid();
		hdr.underlay_udp.srcPort = 2152;
		hdr.underlay_udp.dstPort = 2152;
		hdr.underlay_udp.len =
			8 + // UDP length
			8 + // GTP-U header
			4 + // GTP-U optional header
			4 + // GTP-U PSC ext header
			meta.overlay_length;
		// TODO: UDP checksum
		hdr.underlay_udp.checksum = 0;


		hdr.underlay_ipv4.totalLen =
			20 + // IP header
			8 +  // UDP length
			8 +  // GTP-U header
			4 +  // GTP-U optional header
			4 +  // GTP-U PSC ext header
			meta.overlay_length;

		hdr.gtpu.setValid();
		hdr.gtpu.version = 3w1;
		hdr.gtpu.protocolType = 1;
		hdr.gtpu.spare = 0;
		hdr.gtpu.extensionHeaderFlag = 1;
		hdr.gtpu.sequenceNumberFlag = 0;
		hdr.gtpu.npduNumberFlag = 0;
		hdr.gtpu.messageType = 255;
		hdr.gtpu.payloadLength =
			4 + // GTP-U optional header
			4 + // GTP-U PSC ext header
			meta.overlay_length;
		hdr.gtpu.teid = meta.teid;

		hdr.gtpu_optional.setValid();
		hdr.gtpu_optional.sequenceNumber = 0;
		hdr.gtpu_optional.npduNumber = 0;
		hdr.gtpu_optional.nextExtensionHeaderType = 8w0b10000101;

		hdr.gtpu_ext_psc.setValid();
		hdr.gtpu_ext_psc.extHdrLength = 1;
		hdr.gtpu_ext_psc.pduType = 0;
		hdr.gtpu_ext_psc.dontCare = 0;
		hdr.gtpu_ext_psc.qfi = qfi_v;
		meta.tunnel_qfi = qfi_v;

		hdr.gtpu_ext_psc_next_header_type.setValid();
		hdr.gtpu_ext_psc_next_header_type.content = 0;
	}

	// DL
	table dl_to_N3N9_table {
		key = {
			meta.ma_id : ternary;
		}
		actions = {
			@defaultonly NoAction;
			dl_to_N3N9_table_v4;
			mark_for_drop;
		}
		const default_action = NoAction();
		const size = 512 * 14;
	}

	// table dl_to_drop_table {
	// 	key = {
	// 		meta.ma_id : ternary;
	// 	}
	// 	actions = {
	// 		@defaultonly NoAction;
	// 		mark_for_drop;
	// 	}
	// 	const default_action = NoAction();
	// 	const size = 512;
	// }

	// step 2:

	action set_bridge_header() {
		hdr.bridge.header_type = HEADER_TYPE_BRIDGE;
		hdr.bridge.header_info = 0;
		//hdr.bridge.mirror_session = 0;
		hdr.bridge.ma_id = meta.ma_id;
	}

	ACL() acl;

	DirectMeter(MeterType_t.BYTES) bitrate_enforce_meters;

	action set_meter_color() {
		meta.individual_gbr_meter_color = bitrate_enforce_meters.execute();
	}

	table bitrate_enforce_table {
		key = {
			meta.qer_id     : exact;
		}
		actions = {
			@defaultonly NoAction;
			set_meter_color;
		}
		const default_action = NoAction();
		meters = bitrate_enforce_meters;
		size = TABLE_SIZE_ACCOUNTING >> 3;
	}

	action put_in_queue(QueueId_t qid) {
		ig_tm_md.qid = qid;
	}

	table qfi_to_queue_table {
		key = {
			meta.qfi : exact;
		}
		actions = {
			put_in_queue;
			@defaultonly NoAction;
		}
		const default_action = NoAction();
		const size = 64;
	}

	// step 3:

	action send_to_cpu_table_action(bit<8> flags, bool copy_to_cpu) {
		if (copy_to_cpu) {
			ig_tm_md.copy_to_cpu = 1;
		}
		hdr.cpu_header.setValid();
		hdr.cpu_header.cpu_header_magic = CPU_HEADER_MAGIC;
		hdr.cpu_header.flags = flags;
		hdr.cpu_header.ma_id = (bit<32>)meta.ma_id;
	}

	table send_to_cpu_table {
		key = {
			do_buffer : exact;
			do_nocp   : exact;
		}
		actions = {
			@defaultonly NoAction;
			send_to_cpu_table_action;
		}
		const default_action = NoAction();
		const entries = {
			(true, false) : send_to_cpu_table_action(8w0b00000010, false);
			(true,  true) : send_to_cpu_table_action(8w0b00000011, true);
			(false, true) : send_to_cpu_table_action(8w0b00000001, true);
		}
		const size = 3;
	}

	// step 3: routing
	RoutingIPv4() ipv4_routing_overlay;
	RoutingIPv4() ipv4_routing_underlay;

	//bit<(MA_ID_BITS - COMPRESSED_QFI_BITS)> extracted_teid;

	// step 3: track packet stats
	port_queue_id_t stats_index = 0;
	Counter<bit<32>, port_queue_id_t>(1024, CounterType_t.PACKETS) qos_reach_ig;

	action set_index(port_queue_id_t idx) {
		stats_index = idx;
	}
	table ig_stats_set_index_table {
		key = {
			ig_tm_md.ucast_egress_port[8:3] : exact;
			ig_tm_md.qid                    : exact;
		}
		actions = {
			@defaultonly NoAction;
			set_index;
		}
		const size = 1024;
	}

	ipv4_addr_t forward_to_offpath_table_match_ip = 0;

	action forward_to_offpath_table_action() {
		// ig_dprsr_md.mirror_type = EG_MIRROR_TYPE_SIMPLE_COPY;
		// hdr.bridge.header_type = HEADER_TYPE_IG_MIRROR;
		// hdr.bridge.header_info = HEADER_INFO_OFFPATH_DETECTION;
		//hdr.bridge.mirror_session = MIRROR_SESSION_OFFPATH_DETECTION;
		hdr.bridge.mirror = 1;
	}

	table forward_to_offpath_table {
		key = {
			forward_to_offpath_table_match_ip : exact;
		}
		actions = {
			@defaultonly NoAction;
			forward_to_offpath_table_action;
		}
		const size = DOMAIN_WATCHER_SIZE;
	}

	apply {
		// step 0: common ops
		//extracted_teid = meta.extracted_teid[(MA_ID_BITS  - COMPRESSED_QFI_BITS - 1):0];
		if (hdr.ethernet.etherType == TYPE_ARP) {
			arp_reply(UPF_MAC);
		}
		set_upf_ip_table.apply();
		hdr.bridge.setValid();
		handle_gtpu_control_msg_table.apply();
		if ((hdr.gtpu.messageType == 255 && hdr.gtpu.isValid()) || !hdr.gtpu.isValid()) { // only do PDR match and action if not GTP-U control messages
			// step 1: match
			pdr.apply(hdr, meta, meta.extracted_teid);
			// step 2: match MA_ID to do actions
			if (meta.uplink) {
				//if (ul_no_mark_tos_table.apply().miss) {
					if (ul_mark_tos_table.apply().miss) {
						ul_to_N3N9_table.apply();
					}
				//}
				forward_to_offpath_table_match_ip = hdr.overlay_ipv4.srcAddr;
				hdr.bridge.is_downlink = 0;
			} else {
				if (dl_to_N3N9_table.apply().miss) {
					//dl_to_drop_table.apply();
				}
				forward_to_offpath_table_match_ip = hdr.overlay_ipv4.dstAddr;
				hdr.bridge.is_downlink = 1;
			}
		}
		// step 2: AMBR enforcement
		bitrate_enforce_table.apply();
		if (meta.individual_gbr_meter_color == MeterColor_t.RED || do_drop) {
			//mark_for_drop();
		} else if (meta.individual_gbr_meter_color == MeterColor_t.GREEN || hdr.gtpu.messageType != 255) {
			ig_tm_md.qid = 0;
		} else {
			qfi_to_queue_table.apply();
		}
		// step 2: set_bridge_header
		set_bridge_header();
		// step 2: ACL
		acl.apply(hdr, meta, ig_intr_md, ig_prsr_md, ig_dprsr_md, ig_tm_md);
		// step 3: NoCP and Buffer (send to CPU)
		send_to_cpu_table.apply();
		forward_to_offpath_table.apply();
		if (do_buffer) {
			ig_tm_md.ucast_egress_port = PORT_BUFFER;
		} else {
			// step 3: IP routing
			if (hdr.underlay_ipv4.isValid()) {
				ipv4_routing_underlay.apply(hdr, hdr.underlay_ipv4.dstAddr, ig_tm_md);
			} else if (hdr.overlay_ipv4.isValid()) {
				ipv4_routing_overlay.apply(hdr, hdr.overlay_ipv4.dstAddr, ig_tm_md);
			}
		}
	}
}

control IngressDeparser(
			packet_out                                  pkt,
	/* User */
	inout   ig_header_t                                 hdr,
	in      ig_metadata_t                               meta,
	/* Intrinsic */
	in      ingress_intrinsic_metadata_for_deparser_t   ig_intr_dprsr_md
) {
	Checksum() underlay_ip_checksum;
	Checksum() overlay_ip_checksum;
	apply {
		if (hdr.underlay_ipv4.isValid()) {
			hdr.underlay_ipv4.hdrChecksum = underlay_ip_checksum.update(
				{
					hdr.underlay_ipv4.version,
					hdr.underlay_ipv4.ihl,
					hdr.underlay_ipv4.diffserv,
					hdr.underlay_ipv4.totalLen,
					hdr.underlay_ipv4.identification,
					hdr.underlay_ipv4.flags,
					hdr.underlay_ipv4.fragOffset,
					hdr.underlay_ipv4.ttl,
					hdr.underlay_ipv4.protocol,
					hdr.underlay_ipv4.srcAddr,
					hdr.underlay_ipv4.dstAddr
				}
			);
		}
		if (hdr.overlay_ipv4.isValid()) {
			hdr.overlay_ipv4.hdrChecksum = overlay_ip_checksum.update(
				{
					hdr.overlay_ipv4.version,
					hdr.overlay_ipv4.ihl,
					hdr.overlay_ipv4.diffserv,
					hdr.overlay_ipv4.totalLen,
					hdr.overlay_ipv4.identification,
					hdr.overlay_ipv4.flags,
					hdr.overlay_ipv4.fragOffset,
					hdr.overlay_ipv4.ttl,
					hdr.overlay_ipv4.protocol,
					hdr.overlay_ipv4.srcAddr,
					hdr.overlay_ipv4.dstAddr
				}
			);
		}
		pkt.emit(hdr.bridge);
		pkt.emit(hdr.ethernet);
		pkt.emit(hdr.arp);
		pkt.emit(hdr.cpu_header);
		pkt.emit(hdr.detection_trigger);
		pkt.emit(hdr.underlay_ipv4);
		pkt.emit(hdr.underlay_ipv6);
		pkt.emit(hdr.underlay_udp);
		pkt.emit(hdr.gtpu);
		pkt.emit(hdr.gtpu_optional);
		pkt.emit(hdr.gtpu_ext_psc);
		pkt.emit(hdr.gtpu_ext_psc_optional);
		pkt.emit(hdr.gtpu_ext_psc_next_header_type);
		pkt.emit(hdr.overlay_ipv4);
		pkt.emit(hdr.overlay_ipv6);
		pkt.emit(hdr.overlay_tcp);
		pkt.emit(hdr.overlay_udp);

        pkt.emit(hdr.dns);
        pkt.emit(hdr.len4);
        
        pkt.emit(hdr.label4_16);
        pkt.emit(hdr.label4_8);
        pkt.emit(hdr.label4_4);
        pkt.emit(hdr.label4_2);
        pkt.emit(hdr.label4_1);
        pkt.emit(hdr.len3);
        
        pkt.emit(hdr.label3_16);
        pkt.emit(hdr.label3_8);
        pkt.emit(hdr.label3_4);
        pkt.emit(hdr.label3_2);
        pkt.emit(hdr.label3_1);
        pkt.emit(hdr.len2);
        
        pkt.emit(hdr.label2_16);
        pkt.emit(hdr.label2_8);
        pkt.emit(hdr.label2_4);
        pkt.emit(hdr.label2_2);
        pkt.emit(hdr.label2_1);
        pkt.emit(hdr.len1);
        
        pkt.emit(hdr.label1_16);
        pkt.emit(hdr.label1_8);
        pkt.emit(hdr.label1_4);
        pkt.emit(hdr.label1_2);
        pkt.emit(hdr.label1_1);
        pkt.emit(hdr.dns_extra);
	}
}

// ------------------------------------------------------------------------------------------
//                                      EGRESS STAGE
// ------------------------------------------------------------------------------------------

parser EgressParser(
			packet_in                                   pkt,
	/* User */
	out     eg_headers_t                                hdr,
	out     eg_metadata_t                               meta,
	/* Intrinsic */
	out     egress_intrinsic_metadata_t                 eg_intr_md
)
{
    ParserCounter()         len_limit;
	internal_header_h inthdr;
	
	/* This is a mandatory state, required by Tofino Architecture */
	state start {
		meta.flow_key_reporting_packet = false;
		meta.offpath_detection_packet = false;
		pkt.extract(eg_intr_md);
		inthdr = pkt.lookahead<internal_header_h>();
		transition select(inthdr.header_type, inthdr.header_info) {
			(HEADER_TYPE_BRIDGE   ,                              _) : parse_bridge;
			(HEADER_TYPE_EG_MIRROR, HEADER_INFO_FLOW_KEY_REPORTING) : extract_eg_mirror_flow_key_reporting;
			(HEADER_TYPE_EG_MIRROR, HEADER_INFO_OFFPATH_DETECTION) : extract_eg_mirror_offpath_detection;
			default: reject;
		}
	}

	state parse_bridge {
		pkt.extract(meta.bridge);
		transition parse_ethernet;
	}

	state parse_ethernet {
		pkt.extract(hdr.ethernet);
		bit<8> nextHdr = pkt.lookahead<bit<8>>();
		transition select(nextHdr) {
			CPU_HEADER_MAGIC: parse_cpu_header;
			default: parse_ip_gateway;
		}
	}

	state parse_ip_gateway {
		transition select(meta.bridge.is_downlink) {
			1: parse_underlay;
			0: parse_overlay;
		}
	}


	state extract_eg_mirror_flow_key_reporting {
		pkt.extract<eg_mirror_header_flow_key_reporting_h>(_);
		meta.flow_key_reporting_packet = true;
		transition accept;
	}

	state extract_eg_mirror_offpath_detection {
		pkt.extract<eg_mirror_header_offpath_detection_h>(_);
		meta.offpath_detection_packet = true;
		transition accept;
	}

	state parse_ethernet_no_bridge {
		pkt.extract(hdr.ethernet);
		transition parse_underlay;
	}

	state parse_cpu_header {
		pkt.extract(hdr.cpu_header);
		transition parse_ip_gateway;
	}

    state parse_detection_trigger {
        pkt.extract(hdr.detection_trigger);
        transition accept;
    }
	
    state parse_underlay {
		transition select(hdr.ethernet.etherType) {
			0x0806: parse_arp;
			0x0800: parse_underlay_ipv4;
			DETECTION_TRIGGER:  parse_detection_trigger;
			default: reject;
		}
	}

	state parse_arp {
		pkt.extract(hdr.arp);
		transition accept;
	}

	state parse_underlay_ipv4 {
		pkt.extract(hdr.underlay_ipv4);
		transition select(hdr.underlay_ipv4.ihl) {
				  5 : parse_underlay_ipv4_no_options;
			default : reject;
		}
	}

	state parse_underlay_ipv4_no_options {
		transition select(hdr.underlay_ipv4.protocol) {
			17: parse_underlay_udp;
			default: reject;
		}
	}


	state parse_underlay_udp {
		pkt.extract(hdr.underlay_udp);
		transition select(hdr.underlay_udp.dstPort) {
			2152: parse_gtpu;
			default: reject;
		}
	}

	state parse_gtpu {
		pkt.extract(hdr.gtpu);
		transition select(hdr.gtpu.extensionHeaderFlag, hdr.gtpu.sequenceNumberFlag, hdr.gtpu.npduNumberFlag) {
			(0, 0, 0): parse_overlay_gateway;
			default: parse_gtpu_optional;
		}
	}

	state parse_gtpu_optional {
		pkt.extract(hdr.gtpu_optional);
		transition select(hdr.gtpu_optional.nextExtensionHeaderType) {
			8w0b10000101: parse_gtpu_psc;
			default: reject; // not handled we can only reject
		}
	}

	state parse_gtpu_psc {
		pkt.extract(hdr.gtpu_ext_psc);
		transition select(hdr.gtpu_ext_psc.extHdrLength) {
			0: reject;
			1: parse_gtpu_psc_optional_1;
			2: parse_gtpu_psc_optional_2;
			3: parse_gtpu_psc_optional_3;
			4: parse_gtpu_psc_optional_4;
			5: parse_gtpu_psc_optional_5;
			6: parse_gtpu_psc_optional_6;
			7: parse_gtpu_psc_optional_7;
			8: parse_gtpu_psc_optional_8;
			9: parse_gtpu_psc_optional_9;
			10: parse_gtpu_psc_optional_10;
			11: parse_gtpu_psc_optional_11;
			12: parse_gtpu_psc_optional_12;
			default: reject;
		}
	}

	state parse_gtpu_psc_optional_1 {
		// skip 8 bit nextHdr
		pkt.advance(8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_2 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (2 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_3 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (3 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_4 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (4 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_5 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (5 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_6 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (6 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_7 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (7 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_8 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (8 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_9 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (9 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_10 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (10 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_11 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (11 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}
	state parse_gtpu_psc_optional_12 {
		pkt.extract(hdr.gtpu_ext_psc_optional, (12 * 4 - 3) * 8);
		transition parse_overlay_gateway;
	}

	state parse_overlay_gateway {
		// handle cases where no more data after GTP-U mandatory header
		transition select(hdr.gtpu.messageType) {
			255: parse_overlay; // G-PDU type with overlay data
			default: accept; // other GTP-U control messages
		}
	}

	state parse_overlay {
		transition parse_overlay_ipv4;
		// bit<4> ip_ver = pkt.lookahead<bit<4>>();
		// transition select(ip_ver) {
		// 	4w4: parse_overlay_ipv4;
		// 	default: reject; // other L3 protocol not supported
		// }
	}

	state parse_overlay_ipv4 {
		pkt.extract(hdr.overlay_ipv4);
		transition select(hdr.overlay_ipv4.ihl) {
				  5 : parse_overlay_ipv4_no_options;
			default : reject; // invalid IHL value
		}
	}

	state parse_overlay_ipv4_no_options {
		transition select(hdr.overlay_ipv4.protocol) {
			6: parse_overlay_tcp;
			17: parse_overlay_udp;
			default: accept; // ICMP and other L4 protocols
		}
	}

	state parse_overlay_udp {
		pkt.extract(hdr.overlay_udp);
		transition select(hdr.overlay_udp.srcPort, hdr.overlay_udp.dstPort) {
            (_,53) :      parse_dns_h;
            //(53,_) :      parse_dns_h;
            default :   accept;
        }
	}

	state parse_overlay_tcp {
		pkt.extract(hdr.overlay_tcp);
		transition accept; // Done
	}


    state parse_dns_h {
        pkt.extract(hdr.dns);
        len_limit.set(8w60);

        transition parse_len1;
    }

    
    state parse_len1 {
        bit<8> len = pkt.lookahead<bit<8>>();
        len_limit.decrement(8w1);

        transition select(len) {
            1 .. 31:           preparse_label1;
            0:                                      postparse_dns;
            default:                                accept;
        }
    }
    
    state parse_len2 {
        bit<8> len = pkt.lookahead<bit<8>>();
        len_limit.decrement(8w1);

        transition select(len) {
            1 .. 31:           preparse_label2;
            0:                                      postparse_dns;
            default:                                accept;
        }
    }
    
    state parse_len3 {
        bit<8> len = pkt.lookahead<bit<8>>();
        len_limit.decrement(8w1);

        transition select(len) {
            1 .. 31:           preparse_label3;
            0:                                      postparse_dns;
            default:                                accept;
        }
    }
    
    state parse_len4 {
        bit<8> len = pkt.lookahead<bit<8>>();
        len_limit.decrement(8w1);

        transition select(len) {
            1 .. 31:           preparse_label4;
            0:                                      postparse_dns;
            default:                                accept;
        }
    }
    
    state parse_len5 {
        bit<8> len = pkt.lookahead<bit<8>>();
        len_limit.decrement(8w1);

        transition select(len) {
            
            0:                                      postparse_dns;
            default:                                accept;
        }
    }
    

    
    state preparse_label1 {
        pkt.extract(hdr.len1);
        
        transition select(hdr.len1.l) { 
            16 &&& 16:          parse_label1_16;
            8 &&& 8:          parse_label1_8;
            4 &&& 4:          parse_label1_4;
            2 &&& 2:          parse_label1_2;
            1 &&& 1:          parse_label1_1;
        }
    }

    state postparse_label1 {
        transition select(len_limit.is_negative()) {
            false:                    parse_len2;
            true:                     accept;
        }
    }
    
    state parse_label1_16 {
        pkt.extract(hdr.label1_16);
        len_limit.decrement(8w16);

        transition select(hdr.len1.l) { 
            8 &&& 8:          parse_label1_8;
            4 &&& 4:          parse_label1_4;
            2 &&& 2:          parse_label1_2;
            1 &&& 1:          parse_label1_1;
            default:                  postparse_label1;
        }
    }
    
    state parse_label1_8 {
        pkt.extract(hdr.label1_8);
        len_limit.decrement(8w8);

        transition select(hdr.len1.l) { 
            4 &&& 4:          parse_label1_4;
            2 &&& 2:          parse_label1_2;
            1 &&& 1:          parse_label1_1;
            default:                  postparse_label1;
        }
    }
    
    state parse_label1_4 {
        pkt.extract(hdr.label1_4);
        len_limit.decrement(8w4);

        transition select(hdr.len1.l) { 
            2 &&& 2:          parse_label1_2;
            1 &&& 1:          parse_label1_1;
            default:                  postparse_label1;
        }
    }
    
    state parse_label1_2 {
        pkt.extract(hdr.label1_2);
        len_limit.decrement(8w2);

        transition select(hdr.len1.l) { 
            1 &&& 1:          parse_label1_1;
            default:                  postparse_label1;
        }
    }
    
    state parse_label1_1 {
        pkt.extract(hdr.label1_1);
        len_limit.decrement(8w1);

        transition postparse_label1;
    }
    
    state preparse_label2 {
        pkt.extract(hdr.len2);
        
        transition select(hdr.len2.l) { 
            16 &&& 16:          parse_label2_16;
            8 &&& 8:          parse_label2_8;
            4 &&& 4:          parse_label2_4;
            2 &&& 2:          parse_label2_2;
            1 &&& 1:          parse_label2_1;
        }
    }

    state postparse_label2 {
        transition select(len_limit.is_negative()) {
            false:                    parse_len3;
            true:                     accept;
        }
    }
    
    state parse_label2_16 {
        pkt.extract(hdr.label2_16);
        len_limit.decrement(8w16);

        transition select(hdr.len2.l) { 
            8 &&& 8:          parse_label2_8;
            4 &&& 4:          parse_label2_4;
            2 &&& 2:          parse_label2_2;
            1 &&& 1:          parse_label2_1;
            default:                  postparse_label2;
        }
    }
    
    state parse_label2_8 {
        pkt.extract(hdr.label2_8);
        len_limit.decrement(8w8);

        transition select(hdr.len2.l) { 
            4 &&& 4:          parse_label2_4;
            2 &&& 2:          parse_label2_2;
            1 &&& 1:          parse_label2_1;
            default:                  postparse_label2;
        }
    }
    
    state parse_label2_4 {
        pkt.extract(hdr.label2_4);
        len_limit.decrement(8w4);

        transition select(hdr.len2.l) { 
            2 &&& 2:          parse_label2_2;
            1 &&& 1:          parse_label2_1;
            default:                  postparse_label2;
        }
    }
    
    state parse_label2_2 {
        pkt.extract(hdr.label2_2);
        len_limit.decrement(8w2);

        transition select(hdr.len2.l) { 
            1 &&& 1:          parse_label2_1;
            default:                  postparse_label2;
        }
    }
    
    state parse_label2_1 {
        pkt.extract(hdr.label2_1);
        len_limit.decrement(8w1);

        transition postparse_label2;
    }
    
    state preparse_label3 {
        pkt.extract(hdr.len3);
        
        transition select(hdr.len3.l) { 
            16 &&& 16:          parse_label3_16;
            8 &&& 8:          parse_label3_8;
            4 &&& 4:          parse_label3_4;
            2 &&& 2:          parse_label3_2;
            1 &&& 1:          parse_label3_1;
        }
    }

    state postparse_label3 {
        transition select(len_limit.is_negative()) {
            false:                    parse_len4;
            true:                     accept;
        }
    }
    
    state parse_label3_16 {
        pkt.extract(hdr.label3_16);
        len_limit.decrement(8w16);

        transition select(hdr.len3.l) { 
            8 &&& 8:          parse_label3_8;
            4 &&& 4:          parse_label3_4;
            2 &&& 2:          parse_label3_2;
            1 &&& 1:          parse_label3_1;
            default:                  postparse_label3;
        }
    }
    
    state parse_label3_8 {
        pkt.extract(hdr.label3_8);
        len_limit.decrement(8w8);

        transition select(hdr.len3.l) { 
            4 &&& 4:          parse_label3_4;
            2 &&& 2:          parse_label3_2;
            1 &&& 1:          parse_label3_1;
            default:                  postparse_label3;
        }
    }
    
    state parse_label3_4 {
        pkt.extract(hdr.label3_4);
        len_limit.decrement(8w4);

        transition select(hdr.len3.l) { 
            2 &&& 2:          parse_label3_2;
            1 &&& 1:          parse_label3_1;
            default:                  postparse_label3;
        }
    }
    
    state parse_label3_2 {
        pkt.extract(hdr.label3_2);
        len_limit.decrement(8w2);

        transition select(hdr.len3.l) { 
            1 &&& 1:          parse_label3_1;
            default:                  postparse_label3;
        }
    }
    
    state parse_label3_1 {
        pkt.extract(hdr.label3_1);
        len_limit.decrement(8w1);

        transition postparse_label3;
    }
    
    state preparse_label4 {
        pkt.extract(hdr.len4);
        
        transition select(hdr.len4.l) { 
            16 &&& 16:          parse_label4_16;
            8 &&& 8:          parse_label4_8;
            4 &&& 4:          parse_label4_4;
            2 &&& 2:          parse_label4_2;
            1 &&& 1:          parse_label4_1;
        }
    }

    state postparse_label4 {
        transition select(len_limit.is_negative()) {
            false:                    parse_len5;
            true:                     accept;
        }
    }
    
    state parse_label4_16 {
        pkt.extract(hdr.label4_16);
        len_limit.decrement(8w16);

        transition select(hdr.len4.l) { 
            8 &&& 8:          parse_label4_8;
            4 &&& 4:          parse_label4_4;
            2 &&& 2:          parse_label4_2;
            1 &&& 1:          parse_label4_1;
            default:                  postparse_label4;
        }
    }
    
    state parse_label4_8 {
        pkt.extract(hdr.label4_8);
        len_limit.decrement(8w8);

        transition select(hdr.len4.l) { 
            4 &&& 4:          parse_label4_4;
            2 &&& 2:          parse_label4_2;
            1 &&& 1:          parse_label4_1;
            default:                  postparse_label4;
        }
    }
    
    state parse_label4_4 {
        pkt.extract(hdr.label4_4);
        len_limit.decrement(8w4);

        transition select(hdr.len4.l) { 
            2 &&& 2:          parse_label4_2;
            1 &&& 1:          parse_label4_1;
            default:                  postparse_label4;
        }
    }
    
    state parse_label4_2 {
        pkt.extract(hdr.label4_2);
        len_limit.decrement(8w2);

        transition select(hdr.len4.l) { 
            1 &&& 1:          parse_label4_1;
            default:                  postparse_label4;
        }
    }
    
    state parse_label4_1 {
        pkt.extract(hdr.label4_1);
        len_limit.decrement(8w1);

        transition postparse_label4;
    }
    

    state postparse_dns {
        pkt.extract(hdr.dns_extra);

        transition accept;
    }
}


// --------------------------------------------------
//                  Egress Control
// --------------------------------------------------
control Egress(
	/* User */
	inout   eg_headers_t                                hdr,
	inout   eg_metadata_t                               meta,
	/* Intrinsic */
	in      egress_intrinsic_metadata_t                 eg_intr_md,
	in      egress_intrinsic_metadata_from_parser_t     eg_intr_md_from_prsr,
	inout   egress_intrinsic_metadata_for_deparser_t    eg_intr_dprs_md,
	inout   egress_intrinsic_metadata_for_output_port_t eg_intr_oport_md
) {
	Accounting(27) accounting;

	Hash<bit<32>>(HashAlgorithm_t.CRC32)    domain_hash_func_1;
    Hash<bit<32>>(HashAlgorithm_t.CRC32)    domain_hash_func_1_0;
	Hash<bit<32>>(HashAlgorithm_t.CRC32)    domain_hash_func_2;
    Hash<bit<32>>(HashAlgorithm_t.CRC32)    domain_hash_func_2_0;
    Hash<bit<32>>(HashAlgorithm_t.CRC32)    domain_hash_func_3;
    Hash<bit<32>>(HashAlgorithm_t.CRC32)    domain_hash_func_3_0;
    
    bit<32>                                 domain_hash;
    bool                                    domain_labels_2 = false;
    
    action dns_split_labels_action_2() {
        domain_labels_2 = true;
    }
    
    table dns_domain_parts_2 {
        key = { 
            hdr.label1_16.l: exact;
            hdr.label2_16.l: exact;
            hdr.label1_8.l: exact;
            hdr.label2_8.l: exact;
            hdr.label1_4.l: exact;
            hdr.label2_4.l: exact;
            hdr.label1_2.l: exact;
            hdr.label2_2.l: exact;
            hdr.label1_1.l: exact;
            hdr.label2_1.l: exact;
        }
        actions = {
            dns_split_labels_action_2;
            NoAction;            
        }
        size = 12800;
        default_action = NoAction;
    }

    
    action calc_domain_3() {
        domain_hash = domain_hash_func_3.get({  hdr.label3_8.l, hdr.label3_4.l, hdr.label3_2.l, hdr.label3_1.l, hdr.label2_8.l, hdr.label2_4.l, hdr.label2_2.l, hdr.label2_1.l, hdr.label1_8.l, hdr.label1_4.l, hdr.label1_2.l, hdr.label1_1.l });
    }
    
    action calc_domain_2() {
        domain_hash = domain_hash_func_2.get({  hdr.label2_8.l, hdr.label2_4.l, hdr.label2_2.l, hdr.label2_1.l, hdr.label1_8.l, hdr.label1_4.l, hdr.label1_2.l, hdr.label1_1.l });
    }
    
    action calc_domain_1() {
        domain_hash = domain_hash_func_1.get({  hdr.label1_8.l, hdr.label1_4.l, hdr.label1_2.l, hdr.label1_1.l });
    }

    action calc_domain_3_extra() {
        domain_hash = domain_hash + domain_hash_func_3_0.get({  hdr.label3_16.l, hdr.label2_16.l, hdr.label1_16.l });
    }

    action calc_domain_2_extra() {
        domain_hash = domain_hash + domain_hash_func_2_0.get({  hdr.label2_16.l, hdr.label1_16.l });
    }

    action calc_domain_1_extra() {
        domain_hash = domain_hash + domain_hash_func_1_0.get({  hdr.label1_16.l });
    }
    
	score_t score1 = 0;
	score_t acc_score_1 = 0;
	domain_watcher_id_t detection_id = 0;
	bit<2>              model_id = 0;

	action set_domain_scores(score_t s1) {
		score1 = s1;
	}


	table domain_hash2score {
		key = {
			model_id     : exact;
			domain_hash  : exact;
		}
		actions = {
			set_domain_scores;
		}
		const size = DOMAIN_WATCHER_DOMAINS * DOMAIN_WATCHER_MODELS;
	}

	action set_domain_scores_UNKNOWN(score_t s1) {
		score1 = s1;
	}
	table unknown_domain_score {
		key = {
			model_id : exact;
		}
		actions = {
			set_domain_scores_UNKNOWN;
		}
		const size = DOMAIN_WATCHER_MODELS;
	}

	Register<score_t, domain_watcher_id_t>(DOMAIN_WATCHER_SIZE)               domain_scores_1;
	RegisterAction<score_t, domain_watcher_id_t, score_t>(domain_scores_1)    update_and_read_domain_scores_1 = {
        void apply(inout score_t value, out score_t res){
			value = value |+| score1;
            res = value;
        }
    };
	
	// domainwatcher
	action set_domain_watcher_detection_id(domain_watcher_id_t detection_id2, bit<2> model_id2) {
		detection_id = detection_id2;
		model_id = model_id2;
	}

	table set_domainwatcher_id {
		key = {
			meta.bridge.ma_id : exact;
		}
		actions = {
			set_domain_watcher_detection_id;
			@defaultonly NoAction;
		}
		const default_action = NoAction();
		const size = DOMAIN_WATCHER_SIZE;
	}

	apply {
		if (eg_intr_md.egress_port != PORT_CPU && eg_intr_md.egress_port != PORT_BUFFER && !meta.offpath_detection_packet) {
			// for packets not going to CPU
			hdr.cpu_header.setInvalid();
			if (hdr.gtpu_ext_psc.isValid()) {
				hdr.gtpu_ext_psc_next_header_type.setValid();
				hdr.gtpu_ext_psc_next_header_type.content = 0;
			}
			// -----------------------------------------------------------------
			// step 2 : Post QoS Accounting
			accounting.apply(meta.bridge.ma_id);

			// XGB
			dns_domain_parts_2.apply();
			
			if (hdr.dns.isValid()) {
				if (set_domainwatcher_id.apply().hit) {

					// //if (domain_labels_2 == true) {
					// if (hdr.len3.isValid()) {
					// 	calc_domain_3();
					// 	calc_domain_3_extra();
					// } else if (hdr.len2.isValid()) {
					// 	calc_domain_2();
					// 	calc_domain_2_extra();
					// } else {
					// 	calc_domain_1();
					// 	calc_domain_1_extra();
					// }


					if (domain_labels_2 == true && hdr.len3.isValid()) {
						calc_domain_3();
						calc_domain_3_extra();
					} else if (hdr.len2.isValid()) {
						calc_domain_2();
						calc_domain_2_extra();
					} else {
						calc_domain_1();
						calc_domain_1_extra();
					}
					if (domain_hash2score.apply().miss) {
						// apply scores for <UNKNOWN>
						unknown_domain_score.apply();
					}
					update_and_read_domain_scores_1.execute(detection_id);
				}
			}

			if (meta.bridge.mirror != 0) {
				eg_intr_dprs_md.mirror_type = EG_MIRROR_TYPE_SIMPLE_COPY;
				meta.mirror_session = MIRROR_SESSION_OFFPATH_DETECTION;
				meta.mirror_hdr_type = HEADER_TYPE_EG_MIRROR;
				meta.mirror_hdr_info = HEADER_INFO_OFFPATH_DETECTION;
			}

		}
	}
}


control EgressDeparser(
			packet_out                                  pkt,
	/* User */
	inout   eg_headers_t                                hdr,
	in      eg_metadata_t                               meta,
	/* Intrinsic */
	in      egress_intrinsic_metadata_for_deparser_t    eg_dprsr_md
) {
	Mirror() offpath_mirror;
	apply {
		if (eg_dprsr_md.mirror_type == EG_MIRROR_TYPE_SIMPLE_COPY) {
			offpath_mirror.emit<eg_mirror_header_offpath_detection_h>(
				meta.mirror_session,
				{
					meta.mirror_hdr_type,
					meta.mirror_hdr_info
				}
			);
		}

		pkt.emit(hdr.ethernet);
		pkt.emit(hdr.arp);
		pkt.emit(hdr.cpu_header);
		pkt.emit(hdr.detection_trigger);
		pkt.emit(hdr.underlay_ipv4);
		pkt.emit(hdr.underlay_ipv6);
		pkt.emit(hdr.underlay_udp);
		pkt.emit(hdr.gtpu);
		pkt.emit(hdr.gtpu_optional);
		pkt.emit(hdr.gtpu_ext_psc);
		pkt.emit(hdr.gtpu_ext_psc_optional);
		pkt.emit(hdr.gtpu_ext_psc_next_header_type);
		pkt.emit(hdr.overlay_ipv4);
		pkt.emit(hdr.overlay_ipv6);
		pkt.emit(hdr.overlay_tcp);
		pkt.emit(hdr.overlay_udp);

        pkt.emit(hdr.dns);

        
        pkt.emit(hdr.len4);
        
        pkt.emit(hdr.label4_16);
        pkt.emit(hdr.label4_8);
        pkt.emit(hdr.label4_4);
        pkt.emit(hdr.label4_2);
        pkt.emit(hdr.label4_1);
        pkt.emit(hdr.len3);
        
        pkt.emit(hdr.label3_16);
        pkt.emit(hdr.label3_8);
        pkt.emit(hdr.label3_4);
        pkt.emit(hdr.label3_2);
        pkt.emit(hdr.label3_1);
        pkt.emit(hdr.len2);
        
        pkt.emit(hdr.label2_16);
        pkt.emit(hdr.label2_8);
        pkt.emit(hdr.label2_4);
        pkt.emit(hdr.label2_2);
        pkt.emit(hdr.label2_1);
        pkt.emit(hdr.len1);
        
        pkt.emit(hdr.label1_16);
        pkt.emit(hdr.label1_8);
        pkt.emit(hdr.label1_4);
        pkt.emit(hdr.label1_2);
        pkt.emit(hdr.label1_1);
        pkt.emit(hdr.dns_extra);
	}
}


Pipeline(
	IngressParser(),
	Ingress(),
	IngressDeparser(),
	EgressParser(),
	Egress(),
	EgressDeparser()
) pipe;

Switch(pipe) main;
