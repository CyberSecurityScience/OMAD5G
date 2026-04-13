

control PDR_builtin_counters(
	/* User */
	inout   ig_header_t                                 hdr,
	inout   ig_metadata_t                               meta,
	in bit<(MA_ID_BITS - COMPRESSED_QFI_BITS)> extracted_teid
) {
	bit<COMPRESSED_QFI_BITS> compressed_qfi = 0;

	DirectCounter<bit<36>>(CounterType_t.PACKETS_AND_BYTES) usage_counters_dl_N6_simple_ipv4;
    action set_ma_id_and_tunnel_dl_N6_simple_ipv4(ma_id_t ma_id_v, teid_t teid_v, qer_id_t qer_id) {
		meta.ma_id = ma_id_v;
        //meta.qfi = qfi_v;
        meta.teid = (teid_t)teid_v;
		meta.qer_id = qer_id;
		usage_counters_dl_N6_simple_ipv4.count();
	}
	table dl_N6_simple_ipv4 {
		key = {
			hdr.overlay_ipv4.dstAddr : exact;
		}
		actions = {
			set_ma_id_and_tunnel_dl_N6_simple_ipv4;
			@defaultonly NoAction;
		}
		counters = usage_counters_dl_N6_simple_ipv4;
		const default_action = NoAction();
		const size = TABLE_SIZE_DL_N6_SIMPLE_IPV4;
	}

	DirectCounter<bit<36>>(CounterType_t.PACKETS_AND_BYTES) usage_counters_dl_N6_complex_ipv4;
    action set_ma_id_and_tunnel_dl_N6_complex_ipv4(ma_id_t ma_id_v, teid_t teid_v, qer_id_t qer_id) {
		meta.ma_id = ma_id_v;
        //meta.qfi = qfi_v;
        meta.teid = (teid_t)teid_v;
		meta.qer_id = qer_id;
		usage_counters_dl_N6_complex_ipv4.count();
	}
	table dl_N6_complex_ipv4 { // require 5 TCAM join
		key = {
			hdr.overlay_ipv4.dstAddr    : ternary;
			hdr.overlay_ipv4.srcAddr    : ternary;
			hdr.overlay_ipv4.protocol   : ternary;
			hdr.overlay_ipv4.diffserv   : ternary;
			meta.l4_src_port            : ternary;
			meta.l4_dst_port            : ternary;
		}
		actions = {
			set_ma_id_and_tunnel_dl_N6_complex_ipv4;
			@defaultonly NoAction;
		}
		counters = usage_counters_dl_N6_complex_ipv4;
		const default_action = NoAction();
		const size = TABLE_SIZE_DL_N6_COMPLEX_IPV4;
	}

	DirectCounter<bit<36>>(CounterType_t.PACKETS_AND_BYTES) usage_counters_dl_N9_simple_ipv4;
    action set_ma_id_and_tunnel_dl_N9_simple_ipv4(ma_id_t ma_id_v, teid_t teid_v, qer_id_t qer_id) {
		meta.ma_id = ma_id_v;
        //meta.qfi = qfi_v;
        meta.teid = (teid_t)teid_v;
		meta.qer_id = qer_id;
		usage_counters_dl_N9_simple_ipv4.count();
	}
	table dl_N9_simple_ipv4 {
		key = {
			hdr.gtpu.teid[23:0]  : exact;
			hdr.gtpu_ext_psc.qfi : exact;
		}
		actions = {
			set_ma_id_and_tunnel_dl_N9_simple_ipv4;
			@defaultonly NoAction;
		}
		counters = usage_counters_dl_N9_simple_ipv4;
		const default_action = NoAction();
		const size = TABLE_SIZE_DL_N9_SIMPLE_IPV4;
	}

	DirectCounter<bit<36>>(CounterType_t.PACKETS_AND_BYTES) usage_counters_ul_N6_simple_ipv4;
    action set_ma_id_ul_N6_simple_ipv4(ma_id_t ma_id_v, qer_id_t qer_id) {
		meta.ma_id = ma_id_v;
		meta.qer_id = qer_id;
		usage_counters_ul_N6_simple_ipv4.count();
	}
	table ul_N6_simple_ipv4 {
		key = {
			hdr.gtpu.teid[23:0]  : exact;
			hdr.gtpu_ext_psc.qfi : exact;
		}
		actions = {
			set_ma_id_ul_N6_simple_ipv4;
			@defaultonly NoAction;
		}
		counters = usage_counters_ul_N6_simple_ipv4;
		const default_action = NoAction();
		const size = TABLE_SIZE_UL_N6_SIMPLE_IPV4;
	}

	DirectCounter<bit<36>>(CounterType_t.PACKETS_AND_BYTES) usage_counters_ul_N6_complex_ipv4;
    action set_ma_id_ul_N6_complex_ipv4(ma_id_t ma_id_v, qer_id_t qer_id) {
		meta.ma_id = ma_id_v;
		meta.qer_id = qer_id;
		usage_counters_ul_N6_complex_ipv4.count();
	}
	table ul_N6_complex_ipv4 { // require 5 TCAM join
		key = {
			hdr.gtpu.teid[23:0]         : exact;
			hdr.gtpu_ext_psc.qfi        : ternary;
			hdr.overlay_ipv4.dstAddr    : ternary;
			hdr.overlay_ipv4.srcAddr    : ternary;
			hdr.overlay_ipv4.protocol   : ternary;
			hdr.overlay_ipv4.diffserv   : ternary;
			meta.l4_src_port            : ternary;
			meta.l4_dst_port            : ternary;
		}
		actions = {
			set_ma_id_ul_N6_complex_ipv4;
			@defaultonly NoAction;
		}
		counters = usage_counters_ul_N6_complex_ipv4;
		const default_action = NoAction();
		const size = TABLE_SIZE_UL_N6_COMPLEX_IPV4;
	}

	DirectCounter<bit<36>>(CounterType_t.PACKETS_AND_BYTES) usage_counters_ul_N9_simple_ipv4;
    action set_ma_id_and_tunnel_ul_N9_simple_ipv4(ma_id_t ma_id_v, teid_t teid_v, qer_id_t qer_id) {
		meta.ma_id = ma_id_v;
        //meta.qfi = qfi_v;
        meta.teid = (teid_t)teid_v;
		meta.qer_id = qer_id;
		usage_counters_ul_N9_simple_ipv4.count();
	}
	table ul_N9_simple_ipv4 {
		key = {
			hdr.gtpu.teid[23:0]  : exact;
			hdr.gtpu_ext_psc.qfi : exact;
		}
		actions = {
			set_ma_id_and_tunnel_ul_N9_simple_ipv4;
			@defaultonly NoAction;
		}
		counters = usage_counters_ul_N9_simple_ipv4;
		const default_action = NoAction();
		const size = TABLE_SIZE_UL_N9_SIMPLE_IPV4;
	}

	DirectCounter<bit<36>>(CounterType_t.PACKETS_AND_BYTES) usage_counters_ul_N9_complex_ipv4;
    action set_ma_id_and_tunnel_ul_N9_complex_ipv4(ma_id_t ma_id_v, teid_t teid_v, qer_id_t qer_id) {
		meta.ma_id = ma_id_v;
        //meta.qfi = qfi_v;
        meta.teid = (teid_t)teid_v;
		meta.qer_id = qer_id;
		usage_counters_ul_N9_complex_ipv4.count();
	}
	table ul_N9_complex_ipv4 { // require 5 TCAM join
		key = {
			hdr.gtpu.teid[23:0]         : exact;
			hdr.gtpu_ext_psc.qfi        : ternary;
			hdr.overlay_ipv4.dstAddr    : ternary;
			hdr.overlay_ipv4.srcAddr    : ternary;
			hdr.overlay_ipv4.protocol   : ternary;
			hdr.overlay_ipv4.diffserv   : ternary;
			meta.l4_src_port            : ternary;
			meta.l4_dst_port            : ternary;
		}
		actions = {
			set_ma_id_and_tunnel_ul_N9_complex_ipv4;
			@defaultonly NoAction;
		}
		counters = usage_counters_ul_N9_complex_ipv4;
		const default_action = NoAction();
		const size = TABLE_SIZE_UL_N9_COMPLEX_IPV4;
	}

    apply {
        if (meta.uplink) {
			// UL
			if (hdr.overlay_ipv4.isValid()) {
				if (ul_N6_complex_ipv4.apply().miss) {
					if (ul_N6_simple_ipv4.apply().miss) {
						if (ul_N9_complex_ipv4.apply().miss) {
							if (ul_N9_simple_ipv4.apply().miss) {
							}
						}
					}
				}
			}
		} else {
			// DL
			if (hdr.underlay_ipv4.isValid()) {
				// from N9
				if (hdr.underlay_ipv4.isValid()) {
					dl_N9_simple_ipv4.apply();
				}
			} else {
				// from N6
				if (hdr.overlay_ipv4.isValid()) {
					if (dl_N6_complex_ipv4.apply().miss) {
						if (dl_N6_simple_ipv4.apply().miss) {
						}
					}
				}
			}
		}
    }
}

