
///////////////////////////////////////////////////////////////////////
//                               XGBoost                             //
///////////////////////////////////////////////////////////////////////


typedef int<16> score_t;
typedef bit<8>  xgb_feature_t;
typedef int<16> xgb_weight_t;

control XGBoostTree(in xgb_feature_t feat, out xgb_weight_t weight) {
    action set_weight(xgb_weight_t w) {
        weight = w;
    }
    table xgb_table {
        key = {
            feat : ternary;
        }
        actions = {
            set_weight;
			@defaultonly NoAction;
        }
		const size = 32;
    }

	apply {
        xgb_table.apply();
	}
}

control XGBoost(in xgb_feature_t feat_tree1, in xgb_feature_t feat_tree2, in xgb_feature_t feat_tree3, in xgb_feature_t feat_tree4, out bit<1> result) {
    xgb_weight_t xgb_result_table_1 = 0;
    xgb_weight_t xgb_result_table_2 = 0;
    xgb_weight_t xgb_result_table_3 = 0;
    xgb_weight_t xgb_result_table_4 = 0;
    
    XGBoostTree() tree_1;
    XGBoostTree() tree_2;
    XGBoostTree() tree_3;
    XGBoostTree() tree_4;

    xgb_weight_t reduce_1_1;
    xgb_weight_t reduce_1_2;

    xgb_weight_t reduce_2_1;

    action set_reduce_1() {
        reduce_1_1 = xgb_result_table_1 + xgb_result_table_2;
        reduce_1_2 = xgb_result_table_3 + xgb_result_table_4;
    }

    action set_reduce_2() {
        reduce_2_1 = reduce_1_1 + reduce_1_2;
    }

    Register<xgb_weight_t, bit<1>>(1)                           comparison;
	RegisterAction<xgb_weight_t, bit<1>, bool>(comparison)    do_comparison = {
        void apply(inout xgb_weight_t threshold, out bool res){
            res = reduce_2_1 > threshold;
        }
    };

	apply {
        tree_1.apply(feat_tree1, xgb_result_table_1);
        tree_2.apply(feat_tree2, xgb_result_table_2);
        tree_3.apply(feat_tree3, xgb_result_table_3);
        tree_4.apply(feat_tree4, xgb_result_table_4);
        
        set_reduce_1();
        set_reduce_2();

        result = (bit<1>)do_comparison.execute(0);
	}
}
