#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json, os
import numpy as np
from pyBKT.models import Model
from utils.get_tabular_data import get_exact_scores, get_accumulated_scores, linear_interpolate_a_table
from utils.static import *
import tensorflow as tf
from utils.deep_knowledge_tracing_plus.utils import DKT
from utils.deep_knowledge_tracing_plus.load_data import DKTData

####### get tabular data from handcrafted scoring metrics #######
def get_tabular_data_from_handcrafted_scores():
    ddcts = [get_exact_scores(), get_accumulated_scores(normed=True)]
    names = ['exact_scores', 'accumulated_scores_normed']

    for dd, name in zip(ddcts, names):
        whole_scores = []
        for user, user_dict in dd.items():
            score_table = user_dict['scores']
            user_column = np.full((score_table.shape[0], 1), int(user))
            whole_scores.append(np.concatenate([user_column, score_table], axis=1))
        whole_scores = np.concatenate(whole_scores, axis=0)
        os.makedirs('data/inoutpair/handcraft/', exist_ok=True)
        np.save(f'./data/inoutpair/handcraft/{name}.npy', whole_scores)

####### get tabular data using Bayesian Knowledge Tracing #######
def get_tabular_data_using_BKT():
    # ref: pyBKT https://github.com/CAHLR/pyBKT
    for dirr in ['./data/eachstamp', './data/inoutpair']:
        os.makedirs(os.path.join(dirr, 'bkt'), exist_ok=True)
        model = Model(seed=42, num_fits = 1)
        datapath = os.path.join(dirr, 'bkt_format_data.csv')

        model.fit(data_path=datapath)
        model.params().to_csv(os.path.join(dirr, 'bkt', 'params.csv'))
        preds_df = model.predict(data_path=datapath)
        preds_df.to_csv(os.path.join(dirr, 'bkt', 'preds.csv'), index=False)
        preds_user_group = {df["user_id"].iloc[0]: df for _, df in preds_df.groupby("user_id", as_index=False)}

        whole_arr = []
        for user_id, user_df in preds_user_group.items():
            num_rows_of_user_df = user_df.shape[0]
            raw_scores_table = np.zeros((num_rows_of_user_df, 116))
            real_observed_stamps = {}
            rid = 0
            for _, row in user_df.iterrows():
                cid = int(row["skill_name"])
                score = row["state_predictions"]
                if cid not in real_observed_stamps: real_observed_stamps[cid] = []
                real_observed_stamps[cid].append(rid)
                raw_scores_table[rid, cid] = score
                rid += 1
            user_table = linear_interpolate_a_table(raw_scores_table, real_observed_stamps)
            user_column = np.full((num_rows_of_user_df, 1), user_id)
            user_table = np.concatenate([user_column, user_table], axis=1)
            whole_arr.append(user_table)
        whole_arr = np.concatenate(whole_arr, axis=0)
        np.save(os.path.join(dirr, 'bkt', 'preds.npy'), whole_arr)

####### get tabular data using Deep Knowledge Tracing Plus #######
def get_tabular_data_using_DKT():
    # ref: DKT https://github.com/ckyeungac/deep-knowledge-tracing-plus
    rnn_cells = {
        "LSTM": tf.compat.v1.nn.rnn_cell.LSTMCell,
        "GRU": tf.compat.v1.nn.rnn_cell.GRUCell,
        "BasicRNN": tf.compat.v1.nn.rnn_cell.BasicRNNCell,
    }
    network_config = {}
    network_config['batch_size'] = 32
    network_config['hidden_layer_structure'] = [200, ]
    network_config['learning_rate'] = 1e-2
    network_config['keep_prob'] = 0.5
    network_config['rnn_cell'] = rnn_cells["LSTM"]
    network_config['lambda_w1'] = 0.00
    network_config['lambda_w2'] = 0.00
    network_config['lambda_o'] = 0.00

    num_runs = 5
    num_epochs = 500
    batch_size = 32
    keep_prob = 0.5

    for dirr in ['./data/eachstamp', './data/inoutpair']:
        os.makedirs(os.path.join(dirr, 'dkt'), exist_ok=True)
        datapath = os.path.join(dirr, 'dkt_format_data.csv')
        train_path = test_path = datapath
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)

        data = DKTData(train_path, test_path, batch_size=batch_size)
        data_train = data.train
        data_test = data.test
        num_problems = data.num_problems

        dkt = DKT(sess, data_train, data_test, num_problems, network_config,
                  save_dir_prefix=os.path.join(dirr, 'dkt'),
                  num_runs=num_runs, num_epochs=num_epochs,
                  keep_prob=keep_prob, logging=True, save=True)

        # run optimization of the created model
        dkt.model.build_graph()
        dkt.run_optimization()
        sess.close()

    def _predict(save_dir_prefix, data_path, stored_users, save_path):
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        data = DKTData(data_path, data_path, batch_size=batch_size)
        data_train = data.train
        data_test = data.test
        num_problems = data.num_problems

        dkt_original = DKT(sess, data_train, data_test, num_problems, network_config,
                           num_runs=num_runs, num_epochs=num_epochs,
                           save_dir_prefix=save_dir_prefix,
                           keep_prob=keep_prob, logging=False, save=False)

        dkt_original.model.build_graph()
        dkt_original.load_model()
        problem_seqs_test = dkt_original.data_test.problem_seqs
        correct_seqs_test = dkt_original.data_test.correct_seqs

        whole_array = []
        for i in range(len(problem_seqs_test)):
            user = stored_users[i]
            ttst = dkt_original.get_output_layer([problem_seqs_test[i]], [correct_seqs_test[i]])
            thisarr = np.array(ttst[0])
            user_column = np.full((thisarr.shape[0], 1), user)
            thisarr = np.concatenate((user_column, thisarr), axis=1)
            whole_array.append(thisarr)

        whole_array = np.concatenate(whole_array, axis=0)
        svdir = os.path.dirname(save_path)
        os.makedirs(svdir, exist_ok=True)
        np.save(save_path, whole_array)

    for dirr in ['./data/eachstamp', './data/inoutpair']:
        save_dir_prefix = os.path.join(dirr, 'dkt')
        data_path = os.path.join(dirr, 'dkt_format_data.csv')
        with open(os.path.join(dirr, 'dkt_stored_users.json'), 'r') as f: stored_users = json.load(f)
        save_path = os.path.join(save_dir_prefix, 'preds.npy')
        _predict(save_dir_prefix, data_path, stored_users, save_path)

        # we have trained the KT model based on ground-truth data. Now with some guess that the model already learned
        # something about mutual relationships between constructs - we may try to "fake" some do() operations.
        # with the hypothesis ci->cj, we choose to make ci always correct/incorrect, or always missing,
        # and see how the model will change its prediction on cj (in BKT cross-constructs are independent),
        # while in this DKT case, they are not.
        for cid in range(116):
            for tag in ['allwin', 'alllose', 'removed']:
                with open(data_path, 'r') as f: lines = f.readlines()
                assert len(lines) % 3 == 0
                newstr = ''''''
                for l1, l2, l3 in zip(lines[0::3], lines[1::3], lines[2::3]):
                    len_of_usr = int(l1.strip())
                    constructs_sequence = np.array(eval('[' + l2.strip() + ']'), dtype=int)
                    corrects_sequence = np.array(eval('[' + l3.strip() + ']'), dtype=int)
                    assert len(constructs_sequence) == len(corrects_sequence) == len_of_usr
                    cid_indices = np.where(constructs == cid)[0]
                    if tag == 'allwin': corrects_sequence[cid_indices] = 1
                    elif tag == 'alllose': corrects_sequence[cid_indices] = 0
                    elif tag == 'removed':
                        constructs_sequence[cid_indices] = -1
                        corrects_sequence[cid_indices] = -1

                    newstr += str(len_of_usr) + '\n'
                    newstr += ','.join([str(cid) for cid in constructs_sequence]) + '\n'
                    newstr += ','.join([str(correct) for correct in corrects_sequence]) + '\n'
                tmp_data_path = os.path.join(save_dir_prefix, 'tmp.csv')
                with open(tmp_data_path, 'w') as f: f.write(newstr)
                _predict(save_dir_prefix, tmp_data_path, stored_users,
                         os.path.join(save_dir_prefix, f'preds_{tag}_{cid}.npy'))
                os.system(f'rm -rf {tmp_data_path}')



if __name__ == '__main__':
    get_tabular_data_from_handcrafted_scores()
    get_tabular_data_using_BKT()
    get_tabular_data_using_DKT()